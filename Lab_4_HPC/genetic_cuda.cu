#include "genetic_cuda.cuh"

#include <iostream>
#include <vector>
#include <cmath>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

using namespace std;

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t error = call;                                            \
        if (error != cudaSuccess) {                                          \
            cerr << "CUDA error: " << cudaGetErrorString(error)              \
                 << " at line " << __LINE__ << endl;                        \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

struct DeviceIndividual {
    double coeffs[COEFF_COUNT];
    double fitness;
};

struct FitnessComparator {
    __host__ __device__
        bool operator()(const DeviceIndividual& a, const DeviceIndividual& b) const {
        return a.fitness < b.fitness;
    }
};

__global__ void setupCurandKernel(
    curandState* states,
    unsigned long long seed,
    int stateCount
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < stateCount) {
        curand_init(seed, index, 0, &states[index]);
    }
}

__global__ void calculateFitnessKernel(
    DeviceIndividual* population,
    const double* pointX,
    const double* pointY,
    int populationSize,
    int pointCount
) {
    int individualIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (individualIndex >= populationSize) {
        return;
    }

    double sumSquaredError = 0.0;

    for (int pointIndex = 0; pointIndex < pointCount; ++pointIndex) {
        double x = pointX[pointIndex];

        double predictedY = 0.0;
        double power = 1.0;

        for (int coeffIndex = 0; coeffIndex < COEFF_COUNT; ++coeffIndex) {
            predictedY += population[individualIndex].coeffs[coeffIndex] * power;
            power *= x;
        }

        double error = predictedY - pointY[pointIndex];
        sumSquaredError += error * error;
    }

    population[individualIndex].fitness = sumSquaredError;
}

__global__ void crossoverKernel(
    DeviceIndividual* population,
    curandState* states,
    int populationSize
) {
    int childIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int parentCount = populationSize / 2;

    if (childIndex >= parentCount) {
        return;
    }

    int targetIndex = parentCount + childIndex;

    if (targetIndex >= populationSize) {
        return;
    }

    curandState localState = states[targetIndex];

    int parent1Index = static_cast<int>(curand_uniform_double(&localState) * parentCount);
    int parent2Index = static_cast<int>(curand_uniform_double(&localState) * parentCount);

    if (parent1Index >= parentCount) {
        parent1Index = parentCount - 1;
    }

    if (parent2Index >= parentCount) {
        parent2Index = parentCount - 1;
    }

    DeviceIndividual child;

    for (int coeffIndex = 0; coeffIndex < COEFF_COUNT; ++coeffIndex) {
        double randomValue = curand_uniform_double(&localState);

        if (randomValue < 0.5) {
            child.coeffs[coeffIndex] = population[parent1Index].coeffs[coeffIndex];
        }
        else {
            child.coeffs[coeffIndex] = population[parent2Index].coeffs[coeffIndex];
        }
    }

    child.fitness = 0.0;

    population[targetIndex] = child;
    states[targetIndex] = localState;
}

__global__ void generateMutationCountKernel(
    curandState* states,
    int* mutationCount,
    double Em,
    double Dm,
    int maxMutationCount
) {
    curandState localState = states[0];

    double stdDev = sqrt(Dm);
    double randomNormal = curand_normal_double(&localState);
    int generatedCount = static_cast<int>(round(Em + stdDev * randomNormal));

    if (generatedCount < 0) {
        generatedCount = 0;
    }

    if (generatedCount > maxMutationCount) {
        generatedCount = maxMutationCount;
    }

    *mutationCount = generatedCount;
    states[0] = localState;
}

__global__ void mutationKernel(
    DeviceIndividual* population,
    curandState* states,
    const int* mutationCount,
    int populationSize,
    double mutationValueStdDev
) {
    int mutationIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int count = *mutationCount;

    if (mutationIndex >= count) {
        return;
    }

    int totalGenes = populationSize * COEFF_COUNT;
    int mutableGenes = totalGenes - COEFF_COUNT;

    curandState localState = states[mutationIndex + 1];

    int randomGeneOffset = static_cast<int>(curand_uniform_double(&localState) * mutableGenes);

    if (randomGeneOffset >= mutableGenes) {
        randomGeneOffset = mutableGenes - 1;
    }

    int globalGeneIndex = COEFF_COUNT + randomGeneOffset;

    int individualIndex = globalGeneIndex / COEFF_COUNT;
    int coeffIndex = globalGeneIndex % COEFF_COUNT;

    double mutationValue = curand_normal_double(&localState) * mutationValueStdDev;

    population[individualIndex].coeffs[coeffIndex] += mutationValue;
    population[individualIndex].fitness = 0.0;

    states[mutationIndex + 1] = localState;
}

static vector<DeviceIndividual> convertToDeviceIndividuals(
    const vector<Individual>& population
) {
    vector<DeviceIndividual> deviceReadyPopulation(population.size());

    for (int i = 0; i < static_cast<int>(population.size()); ++i) {
        for (int j = 0; j < COEFF_COUNT; ++j) {
            deviceReadyPopulation[i].coeffs[j] = population[i].coeffs[j];
        }

        deviceReadyPopulation[i].fitness = population[i].fitness;
    }

    return deviceReadyPopulation;
}

static Individual convertToHostIndividual(
    const DeviceIndividual& deviceIndividual
) {
    Individual individual;

    for (int i = 0; i < COEFF_COUNT; ++i) {
        individual.coeffs[i] = deviceIndividual.coeffs[i];
    }

    individual.fitness = deviceIndividual.fitness;

    return individual;
}

static vector<double> extractPointX(const vector<Point>& points) {
    vector<double> x(points.size());

    for (int i = 0; i < static_cast<int>(points.size()); ++i) {
        x[i] = points[i].x;
    }

    return x;
}

static vector<double> extractPointY(const vector<Point>& points) {
    vector<double> y(points.size());

    for (int i = 0; i < static_cast<int>(points.size()); ++i) {
        y[i] = points[i].y;
    }

    return y;
}

static void evaluateFitnessOnGPU(
    thrust::device_vector<DeviceIndividual>& population,
    const thrust::device_vector<double>& pointX,
    const thrust::device_vector<double>& pointY,
    int populationSize,
    int pointCount
) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (populationSize + threadsPerBlock - 1) / threadsPerBlock;

    calculateFitnessKernel << <blocksPerGrid, threadsPerBlock >> > (
        thrust::raw_pointer_cast(population.data()),
        thrust::raw_pointer_cast(pointX.data()),
        thrust::raw_pointer_cast(pointY.data()),
        populationSize,
        pointCount
        );

    CUDA_CHECK(cudaGetLastError());
}

static void runCrossoverOnGPU(
    thrust::device_vector<DeviceIndividual>& population,
    curandState* states,
    int populationSize
) {
    int parentCount = populationSize / 2;

    int threadsPerBlock = 256;
    int blocksPerGrid = (parentCount + threadsPerBlock - 1) / threadsPerBlock;

    crossoverKernel << <blocksPerGrid, threadsPerBlock >> > (
        thrust::raw_pointer_cast(population.data()),
        states,
        populationSize
        );

    CUDA_CHECK(cudaGetLastError());
}

static void runMutationOnGPU(
    thrust::device_vector<DeviceIndividual>& population,
    curandState* states,
    int* d_mutationCount,
    int populationSize,
    double Em,
    double Dm,
    double mutationValueStdDev
) {
    int totalGenes = populationSize * COEFF_COUNT;
    int maxMutationCount = totalGenes - COEFF_COUNT;

    generateMutationCountKernel << <1, 1 >> > (
        states,
        d_mutationCount,
        Em,
        Dm,
        maxMutationCount
        );

    CUDA_CHECK(cudaGetLastError());

    int threadsPerBlock = 256;
    int blocksPerGrid = (maxMutationCount + threadsPerBlock - 1) / threadsPerBlock;

    mutationKernel << <blocksPerGrid, threadsPerBlock >> > (
        thrust::raw_pointer_cast(population.data()),
        states,
        d_mutationCount,
        populationSize,
        mutationValueStdDev
        );

    CUDA_CHECK(cudaGetLastError());
}

GeneticAlgorithmResult runGeneticAlgorithmCUDA(
    const vector<Point>& points,
    const vector<Individual>& initialPopulation,
    const GeneticAlgorithmParams& params
) {
    GeneticAlgorithmResult result;

    const int populationSize = static_cast<int>(initialPopulation.size());
    const int pointCount = static_cast<int>(points.size());

    vector<DeviceIndividual> hostPopulation = convertToDeviceIndividuals(initialPopulation);

    vector<double> hostPointX = extractPointX(points);
    vector<double> hostPointY = extractPointY(points);

    thrust::device_vector<DeviceIndividual> devicePopulation = hostPopulation;
    thrust::device_vector<double> devicePointX = hostPointX;
    thrust::device_vector<double> devicePointY = hostPointY;

    int totalGenes = populationSize * COEFF_COUNT;
    int curandStateCount = totalGenes + 1;

    curandState* d_states = nullptr;
    int* d_mutationCount = nullptr;

    CUDA_CHECK(cudaMalloc(&d_states, curandStateCount * sizeof(curandState)));
    CUDA_CHECK(cudaMalloc(&d_mutationCount, sizeof(int)));

    int threadsPerBlock = 256;
    int blocksPerGrid = (curandStateCount + threadsPerBlock - 1) / threadsPerBlock;

    setupCurandKernel << <blocksPerGrid, threadsPerBlock >> > (
        d_states,
        params.seed + 777,
        curandStateCount
        );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;

    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));

    CUDA_CHECK(cudaEventRecord(startEvent));

    evaluateFitnessOnGPU(
        devicePopulation,
        devicePointX,
        devicePointY,
        populationSize,
        pointCount
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    thrust::sort(
        devicePopulation.begin(),
        devicePopulation.end(),
        FitnessComparator()
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    DeviceIndividual bestDeviceIndividual;
    CUDA_CHECK(cudaMemcpy(
        &bestDeviceIndividual,
        thrust::raw_pointer_cast(devicePopulation.data()),
        sizeof(DeviceIndividual),
        cudaMemcpyDeviceToHost
    ));

    double bestFitness = bestDeviceIndividual.fitness;
    int constIter = 0;

    cout << "\nGeneration 0"
        << " | best fitness = " << bestFitness << endl;

    for (int generation = 1; generation <= params.maxIter; ++generation) {
        runCrossoverOnGPU(
            devicePopulation,
            d_states,
            populationSize
        );

        double progress = static_cast<double>(generation) / params.maxIter;
        double mutationValueStdDev = 0.3 * (1.0 - progress) + 0.005 * progress;

        runMutationOnGPU(
            devicePopulation,
            d_states,
            d_mutationCount,
            populationSize,
            params.Em,
            params.Dm,
            mutationValueStdDev
        );

        evaluateFitnessOnGPU(
            devicePopulation,
            devicePointX,
            devicePointY,
            populationSize,
            pointCount
        );

        CUDA_CHECK(cudaDeviceSynchronize());

        thrust::sort(
            devicePopulation.begin(),
            devicePopulation.end(),
            FitnessComparator()
        );

        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(
            &bestDeviceIndividual,
            thrust::raw_pointer_cast(devicePopulation.data()),
            sizeof(DeviceIndividual),
            cudaMemcpyDeviceToHost
        ));

        double currentBestFitness = bestDeviceIndividual.fitness;

        if (currentBestFitness < bestFitness) {
            bestFitness = currentBestFitness;
            constIter = 0;
        }
        else {
            constIter++;
        }

        result.lastGeneration = generation;

        if (generation % 100 == 0 || generation == 1) {
            cout << "Generation " << generation
                << " | best fitness = " << currentBestFitness << endl;
        }

        if (currentBestFitness <= params.aimedFitness) {
            cout << "\nStopped because aimed fitness was achieved." << endl;
            break;
        }

        if (constIter >= params.maxConstIter) {
            cout << "\nStopped because best fitness did not improve for "
                << params.maxConstIter << " generations." << endl;
            break;
        }
    }

    CUDA_CHECK(cudaEventRecord(stopEvent));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));

    float elapsedMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent));

    CUDA_CHECK(cudaMemcpy(
        &bestDeviceIndividual,
        thrust::raw_pointer_cast(devicePopulation.data()),
        sizeof(DeviceIndividual),
        cudaMemcpyDeviceToHost
    ));

    result.bestIndividual = convertToHostIndividual(bestDeviceIndividual);
    result.processingTimeMs = elapsedMs;

    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));

    CUDA_CHECK(cudaFree(d_states));
    CUDA_CHECK(cudaFree(d_mutationCount));

    return result;
}