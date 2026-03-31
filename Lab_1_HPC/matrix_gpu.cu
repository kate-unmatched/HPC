#include "matrix_gpu.h"
#include <cuda_runtime.h>
#include <iostream>

constexpr int kBlockSize = 16;

// Kernel
__global__ void multiplyKernel(const float* matrixA,
    const float* matrixB,
    float* resultMatrix,
    int matrixSize) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < matrixSize && col < matrixSize) {
        float value = 0.0f;

        for (int k = 0; k < matrixSize; k++) {
            value += matrixA[row * matrixSize + k] *
                matrixB[k * matrixSize + col];
        }

        resultMatrix[row * matrixSize + col] = value;
    }
}

float multiplyGPU(const std::vector<float>& matrixA,
    const std::vector<float>& matrixB,
    std::vector<float>& resultMatrix,
    int matrixSize) {

    float* d_matrixA = nullptr;
    float* d_matrixB = nullptr;
    float* d_resultMatrix = nullptr;

    std::size_t bytes = matrixSize * matrixSize * sizeof(float);

    // выделение памяти на GPU
    cudaMalloc(reinterpret_cast<void**>(&d_matrixA), bytes);
    cudaMalloc(reinterpret_cast<void**>(&d_matrixB), bytes);
    cudaMalloc(reinterpret_cast<void**>(&d_resultMatrix), bytes);

    // копирование данных на GPU
    cudaMemcpy(d_matrixA, matrixA.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrixB, matrixB.data(), bytes, cudaMemcpyHostToDevice);

    dim3 block(kBlockSize, kBlockSize);
    dim3 grid((matrixSize + kBlockSize - 1) / kBlockSize,
        (matrixSize + kBlockSize - 1) / kBlockSize);

    // Cuda эвенты
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cudaEventRecord(startEvent);

    multiplyKernel << <grid, block >> > (d_matrixA, d_matrixB, d_resultMatrix, matrixSize);

    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);

    float kernelTimeMs = 0.0f;
    cudaEventElapsedTime(&kernelTimeMs, startEvent, stopEvent);

    // копирование результата обратно
    cudaMemcpy(resultMatrix.data(), d_resultMatrix, bytes, cudaMemcpyDeviceToHost);

    // освобождение памяти
    cudaFree(d_matrixA);
    cudaFree(d_matrixB);
    cudaFree(d_resultMatrix);

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    return kernelTimeMs;
}