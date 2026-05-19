#include <iostream>
#include <iomanip>
#include <array>
#include <vector>

#include "common.h"
#include "data_generation.h"
#include "genetic_cuda.cuh"

using namespace std;

void printCoefficients(const array<double, COEFF_COUNT>& coeffs) {
    for (int i = 0; i < COEFF_COUNT; ++i) {
        cout << "c" << i << " = " << coeffs[i] << endl;
    }
}

void printFinalReport(
    const GeneticAlgorithmResult& result,
    const array<double, COEFF_COUNT>& trueCoeffs,
    double trueFitness
) {
    cout << "\nFINAL REPORT" << endl;

    cout << "\nTrue coefficients:" << endl;
    printCoefficients(trueCoeffs);

    cout << "\nFitness for true coefficients: " << trueFitness << endl;

    cout << "\nFound coefficients:" << endl;
    printCoefficients(result.bestIndividual.coeffs);

    cout << "\nBest fitness: " << result.bestIndividual.fitness << endl;
    cout << "Last generation: " << result.lastGeneration << endl;
    cout << "GPU processing time: " << result.processingTimeMs << " ms" << endl;
}

int main() {
    cout << fixed << setprecision(6);

    GeneticAlgorithmParams params;

    array<double, COEFF_COUNT> trueCoeffs = {
        2.0,
        -1.5,
        0.3,
        0.1,
        -0.01
    };

    vector<Point> points = generatePoints(
        params.pointCount,
        trueCoeffs,
        params.xMin,
        params.xMax,
        params.noiseStdDev,
        params.seed
    );

    vector<Individual> population = initializePopulation(
        params.populationSize,
        params.coeffMin,
        params.coeffMax,
        params.seed + 1
    );

    Individual trueIndividual;
    trueIndividual.coeffs = trueCoeffs;
    trueIndividual.fitness = calculateFitnessCPU(trueIndividual, points);

    cout << "Generated points: " << points.size() << endl;
    cout << "Population size: " << population.size() << endl;
    cout << "Polynomial degree: " << POLYNOMIAL_DEGREE << endl;

    cout << "\nFirst 5 generated points:" << endl;
    for (int i = 0; i < 5; ++i) {
        cout << "x = " << points[i].x
            << ", y = " << points[i].y << endl;
    }

    GeneticAlgorithmResult result = runGeneticAlgorithmCUDA(
        points,
        population,
        params
    );

    printFinalReport(
        result,
        trueCoeffs,
        trueIndividual.fitness
    );

    return 0;
}