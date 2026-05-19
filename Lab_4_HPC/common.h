#pragma once

#include <array>

constexpr int POLYNOMIAL_DEGREE = 4;
constexpr int COEFF_COUNT = POLYNOMIAL_DEGREE + 1;

struct Point {
    double x;
    double y;
};

struct Individual {
    std::array<double, COEFF_COUNT> coeffs{};
    double fitness = 0.0;
};

struct GeneticAlgorithmParams {
    int pointCount = 500;
    int populationSize = 1000;

    double xMin = -5.0;
    double xMax = 5.0;
    double noiseStdDev = 0.1;

    double coeffMin = -10.0;
    double coeffMax = 10.0;

    int maxIter = 5000;
    int maxConstIter = 700;

    double Em = 120.0;
    double Dm = 30.0;

    double aimedFitness = 5.5;

    unsigned int seed = 42;
};

struct GeneticAlgorithmResult {
    Individual bestIndividual;
    int lastGeneration = 0;
    double processingTimeMs = 0.0;
};