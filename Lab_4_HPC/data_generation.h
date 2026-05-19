#pragma once
#pragma once

#include <vector>
#include <array>
#include "common.h"

double polynomialValueCPU(
    const std::array<double, COEFF_COUNT>& coeffs,
    double x
);

std::vector<Point> generatePoints(
    int pointCount,
    const std::array<double, COEFF_COUNT>& trueCoeffs,
    double xMin,
    double xMax,
    double noiseStdDev,
    unsigned int seed
);

std::vector<Individual> initializePopulation(
    int populationSize,
    double coeffMin,
    double coeffMax,
    unsigned int seed
);

double calculateFitnessCPU(
    const Individual& individual,
    const std::vector<Point>& points
);