#include "data_generation.h"

#include <random>

double polynomialValueCPU(
    const std::array<double, COEFF_COUNT>& coeffs,
    double x
) {
    double result = 0.0;
    double power = 1.0;

    for (int i = 0; i < COEFF_COUNT; ++i) {
        result += coeffs[i] * power;
        power *= x;
    }

    return result;
}

std::vector<Point> generatePoints(
    int pointCount,
    const std::array<double, COEFF_COUNT>& trueCoeffs,
    double xMin,
    double xMax,
    double noiseStdDev,
    unsigned int seed
) {
    std::vector<Point> points;
    points.reserve(pointCount);

    std::mt19937 generator(seed);
    std::normal_distribution<double> noiseDistribution(0.0, noiseStdDev);

    double step = (xMax - xMin) / (pointCount - 1);

    for (int i = 0; i < pointCount; ++i) {
        double x = xMin + i * step;
        double y = polynomialValueCPU(trueCoeffs, x);

        if (noiseStdDev > 0.0) {
            y += noiseDistribution(generator);
        }

        points.push_back({ x, y });
    }

    return points;
}

std::vector<Individual> initializePopulation(
    int populationSize,
    double coeffMin,
    double coeffMax,
    unsigned int seed
) {
    std::vector<Individual> population;
    population.reserve(populationSize);

    std::mt19937 generator(seed);
    std::uniform_real_distribution<double> coeffDistribution(coeffMin, coeffMax);

    for (int i = 0; i < populationSize; ++i) {
        Individual individual;

        for (int j = 0; j < COEFF_COUNT; ++j) {
            individual.coeffs[j] = coeffDistribution(generator);
        }

        population.push_back(individual);
    }

    return population;
}

double calculateFitnessCPU(
    const Individual& individual,
    const std::vector<Point>& points
) {
    double sumSquaredError = 0.0;

    for (const Point& point : points) {
        double predictedY = polynomialValueCPU(individual.coeffs, point.x);
        double error = predictedY - point.y;
        sumSquaredError += error * error;
    }

    return sumSquaredError;
}