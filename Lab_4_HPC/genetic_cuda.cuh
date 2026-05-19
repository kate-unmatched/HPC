#pragma once

#include <vector>
#include "common.h"

GeneticAlgorithmResult runGeneticAlgorithmCUDA(
    const std::vector<Point>& points,
    const std::vector<Individual>& initialPopulation,
    const GeneticAlgorithmParams& params
);