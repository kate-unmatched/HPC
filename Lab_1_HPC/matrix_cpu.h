#pragma once
#include <vector>

// умножение матриц на CPU
void multiplyCPU(const std::vector<float>& matrixA,
    const std::vector<float>& matrixB,
    std::vector<float>& resultMatrix,
    int matrixSize);