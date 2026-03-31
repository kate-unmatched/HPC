#pragma once
#include <vector>

// умножение матриц на GPU (возвращает время выполнения kernel в мс)
float multiplyGPU(const std::vector<float>& matrixA,
    const std::vector<float>& matrixB,
    std::vector<float>& resultMatrix,
    int matrixSize);