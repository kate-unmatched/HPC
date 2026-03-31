#pragma once
#include <vector>

// заполнение матриц рандомными числами
void fillMatrix(std::vector<float>& matrixData, int matrixSize);

// сверка результатов умножения матриц у CPU и GPU
bool checkResult(const std::vector<float>& cpuResult,
    const std::vector<float>& gpuResult,
    int totalElements);