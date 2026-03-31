#include "matrix_utils.h"
#include <random>
#include <cmath>


// заполнение матрицы случайными значениями
void fillMatrix(std::vector<float>& matrixData, int matrixSize) {
    std::mt19937 randomEngine(42);
    std::uniform_real_distribution<float> distribution(0.0f, 10.0f);

    for (auto& value : matrixData) {
        value = distribution(randomEngine);
    }
}

// проверка совпадения результатов
bool checkResult(const std::vector<float>& cpuResult,
    const std::vector<float>& gpuResult,
    int totalElements) {

    const float kEpsilon = 1e-3f;

    for (int i = 0; i < totalElements; i++) {
        if (std::fabs(cpuResult[i] - gpuResult[i]) > kEpsilon) {
            return false;
        }
    }

    return true;
}