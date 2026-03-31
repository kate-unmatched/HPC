#include "matrix_cpu.h"

// умножение квадратных матриц (CPU)
void multiplyCPU(const std::vector<float>& matrixA, const std::vector<float>& matrixB, std::vector<float>& resultMatrix, int matrixSize) {

    for (int row = 0; row < matrixSize; row++) {
        for (int col = 0; col < matrixSize; col++) {

            float value = 0.0f;

            for (int k = 0; k < matrixSize; k++) {
                value += matrixA[row * matrixSize + k] * matrixB[k * matrixSize + col];
            }

            resultMatrix[row * matrixSize + col] = value;
        }
    }
}