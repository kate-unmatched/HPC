#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

#include "matrix_utils.h"
#include "matrix_cpu.h"
#include "matrix_gpu.h"

int main() {

    std::vector<int> matrixSizes = { 100, 250, 500, 1000, 1500, 2000 };

    std::cout << std::left
        << std::setw(10) << "Size"
        << std::setw(15) << "CPU (ms)"
        << std::setw(18) << "GPU kernel (ms)"
        << std::setw(15) << "Speedup"
        << std::setw(10) << "Result"
        << std::endl;

    std::cout << std::string(65, '-') << std::endl;

    for (int matrixSize : matrixSizes) {

        std::vector<float> matrixA(matrixSize * matrixSize);
        std::vector<float> matrixB(matrixSize * matrixSize);
        std::vector<float> cpuResult(matrixSize * matrixSize, 0.0f);
        std::vector<float> gpuResult(matrixSize * matrixSize, 0.0f);

        fillMatrix(matrixA, matrixSize);
        fillMatrix(matrixB, matrixSize);

        // CPU
        auto cpuStart = std::chrono::high_resolution_clock::now();
        multiplyCPU(matrixA, matrixB, cpuResult, matrixSize);
        auto cpuEnd = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> cpuDuration = cpuEnd - cpuStart;

        // GPU
        float gpuKernelTimeMs = multiplyGPU(matrixA, matrixB, gpuResult, matrixSize);

        // Проверка результатов
        bool isCorrect = checkResult(cpuResult, gpuResult, matrixSize * matrixSize);

        // Ускорение
        double speedup = cpuDuration.count() / gpuKernelTimeMs;

        // Таблица
        std::cout << std::left
            << std::setw(10) << matrixSize
            << std::setw(15) << std::fixed << std::setprecision(3) << cpuDuration.count()
            << std::setw(18) << gpuKernelTimeMs
            << std::setw(15) << speedup
            << std::setw(10) << (isCorrect ? "OK" : "ERROR")
            << std::endl;
    }

    return 0;
}