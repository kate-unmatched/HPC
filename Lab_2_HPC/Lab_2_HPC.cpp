// Lab 2 - Сумма элементов вектора
// 
// Задача: реализовать алгоритм сложения элементов вектора
// 
// Входные данные: вектор размером 1 000..1 000 000 значений.
// Выходные данные: сумма элементов вектора + время вычисления
// 
// Реализация должна содержать 2 функции сложения элементов вектора: на CPU и на
// GPU с применением CUDA.
// 
// Провести эксперименты : получить сумму векторов разных размеров (провести 5 или более экспериментов), 
// посчитать ускорение. Результаты привести в виде таблицы / графика.

#include <iostream>
#include <vector>
#include <iomanip>

#include "utils.h"
#include "cpu_sum.h"
#include "gpu_sum.h"

void run_experiments(const std::vector<size_t>& sizes) {
    for (size_t n : sizes) {
        auto v = generate_vector(n);

        long long cpu_result;
        double cpu_time = measure_cpu(v, cpu_result);

        double gpu_time = 0.0;
        long long gpu_result = sum_gpu(v, gpu_time);

        if (cpu_result != gpu_result) {
            std::cout << "ERROR at N = " << n << std::endl;
            continue;
        }

        double speedup = cpu_time / gpu_time;

        std::cout << std::left << std::setw(12) << n
            << std::setw(15) << cpu_time
            << std::setw(15) << gpu_time
            << std::setw(15) << speedup
            << std::endl;
    }
}

int main() {
    // Основные эксперименты
    std::vector<size_t> base_sizes = {
        1000,
        10000,
        50000,
        100000,
        500000,
        1000000
    };

    // Дополнительные эксперименты для проверки гипотез
    std::vector<size_t> extra_sizes = {
        2000000,
        5000000,
        10000000,
        20000000
    };

    std::cout << std::left << std::setw(12) << "N"
        << std::setw(15) << "CPU (ms)"
        << std::setw(15) << "GPU (ms)"
        << std::setw(15) << "Speedup"
        << std::endl;

    std::cout << "==========================================================\n";

    std::cout << "Within constraints (1K - 1M):\n";
    run_experiments(base_sizes);

    std::cout << "----------------------------------------------------------\n";
    std::cout << "Extended experiments (>1M):\n";

    run_experiments(extra_sizes);

    return 0;
}