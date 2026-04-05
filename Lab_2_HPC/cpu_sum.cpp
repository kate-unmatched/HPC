#include "cpu_sum.h"
#include <chrono>

long long sum_cpu(const std::vector<int>& data) {
    long long total_sum = 0;

    for (int value : data) {
        total_sum += value;
    }

    return total_sum;
}

double measure_cpu(const std::vector<int>& data, long long& out_sum) {
    const auto start_time = std::chrono::high_resolution_clock::now();

    out_sum = sum_cpu(data);

    const auto end_time = std::chrono::high_resolution_clock::now();

    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(end_time - start_time).count();

    return elapsed_ms;
}