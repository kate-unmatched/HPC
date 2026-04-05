#pragma once
#include <vector>

// Вычисление суммы элементов на CPU
long long sum_cpu(const std::vector<int>& data);

// Измерение времени выполнения CPU-суммы
double measure_cpu(const std::vector<int>& data, long long& out_sum);