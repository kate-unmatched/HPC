#pragma once
#include <vector>

// Вычисление суммы элементов на GPU
long long sum_gpu(const std::vector<int>& host_data, double& time_ms);