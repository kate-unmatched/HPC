#include "utils.h"
#include <random>

std::vector<int> generate_vector(size_t n) {
    std::vector<int> v(n);

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(1, 100);

    for (size_t i = 0; i < n; ++i) {
        v[i] = dist(rng);
    }

    return v;
}
