#include "gpu_sum.h"
#include <cuda_runtime.h>
#include <vector>

#define BLOCK_SIZE 256

__global__ void reduce_kernel(int* device_input,
    long long* device_partial_sums,
    int element_count) {
    __shared__ long long shared_data[BLOCK_SIZE];

    const int thread_id = threadIdx.x;
    const int global_index = blockIdx.x * blockDim.x + threadIdx.x;

    shared_data[thread_id] =
        (global_index < element_count) ? device_input[global_index] : 0;

    __syncthreads();

    // reduction внутри блока
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (thread_id < stride) {
            shared_data[thread_id] += shared_data[thread_id + stride];
        }
        __syncthreads();
    }

    if (thread_id == 0) {
        device_partial_sums[blockIdx.x] = shared_data[0];
    }
}

long long sum_gpu(const std::vector<int>& host_data, double& time_ms) {
    const int element_count = static_cast<int>(host_data.size());

    int* device_input = nullptr;
    long long* device_partial_sums = nullptr;

    const int block_count =
        (element_count + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaMalloc(&device_input, element_count * sizeof(int));
    cudaMalloc(&device_partial_sums, block_count * sizeof(long long));

    cudaMemcpy(device_input,
        host_data.data(),
        element_count * sizeof(int),
        cudaMemcpyHostToDevice);

    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    cudaEventRecord(start_event);

    reduce_kernel << <block_count, BLOCK_SIZE >> > (
        device_input,
        device_partial_sums,
        element_count
        );

    std::vector<long long> host_partial_sums(block_count);

    cudaMemcpy(host_partial_sums.data(),
        device_partial_sums,
        block_count * sizeof(long long),
        cudaMemcpyDeviceToHost);

    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event);

    float elapsed_ms_f = 0.0f;
    cudaEventElapsedTime(&elapsed_ms_f, start_event, stop_event);

    time_ms = static_cast<double>(elapsed_ms_f);

    long long total_sum = 0;
    for (const auto value : host_partial_sums) {
        total_sum += value;
    }

    cudaFree(device_input);
    cudaFree(device_partial_sums);

    return total_sum;
}