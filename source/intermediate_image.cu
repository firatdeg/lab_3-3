#include <cstdio>
#include <iostream>
#include <cuda_runtime_api.h>
#include "intermediate_image.h"
#include "gpu_matrix_convolution.h"
#include "gpu_utilities.h"

__global__ void sobel_post_process_kernel(double* data, std::uint32_t total_elements) {
    std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        double f = data[idx];
        double f_prime = f * f;
        double f_double_prime = f_prime + f_prime;
        data[idx] = sqrt(f_double_prime);
    }
}

void IntermediateImage::apply_sobel_filter() {
    std::uint32_t total_elements = width * height;
    if (total_elements == 0) return;

    double* d_source = nullptr;
    cudaMalloc(&d_source, total_elements * sizeof(double));
    cudaMemcpy(d_source, pixels.data(), total_elements * sizeof(double), cudaMemcpyHostToDevice);

    double h_K[9] = {-1.0, 0.0, 1.0,
                     -2.0, 0.0, 2.0,
                     -1.0, 0.0, 1.0};
    double* d_K = nullptr;
    cudaMalloc(&d_K, 9 * sizeof(double));
    cudaMemcpy(d_K, h_K, 9 * sizeof(double), cudaMemcpyHostToDevice);


    void* ptr_source = static_cast<void*>(d_source);
    void* ptr_K = static_cast<void*>(d_K);
    void* ptr_F = nullptr;

    matrix_convolution(&ptr_source, width, height, &ptr_K, 3, 3, &ptr_F);

    double* d_F = static_cast<double*>(ptr_F);

    std::uint32_t threads = 256;
    std::uint32_t blocks = (total_elements + threads - 1) / threads;
    sobel_post_process_kernel<<<blocks, threads>>>(d_F, total_elements);

    cudaMemcpy(pixels.data(), d_F, total_elements * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_source);
    cudaFree(d_K);
    cudaFree(d_F);
}