#include "gpu_reductions.h"
#include <cstdio>
#include <iostream>
#include <cuda_runtime_api.h>
#include <cfloat> 

__device__ void atomicMaxDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        double current_val = __longlong_as_double(assumed);
        if (val <= current_val) break; 
        
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val));
    } while (assumed != old);
}

__global__ void max_reduction_kernel(const double* __restrict__ source, double* __restrict__ global_max, std::uint32_t total_elements) {

    extern __shared__ double sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    double my_max = -DBL_MAX;
    for (unsigned int idx = i; idx < total_elements; idx += blockDim.x * gridDim.x) {
        my_max = fmax(my_max, source[idx]);
    }
    sdata[tid] = my_max;
    __syncthreads(); 
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicMaxDouble(global_max, sdata[0]);
    }
}

double get_max_value(void** d_source_image, std::uint32_t source_image_height, std::uint32_t source_image_width) {
    if (!d_source_image || !(*d_source_image)) return 0.0;
    
    std::uint32_t total_elements = source_image_height * source_image_width;
    if (total_elements == 0) return 0.0;

    double* d_global_max = nullptr;
    cudaMalloc(&d_global_max, sizeof(double));
    
    double init_val = -DBL_MAX;
    cudaMemcpy(d_global_max, &init_val, sizeof(double), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    if (blocks > 1024) blocks = 1024; 

    max_reduction_kernel<<<blocks, threads, threads * sizeof(double)>>>(
        static_cast<const double*>(*d_source_image), 
        d_global_max, 
        total_elements
    );

    double h_max = 0.0;
    cudaMemcpy(&h_max, d_global_max, sizeof(double), cudaMemcpyDeviceToHost);
    
    cudaFree(d_global_max);

    return h_max;
}