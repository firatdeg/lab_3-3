#include "gpu_NN_upscaling.h"
#include "gpu_memory_management.h"
#include <cstdio>
#include <iostream>

std::uint32_t get_NN_upscaled_width(std::uint32_t image_width){
    return image_width * 3;
}

std::uint32_t get_NN_upscaled_height(std::uint32_t image_height){
    return image_height * 3;
}

__global__ void nn_upscale_kernel(const double* __restrict__ source, double* __restrict__ result, 
                                  std::uint32_t src_w, std::uint32_t src_h,
                                  std::uint32_t dst_w, std::uint32_t dst_h) {
    std::uint32_t dx = blockIdx.x * blockDim.x + threadIdx.x;
    std::uint32_t dy = blockIdx.y * blockDim.y + threadIdx.y;

    if (dx < dst_w && dy < dst_h) {
        std::uint32_t sx = dx / 3;
        std::uint32_t sy = dy / 3;
        result[dy * dst_w + dx] = source[sy * src_w + sx];
    }
}

void NN_image_upscaling(void** d_source_image, std::uint32_t source_image_height, std::uint32_t source_image_width, void** d_result){
    if (!d_source_image || !(*d_source_image) || !d_result) return;

    std::uint32_t dst_w = get_NN_upscaled_width(source_image_width);
    std::uint32_t dst_h = get_NN_upscaled_height(source_image_height);
    
    if (dst_w == 0 || dst_h == 0) return;

    cudaMalloc(d_result, dst_w * dst_h * sizeof(double));

    dim3 threads(16, 16);
    dim3 blocks((dst_w + 15) / 16, (dst_h + 15) / 16);

    nn_upscale_kernel<<<blocks, threads>>>(
        static_cast<const double*>(*d_source_image), 
        static_cast<double*>(*d_result), 
        source_image_width, source_image_height, 
        dst_w, dst_h
    );
}