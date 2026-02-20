#include "gpu_LIN_upscaling.h"
#include "gpu_memory_management.h"
#include <cstdio>
#include <iostream>
#include <cuda_runtime_api.h>

std::uint32_t get_LIN_upscaled_width(std::uint32_t image_width){
    return (image_width > 0) ? (image_width * 2 - 1) : 0;
}

std::uint32_t get_LIN_upscaled_height(std::uint32_t image_height){
    return (image_height > 0) ? (image_height * 2 - 1) : 0;
}

__global__ void lin_upscale_kernel(const double* __restrict__ source, double* __restrict__ result, 
                                   std::uint32_t src_w, std::uint32_t src_h,
                                   std::uint32_t dst_w, std::uint32_t dst_h) {
    std::uint32_t dx = blockIdx.x * blockDim.x + threadIdx.x;
    std::uint32_t dy = blockIdx.y * blockDim.y + threadIdx.y;

    if (dx < dst_w && dy < dst_h) {
        bool x_even = (dx % 2 == 0);
        bool y_even = (dy % 2 == 0);
        std::uint32_t sx = dx / 2;
        std::uint32_t sy = dy / 2;

        if (x_even && y_even) {
            // Orijinal piksel
            result[dy * dst_w + dx] = source[sy * src_w + sx];
        } else if (!x_even && y_even) {
            // Yatay eksende (sol ve sağ) interpolasyon
            result[dy * dst_w + dx] = (source[sy * src_w + sx] + source[sy * src_w + sx + 1]) / 2.0;
        } else if (x_even && !y_even) {
            // Dikey eksende (üst ve alt) interpolasyon
            result[dy * dst_w + dx] = (source[sy * src_w + sx] + source[(sy + 1) * src_w + sx]) / 2.0;
        } else {
            // Çapraz (4 komşu) interpolasyon
            double tl = source[sy * src_w + sx];
            double tr = source[sy * src_w + sx + 1];
            double bl = source[(sy + 1) * src_w + sx];
            double br = source[(sy + 1) * src_w + sx + 1];
            result[dy * dst_w + dx] = (tl + tr + bl + br) / 4.0;
        }
    }
}

void LIN_image_upscaling(void** d_source_image, std::uint32_t source_image_height, std::uint32_t source_image_width, void** d_result){
    if (!d_source_image || !d_result) return;
    
    std::uint32_t dst_w = get_LIN_upscaled_width(source_image_width);
    std::uint32_t dst_h = get_LIN_upscaled_height(source_image_height);
    
    if (dst_w == 0 || dst_h == 0) return;

    cudaMalloc(d_result, dst_w * dst_h * sizeof(double));

    dim3 threads(16, 16);
    dim3 blocks((dst_w + 15) / 16, (dst_h + 15) / 16);

    lin_upscale_kernel<<<blocks, threads>>>(
        static_cast<const double*>(*d_source_image),
        static_cast<double*>(*d_result),
        source_image_width, source_image_height,
        dst_w, dst_h
    );
}