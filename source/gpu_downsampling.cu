#include "gpu_downsampling.h"
#include "gpu_memory_management.h"
#include <cstdio>
#include <cuda_runtime_api.h>

std::uint32_t get_downsampled_width(std::uint32_t image_width){
    return (image_width > 0) ? (image_width / 3 + 1) : 0;
}

std::uint32_t get_downsampled_height(std::uint32_t image_height){
    return (image_height > 0) ? (image_height / 3 + 1) : 0;
}

__global__ void downsample_kernel(const double* __restrict__ source, double* __restrict__ result,
                                  std::uint32_t src_w, std::uint32_t src_h,
                                  std::uint32_t dst_w, std::uint32_t dst_h) {
    std::uint32_t dx = blockIdx.x * blockDim.x + threadIdx.x;
    std::uint32_t dy = blockIdx.y * blockDim.y + threadIdx.y;

    if (dx < dst_w && dy < dst_h) {
        std::int32_t cx = dx * 3;
        std::int32_t cy = dy * 3;

        double sum = 0.0;
        std::uint32_t count = 0;

        for (std::int32_t oy = -1; oy <= 1; ++oy) {
            for (std::int32_t ox = -1; ox <= 1; ++ox) {
                std::int32_t px = cx + ox;
                std::int32_t py = cy + oy;

                if (px >= 0 && px < src_w && py >= 0 && py < src_h) {
                    sum += source[py * src_w + px];
                    count++;
                }
            }
        }
        
        result[dy * dst_w + dx] = (count > 0) ? (sum / count) : 0.0;
    }
}

void image_downsampling(void** d_source_image, std::uint32_t source_image_height, std::uint32_t source_image_width, void** d_result){
    if (!d_source_image || !d_result) return;

    std::uint32_t dst_w = get_downsampled_width(source_image_width);
    std::uint32_t dst_h = get_downsampled_height(source_image_height);

    if (dst_w == 0 || dst_h == 0) return;

    cudaMalloc(d_result, dst_w * dst_h * sizeof(double));

    dim3 threads(16, 16);
    dim3 blocks((dst_w + 15) / 16, (dst_h + 15) / 16);

    downsample_kernel<<<blocks, threads>>>(
        static_cast<const double*>(*d_source_image),
        static_cast<double*>(*d_result),
        source_image_width, source_image_height,
        dst_w, dst_h
    );
}