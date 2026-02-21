#include "gpu_matrix_convolution.h"
#include <cstdio>
#include <string>
#include <cuda_runtime_api.h>

__global__ void convolution_kernel(const double *__restrict__ source,
                                   const double *__restrict__ filter,
                                   double *__restrict__ result,
                                   std::uint32_t width, std::uint32_t height,
                                   std::uint32_t k_size)
{
    std::uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    std::uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        std::int32_t a = k_size / 2;
        double sum = 0.0;

        for (std::int32_t j = 0; j < k_size; ++j)
        {
            for (std::int32_t i = 0; i < k_size; ++i)
            {
                std::int32_t target_x = (std::int32_t)x - i + a;
                std::int32_t target_y = (std::int32_t)y - j + a;

                double pixel_val = 0.0;
                if (target_x >= 0 && target_x < width && target_y >= 0 && target_y < height)
                {
                    pixel_val = source[target_y * width + target_x];
                }

                double filter_val = filter[j * k_size + i];
                sum += pixel_val * filter_val;
            }
        }
        result[y * width + x] = sum;
    }
}

void matrix_convolution(void **d_source_matrix, std::uint32_t matrix_width, std::uint32_t matrix_height,
                        void **d_kernel, std::uint32_t kernel_width, std::uint32_t kernel_height,
                        void **d_result)
{
    if (!d_source_matrix || !(*d_source_matrix) || !d_kernel || !(*d_kernel) || !d_result)
        return;
    if (matrix_width == 0 || matrix_height == 0 || kernel_width == 0)
        return;

    cudaMalloc(d_result, matrix_width * matrix_height * sizeof(double));

    dim3 threads(16, 16);
    dim3 blocks((matrix_width + 15) / 16, (matrix_height + 15) / 16);

    convolution_kernel<<<blocks, threads>>>(
        static_cast<const double *>(*d_source_matrix),
        static_cast<const double *>(*d_kernel),
        static_cast<double *>(*d_result),
        matrix_width, matrix_height,
        kernel_width);
}