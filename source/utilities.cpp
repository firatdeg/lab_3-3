#include "utilities.h"
void gpu_max_reduction(const double* d_in, double* d_out, std::size_t size) {
    // Kernel launch parameters
    dim3 blockSize(256);
    dim3 gridSize((size + blockSize.x - 1) / blockSize.x);

    max_reduction_kernel<<<gridSize, blockSize>>>(d_in, d_out, size);
}
