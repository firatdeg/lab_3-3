#include "gpu_memory_management.h"
#include <cuda_runtime_api.h>
#include <cuda.h>

void allocate_device_memory(IntermediateImage& image, void** devPtr) {
    std::size_t bytes = image.height * image.width * sizeof(double);
    cudaMalloc(devPtr, bytes);
}

void free_device_memory(void** devPtr) {
    if (devPtr != nullptr && *devPtr != nullptr) {
        cudaFree(*devPtr);
        *devPtr = nullptr;
    }
}

void copy_data_to_device(IntermediateImage& image, void** devPtr) {
    std::size_t bytes = image.height * image.width * sizeof(double);
    cudaMemcpy(*devPtr, image.pixels.data(), bytes, cudaMemcpyHostToDevice);
}

void copy_data_from_device(void** devPtr, IntermediateImage& image) {
    std::size_t bytes = image.height * image.width * sizeof(double);
    cudaMemcpy(image.pixels.data(), *devPtr, bytes, cudaMemcpyDeviceToHost);
}