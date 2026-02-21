#include "benchmark.h"
#include "../source/gpu_reductions.h"

#include <cstdint>
#include <vector>
#include <iostream>
#include <omp.h>
#include <cuda_runtime_api.h>
#include <cfloat>


double get_max_value_serial(const std::vector<double>& image) {
    double max_val = -DBL_MAX;
    for (double val : image) {
        if (val > max_val) max_val = val;
    }
    return max_val;
}


double get_max_value_omp(const std::vector<double>& image) {
    double max_val = -DBL_MAX;
    #pragma omp parallel for reduction(max:max_val)
    for (std::size_t i = 0; i < image.size(); ++i) {
        if (image[i] > max_val) max_val = image[i];
    }
    return max_val;
}


static void serial_benchmark(benchmark::State& state) {
    const auto n = state.range(0);
    
    for (auto _ : state) {
        state.PauseTiming();

        std::vector<double> dummy_data(n);
        for(std::uint32_t i = 0; i < n; ++i) {
            dummy_data[i] = static_cast<double>(i % 255);
        }
        state.ResumeTiming();

        double res = get_max_value_serial(dummy_data);
        benchmark::DoNotOptimize(res);
    }   
}


static void omp_benchmark(benchmark::State& state) {
    const auto n = state.range(0);
    
    for (auto _ : state) {
        state.PauseTiming();
        std::vector<double> dummy_data(n);
        for(std::uint32_t i = 0; i < n; ++i) {
            dummy_data[i] = static_cast<double>(i % 255);
        }
        state.ResumeTiming();

        double res = get_max_value_omp(dummy_data);
        benchmark::DoNotOptimize(res);
    }   
}


static void gpu_benchmark(benchmark::State& state) {
    const auto n = state.range(0);
    
    for (auto _ : state) {
        state.PauseTiming();
        std::vector<double> dummy_data(n);
        for(std::uint32_t i = 0; i < n; ++i) {
            dummy_data[i] = static_cast<double>(i % 255);
        }
        
        double* d_data = nullptr;
        cudaMalloc((void**)&d_data, n * sizeof(double));
        cudaMemcpy(d_data, dummy_data.data(), n * sizeof(double), cudaMemcpyHostToDevice);
        void* d_ptr = static_cast<void*>(d_data);
        state.ResumeTiming();


        double res = get_max_value(&d_ptr, 1, n);
        benchmark::DoNotOptimize(res);

        state.PauseTiming();
        cudaFree(d_data);
        state.ResumeTiming();
    }   
}

int main(int argc, char** argv) {
    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}

BENCHMARK(serial_benchmark)->Unit(benchmark::kMillisecond)->RangeMultiplier(10)->Range(1000, 10000000);
BENCHMARK(omp_benchmark)->Unit(benchmark::kMillisecond)->RangeMultiplier(10)->Range(1000, 100000000);
BENCHMARK(gpu_benchmark)->Unit(benchmark::kMillisecond)->RangeMultiplier(10)->Range(1000, 100000000);