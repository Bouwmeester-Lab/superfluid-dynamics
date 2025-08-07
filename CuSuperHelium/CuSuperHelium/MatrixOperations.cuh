#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

template <typename Tfunc>
__global__ void matVectMultiply(const double* x, double*  y, int N, Tfunc func) {
    extern __shared__ double x_shared[];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0.0;

    for (int tile = 0; tile < (N + blockDim.x - 1) / blockDim.x; ++tile) {
        int idx = tile * blockDim.x + threadIdx.x;
        if (i < N) {
            for (int j = 0; j < blockDim.x; ++j) {
                int global_j = tile * blockDim.x + j;
                if (global_j < N) {
                    sum += func(i, global_j) * x[idx];
                }
            }
        }
    }

    if (i < N)
        y[i] = sum;
}