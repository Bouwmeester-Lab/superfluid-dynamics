#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void matVecOnTheFlyShared(const double* __restrict__ x, double* __restrict__ y, int N) {
    extern __shared__ double x_shared[];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0.0;

    for (int tile = 0; tile < (N + blockDim.x - 1) / blockDim.x; ++tile) {
        int idx = tile * blockDim.x + threadIdx.x;
        if (idx < N)
            x_shared[threadIdx.x] = x[idx];
        else
            x_shared[threadIdx.x] = 0.0;

        __syncthreads();

        if (i < N) {
#pragma unroll
            for (int j = 0; j < blockDim.x; ++j) {
                int global_j = tile * blockDim.x + j;
                if (global_j < N) {
                    sum += A(i, global_j) * x_shared[j];
                }
            }
        }
        __syncthreads();
    }

    if (i < N)
        y[i] = sum;
}