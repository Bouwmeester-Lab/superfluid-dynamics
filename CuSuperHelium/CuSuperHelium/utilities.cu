#pragma once
#include "utilities.cuh"

__global__ void complex_pointwise_mul(
    const cufftDoubleComplex* a,
    const cufftDoubleComplex* b,
    cufftDoubleComplex* result,
    const int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        cufftDoubleComplex x = a[i];
        cufftDoubleComplex y = b[i];
        result[i].x = x.x * y.x - x.y * y.y;  // real
        result[i].y = x.x * y.y + x.y * y.x;  // imag
    }
}