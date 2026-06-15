#pragma once
#ifndef VECTOR_KERNELS_H
#define VECTOR_KERNELS_H

#include "../constants.cuh"
#include "../Math/ComplexMath.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cufft.h>

/// <summary>
/// Substracts two vectors out = Re(a) - b + iIm(a) and stores the result in out.
/// </summary>
/// <param name="a">Complex vector</param>
/// <param name="b">Real part to substract from a.</param>
/// <param name="out"></param>
/// <param name="n"></param>
__global__ void vector_subtract_complex_real(const cufftDoubleComplex* a, const double* b, cufftDoubleComplex* out, int n);
/// <summary>
/// Element wise addition. It will add b to the real part of a for every element in a.
/// </summary>
/// <param name="a"></param>
/// <param name="b"></param>
/// <param name="out"></param>
/// <param name="n"></param>
__global__ void vector_scalar_add_complex_real(const cufftDoubleComplex* a, const double b, cufftDoubleComplex* out, int n, int start);

__global__ void cotangent_complex(const cufftDoubleComplex* a, cufftDoubleComplex* out, int n);
__global__ void real_to_complex(const double* x, cuDoubleComplex* x_c, int N);
__global__ void complex_to_real(const cuDoubleComplex* x_c, double* x, int N);
__global__ void conjugate_vector(cuDoubleComplex* x, cuDoubleComplex* z, int N);

__global__ void add_k_vectors(cufftDoubleComplex* k1, cufftDoubleComplex* k2, cufftDoubleComplex* k3, cufftDoubleComplex* k4, cufftDoubleComplex* result, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        result[i].x = k1[i].x + 2.0 * k2[i].x + 2.0 * k3[i].x + k4[i].x;
        result[i].y = k1[i].y + 2.0 * k2[i].y + 2.0 * k3[i].y + k4[i].y;
    }
}

__global__ void add_k_vectors(std_complex* k1, std_complex* k2, std_complex* k3, std_complex* k4, std_complex* result, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        result[i] = k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i];
    }
}

__global__ void linspace(double start, double end, double* out, int steps)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < steps)
    {
        out[tid] = start + tid * (end - start) / (steps - 1);
    }
}

__global__ void vector_subtract_complex_real(const std_complex* a, const double* b, std_complex* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] - b[i];
    }
}

/// <summary>
/// Performs batched subtraction of a real single vector (size N) from a complex vector on the GPU (NxbatchSize), storing the result in an output array.
/// </summary>
/// <param name="a">Pointer to the input array of complex numbers (batched vectors).</param>
/// <param name="b">Pointer to the input array of real numbers (single vector).</param>
/// <param name="out">Pointer to the output array where the results are stored.</param>
/// <param name="n">The length of each vector.</param>
/// <param name="batchSize">The number of vectors in the batch.</param>
/// <returns>This is a CUDA kernel function and does not return a value. The results are written to the output array 'out'.</returns>
__global__ void batched_vector_subtract_singletime_complex_real(const std_complex* a, const double* b, std_complex* out, int n, size_t batchSize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n * batchSize) return; // out of bounds
    int i = tid % n; // index within the vector

    if (i < n) {
        out[tid] = a[tid] - b[i];
    }
}

/// <summary>
/// Adds a real scalar value to the real part of each element in a complex vector using CUDA parallelization.
/// </summary>
/// <param name="a">Pointer to the input array of complex numbers.</param>
/// <param name="b">The real scalar value to add to the real part of each complex number.</param>
/// <param name="out">Pointer to the output array where the results are stored.</param>
/// <param name="n">The number of elements to process.</param>
/// <param name="start">The starting index in the arrays for processing.</param>
/// <returns>This function does not return a value; results are written to the output array.</returns>
__global__ void vector_scalar_add_complex_real(const std_complex* a, const double b, std_complex* out, int n, int start)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[start + i] = a[start + i] + b; // adds the scalar to the real part of the complex number
    }
}

__global__ void vector_mutiply_scalar(const std_complex* a, const double b, std_complex* out, int n, int start)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[start + i] = a[start + i] * b; // multiplies the complex number by the scalar
    }
}

/// <summary>
/// Calculates the cotangent using cot(z) = cos(z)/sin(z).
/// </summary>
/// <param name="a"></param>
/// <param name="out"></param>
/// <param name="n"></param>
__global__ void cotangent_complex(const cufftDoubleComplex* a, cufftDoubleComplex* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = cotangent_complex(a[i]); // using the device function to calculate the cotangent
    }
}

__global__ void real_to_complex(const double* x, cuDoubleComplex* x_c, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N)
        x_c[idx] = make_cuDoubleComplex(x[idx], 0.0);
}

__global__ void real_to_complex(const double* x, std_complex* x_c, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N)
        x_c[idx] = std_complex(x[idx], 0.0);
}

__global__ void complex_to_real(const cuDoubleComplex* x_c, double* x, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N)
        x[idx] = x_c[idx].x; // only copy the real part
}

__global__ void complex_to_real(const std_complex* x_c, double* x, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N)
        x[idx] = x_c[idx].real(); // only copy the real part
}

__global__ void conjugate_vector(cuDoubleComplex* x, cuDoubleComplex* z, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        z[i].x = x[i].x; // copy the real part
        z[i].y = -x[i].y; // conjugate
    }
}

__global__ void conjugate_vector(std_complex* x, std_complex* z, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        z[i] = cuda::std::conj(x[i]); // use the standard library to conjugate the complex number
    }
}

#endif
