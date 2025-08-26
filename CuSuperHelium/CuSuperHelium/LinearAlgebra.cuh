#pragma once
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cuda/std/complex>



template <typename T>
/// <summary>
/// AXPY operation: out = alpha * x + y for vectors x and y of length n without having to overwrite y.
/// </summary>
__global__ void axpy(const T* __restrict__ x, const T* __restrict__ y, T* __restrict__ out, T alpha, size_t n) 
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t stride = blockDim.x * gridDim.x;
	for (; i < n; i += stride) 
	{
		out[i] = alpha * x[i] + y[i];
	}
}

template <typename T>
/// <summary>
/// Linear combination of two vectors: out = alpha * x1 + beta * x2 + y for vectors x and y of length n without having to overwrite y.
/// </summary>
__global__ void lincomb(const T* __restrict__ x1, const T* __restrict__ x2, const T* __restrict__ y,
	T* __restrict__ out, T alpha, T beta, size_t n) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t stride = blockDim.x * gridDim.x;
	for (; i < n; i += stride) 
	{
		out[i] = alpha * x1[i] + beta * x2[i] + y[i];
	}
}

template <typename T>
/// <summary>
/// Linear combination of three vectors: out = alpha * x1 + beta * x2 + gamma * x3 + y for vectors x and y of length n without having to overwrite y.
/// </summary>
/// <param name="x1">First input vector.</param>
/// <param name="x2">Second input vector.</param>
/// <param name="x3">Third input vector.</param>
/// <param name="y">Input vector to be added.</param>
/// <param name="out">Output vector.</param>
/// <param name="alpha">Scalar multiplier for x1.</param>
/// <param name="beta">Scalar multiplier for x2.</param>
/// <param name="gamma">Scalar multiplier for x3.</param>
/// <param name="n">Length of the vectors.</param>
/// <returns>None.</returns>
__global__ void lincomb(const T* __restrict__ x1, const T* __restrict__ x2, const T* __restrict__ x3, const T* __restrict__ y,
	T* __restrict__ out, T alpha, T beta, T gamma, size_t n) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t stride = blockDim.x * gridDim.x;
	for (; i < n; i += stride)
	{
		out[i] = alpha * x1[i] + beta * x2[i] + gamma * x3[i] + y[i];
	}
}

template <typename T>
/// <summary>
/// Linear combination of four vectors: out = alpha * x1 + beta * x2 + gamma * x3 + delta * x4 + y for vectors x and y of length n without having to overwrite y.
/// </summary>
/// <param name="x1">First input vector.</param>
/// <param name="x2">Second input vector.</param>
/// <param name="x3">Third input vector.</param>
/// <param name="x4">Fourth input vector.</param>
/// <param name="y">Input vector to be added.</param>
/// <param name="out">Output vector.</param>
/// <param name="alpha">Scalar multiplier for x1.</param>
/// <param name="beta">Scalar multiplier for x2.</param>
/// <param name="gamma">Scalar multiplier for x3.</param>
/// <param name="delta">Scalar multiplier for x4.</param>
/// <param name="n">Length of the vectors.</param>
/// <returns>None.</returns>
__global__ void lincomb(const T* __restrict__ x1, const T* __restrict__ x2, const T* __restrict__ x3, const T* __restrict__ x4, const T* __restrict__ y,
	T* __restrict__ out, T alpha, T beta, T gamma, T delta, size_t n) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t stride = blockDim.x * gridDim.x;
	for (; i < n; i += stride)
	{
		out[i] = alpha * x1[i] + beta * x2[i] + gamma * x3[i] + delta * x4[i] + y[i];
	}
}

template <typename T>
/// <summary>
/// Linear combination of five vectors: out = alpha * x1 + beta * x2 + gamma * x3 + delta * x4 + epsilon * x5 + y for vectors x and y of length n without having to overwrite y.
/// </summary>
/// <param name="x1">First input vector.</param>
/// <param name="x2">Second input vector.</param>
/// <param name="x3">Third input vector.</param>
/// <param name="x4">Fourth input vector.</param>
/// <param name="x5">Fifth input vector.</param>
/// <param name="y">Input vector to be added.</param>
/// <param name="out">Output vector.</param>
/// <param name="alpha">Scalar multiplier for x1.</param>
/// <param name="beta">Scalar multiplier for x2.</param>
/// <param name="gamma">Scalar multiplier for x3.</param>
/// <param name="delta">Scalar multiplier for x4.</param>
/// <param name="epsilon">Scalar multiplier for x5.</param>
/// <param name="n">Length of the vectors.</param>
/// <returns>None.</returns>
__global__ void lincomb(const T* __restrict__ x1, const T* __restrict__ x2, const T* __restrict__ x3, const T* __restrict__ x4, const T* __restrict__ x5, const T* __restrict__ y,
	T* __restrict__ out, T alpha, T beta, T gamma, T delta, T epsilon, size_t n) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t stride = blockDim.x * gridDim.x;
	for (; i < n; i += stride)
	{
		out[i] = alpha * x1[i] + beta * x2[i] + gamma * x3[i] + delta * x4[i] + epsilon * x5[i] + y[i];
	}
}

__device__ __forceinline__ double mag(cuda::std::complex<double> z) {
	return cuda::std::abs(z);
};

__device__ __forceinline__ double mag(double z) {
    return fabs(z);
};