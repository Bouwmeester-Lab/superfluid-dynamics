#pragma once
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cuda/std/complex>

#ifdef __INTELLISENSE__ // for Visual Studio IntelliSense https://stackoverflow.com/questions/77769389/intellisense-in-visual-studio-cannot-find-cuda-cooperative-groups-namespace
#define __CUDACC__
#endif // __INTELLISENSE__
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#ifdef __INTELLISENSE__
#undef __CUDACC__
#endif // __INTELLISENSE__
namespace cg = cooperative_groups;

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

__device__ double mag(cuda::std::complex<double> z) {
	return cuda::std::abs(z);
};

__device__ double mag(double z) {
    return fabs(z);
};


template<int BLOCK_SIZE, typename Op, typename Type>
__device__ Type block_reduce_sum(Type local) {
	auto block = cg::this_thread_block();
	auto warp = cg::tiled_partition<32>(block);       // 32-thread tile

	// 1) reduce within each warp
	Type warp_sum = cg::reduce(warp, local, Op());

	// 2) write one value per warp to shared mem
	__shared__ Type smem[BLOCK_SIZE / 32];
	if (warp.thread_rank() == 0)
		smem[warp.meta_group_rank()] = warp_sum;
	block.sync();

	// 3) first warp reduces the per-warp sums
	Type block_sum = 0.0;
	if (warp.meta_group_rank() == 0) {
		int num_warps = (BLOCK_SIZE + 31) / 32;
		Type v = (warp.thread_rank() < num_warps) ? smem[warp.thread_rank()] : 0.0;
		block_sum = cg::reduce(warp, v, Op());  // result in lane 0 of this warp
	}
	return (warp.meta_group_rank() == 0 && warp.thread_rank() == 0) ? block_sum : 0.0;
}


template <class T>
__global__ void rk45_error_and_y5(
    const T* __restrict__ y,
    const T* __restrict__ k1, const T* __restrict__ k2, const T* __restrict__ k3,
    const T* __restrict__ k4, const T* __restrict__ k5, const T* __restrict__ k6,
    T* __restrict__ y5_out,                // write y5 here (your yTemp)
    double h,
    double atol, double rtol,
    // weights:
    double b5_1, double b5_2, double b5_3, double b5_4, double b5_5, double b5_6,
    double d_1, double d_2, double d_3, double d_4, double d_5, double d_6,
    // reduction outputs
    double* __restrict__ block_sumsq,      // length = gridDim.x
    unsigned int N)
{
    double local_sumsq = 0.0;

    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        i < N; i += blockDim.x * gridDim.x)
    {
        // y5
        T y5 = y[i]
            + (T)(h * b5_1) * k1[i] + (T)(h * b5_2) * k2[i] + (T)(h * b5_3) * k3[i]
            + (T)(h * b5_4) * k4[i] + (T)(h * b5_5) * k5[i] + (T)(h * b5_6) * k6[i];

        y5_out[i] = y5;

        // e = h * sum d_j k_j
        T e = (T)(h * d_1) * k1[i] + (T)(h * d_2) * k2[i] + (T)(h * d_3) * k3[i]
            + (T)(h * d_4) * k4[i] + (T)(h * d_5) * k5[i] + (T)(h * d_6) * k6[i];

        double sc = atol + rtol * fmax(mag(y[i]), mag(y5));
        // guard against sc=0 if both are exactly zero
        sc = fmax(sc, 1e-300);

        double z = mag(e) / sc;
        local_sumsq += z * z;
    }

    // warp- then block-reduce local_sumsq (simple shared-mem reduce shown)
    __shared__ double sh[256]; // adjust to blockDim.x
    int lane = threadIdx.x;
    sh[lane] = local_sumsq;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (lane < s) sh[lane] += sh[lane + s];
        __syncthreads();
    }
    if (lane == 0) block_sumsq[blockIdx.x] = sh[0];
}