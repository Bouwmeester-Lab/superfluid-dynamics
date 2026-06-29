#pragma once
#ifndef BESSEL_MATRIX_KERNELS_H
#define BESSEL_MATRIX_KERNELS_H

#include "../Math/BesselGreenFunctions.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__device__ __forceinline__ size_t besselMatrixIndex(size_t k, size_t j, size_t b, size_t N)
{
	return k + j * N + b * N * N;
}

template <size_t Nb, size_t N_collocations>
__device__ __forceinline__ double calculateBesselGreenSelfTerm(
	size_t k,
	DirichletNeumannBesselGreenFunctions<Nb, N_collocations>& greens,
	RadialPointers pointers,
	RadialProperties properties)
{
	return greens.calculateGreenFunction(k, k, pointers, properties);
}

template <size_t Nb, size_t N_collocations>
__device__ __forceinline__ double calculateBesselDGreenDnSelfTerm(
	size_t k,
	DirichletNeumannBesselGreenFunctions<Nb, N_collocations>& greens,
	RadialPointers pointers,
	RadialProperties properties)
{
	return greens.calculatedGdn(k, k, pointers, properties);
}

/// <summary>
/// Forms the axisymmetric single-layer matrix S and double-layer matrix D.
/// Entries use column-major indexing: k + j * N + b * N * N, where k is the
/// field point, j is the source point, and b is the batch index.
/// </summary>
template <size_t Nb, size_t N_collocations>
static __global__ void formBesselSDMatrices(
	double* S,
	double* D,
	DirichletNeumannBesselGreenFunctions<Nb, N_collocations> greens,
	RadialPointers pointers,
	const double* ds,
	RadialProperties properties,
	size_t batchSize)
{
	const size_t k = blockIdx.x * blockDim.x + threadIdx.x;
	const size_t j = blockIdx.y * blockDim.y + threadIdx.y;
	const size_t b = blockIdx.z;
	const double depth = properties.depth;

	if (b >= batchSize || k >= N_collocations || j >= N_collocations)
	{
		return;
	}

	const size_t nodeOffset = b * N_collocations;
	RadialPointers batchPointers{
		pointers.dev_r + nodeOffset,
		pointers.dev_z + nodeOffset,
		pointers.dev_z_prime + nodeOffset
	};

	const double sourceMeasure = batchPointers.dev_r[j] * ds[nodeOffset + j];
	const double green = (k == j)
		? calculateBesselGreenSelfTerm(k, greens, batchPointers, properties)
		: greens.calculateGreenFunction(k, j, batchPointers, properties);
	const double dGreenDn = (k == j)
		? calculateBesselDGreenDnSelfTerm(k, greens, batchPointers, properties)
		: greens.calculatedGdn(k, j, batchPointers, properties);

	const size_t matrixIndex = besselMatrixIndex(k, j, b, N_collocations);
	S[matrixIndex] = sourceMeasure * green;
	D[matrixIndex] = sourceMeasure * dGreenDn;
}


#endif // BESSEL_MATRIX_KERNELS_H
