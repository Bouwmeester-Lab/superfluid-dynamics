#pragma once
#ifndef BESSEL_MATRIX_KERNELS_H
#define BESSEL_MATRIX_KERNELS_H

#include "../Math/BesselGreenFunctions.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "constants.cuh"

__device__ __forceinline__ size_t besselMatrixIndex(size_t k, size_t j, size_t b, size_t N)
{
	return k + j * N + b * N * N;
}

template <size_t Nb, size_t N_collocations>
__device__ __forceinline__ double calculateSelfTermSingle(
	size_t k,
	DirichletNeumannBesselGreenFunctionsDeviceView<Nb, N_collocations>& greens,
	RadialPointers pointers,
	RadialProperties properties,
	const double* ds)
{
	return ds[k] / (2 * PI_d) * (log(16.0 * pointers.dev_r[k] / ds[k]) + 1);
}

template <size_t Nb, size_t N_collocations>
__device__ __forceinline__ double calculateSelfTermDouble(
	size_t k,
	DirichletNeumannBesselGreenFunctionsDeviceView<Nb, N_collocations>& greens,
	RadialPointers pointers,
	RadialProperties properties,
	const double* ds)
{
	const double etaprime = pointers.dev_z_prime[k];
	const double nr = - etaprime / sqrt(1 + etaprime * etaprime);
	const double dsk = ds[k];
	const double rk = pointers.dev_r[k];
	const double curvature = pointers.dev_z_pp[k] / pow(1.0 + etaprime * etaprime, 1.5); //TODO: maybe change this to doing sqrt(1.0 + etaprime *etaprime)*(1.0+etaprime*etaprime)?
	return dsk / (4.0 * PI_d) * (curvature - nr / rk * log(16.0 * rk / dsk));
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
	DirichletNeumannBesselGreenFunctionsDeviceView<Nb, N_collocations> greens,
	RadialPointers pointers,
	const double* ds,
	RadialProperties properties,
	size_t batchSize)
{
	const size_t k = blockIdx.x * blockDim.x + threadIdx.x;
	const size_t j = blockIdx.y * blockDim.y + threadIdx.y;
	const size_t b = blockIdx.z;

	if (b >= batchSize || k >= N_collocations || j >= N_collocations)
	{
		return;
	}

	const size_t nodeOffset = b * N_collocations;
	RadialPointers batchPointers{
		pointers.dev_r + nodeOffset,
		pointers.dev_z + nodeOffset,
		pointers.dev_z_prime + nodeOffset,
		pointers.dev_z_pp + nodeOffset
	};
	double green;
	double dGreenDn;
	if (k == j) 
	{
		green = calculateSelfTermSingle(k, greens, batchPointers, properties, ds + nodeOffset);
		dGreenDn = calculateSelfTermDouble(k, greens, batchPointers, properties, ds + nodeOffset) + 0.5;
	}
	else 
	{
		const double sourceMeasure = batchPointers.dev_r[j] * ds[nodeOffset + j];
		green = sourceMeasure *  greens.calculateGreenFunction(k, j, batchPointers, properties);
		dGreenDn = sourceMeasure * greens.calculatedGdn(k, j, batchPointers, properties);
	}
	const size_t matrixIndex = besselMatrixIndex(k, j, b, N_collocations);
	S[matrixIndex] =  green;
	D[matrixIndex] =  dGreenDn;
}


#endif // BESSEL_MATRIX_KERNELS_H
