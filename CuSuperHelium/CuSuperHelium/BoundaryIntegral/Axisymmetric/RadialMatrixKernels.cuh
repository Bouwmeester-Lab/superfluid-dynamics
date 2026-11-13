#pragma once
#ifndef RADIAL_MATRIX_KERNELS_H
#define RADIAL_MATRIX_KERNELS_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "BoundaryIntegral/Axisymmetric/BoundaryNodes.cuh"

/// <summary>
/// Forms the axisymmetric single-layer matrix S and double-layer matrix D on all boundary nodes:
///   S_kj = r_j ds_j G(k, j),  D_kj = r_j ds_j dG/dn_j(k, j)       for k != j
///   S_kk = selfSingle(k),     D_kk = selfDouble(k) + 1/2          (1/2: jump of the double layer on a smooth boundary)
/// Column-major n x n with n = nodes.total(): entry (k, j) at k + j * n. k is the field node, j the source node.
///
/// KernelView is the Green function: any type with the device methods
///   double G(const BoundaryNodes&, size_t k, size_t j) const
///   double dGdn(const BoundaryNodes&, size_t k, size_t j) const
///   double selfSingle(const BoundaryNodes&, size_t k) const
///   double selfDouble(const BoundaryNodes&, size_t k) const
/// </summary>
template <typename KernelView>
static __global__ void formRadialSDMatrices(double* S, double* D, BoundaryNodes nodes, KernelView greens)
{
	const size_t k = blockIdx.x * blockDim.x + threadIdx.x;
	const size_t j = blockIdx.y * blockDim.y + threadIdx.y;
	const size_t n = nodes.total();

	if (k >= n || j >= n)
	{
		return;
	}

	double green;
	double dGreenDn;
	if (k == j) // handles the singular case when the source = field point. The solving of the singularity is left to "greens" and abstracted away.
	{
		green = greens.selfSingle(nodes, k);
		dGreenDn = greens.selfDouble(nodes, k) + 0.5; // TODO: 0.5 comes from c(P) in the PV integral. This IS AN ASSUMPTION OF SMOOTH SURFACE. THIS WILL NEED TO BE REVISITED FOR GENERAL CASES.
	}
	else // handles the source != field case. See Slide 269 PowerPoint "Solitons SuperFluid Helium"
	{
		const double sourceMeasure = nodes.r[j] * nodes.ds[j];
		green = sourceMeasure * greens.G(nodes, k, j);
		dGreenDn = sourceMeasure * greens.dGdn(nodes, k, j);
	}
	const size_t matrixIndex = k + j * n;
	S[matrixIndex] = green;
	D[matrixIndex] = dGreenDn;
}

/// <summary>
/// Launches formRadialSDMatrices on an n x n grid of 16 x 16 blocks.
/// </summary>
template <typename KernelView>
inline void launchFormRadialSDMatrices(double* S, double* D, const BoundaryNodes& nodes, const KernelView& greens)
{
	const size_t n = nodes.total();
	const dim3 threads(16, 16);
	const dim3 blocks(static_cast<unsigned>((n + 15) / 16), static_cast<unsigned>((n + 15) / 16));
	formRadialSDMatrices<KernelView><<<blocks, threads>>>(S, D, nodes, greens);
	CUSH_CHECK_LAUNCH();
}

#endif // RADIAL_MATRIX_KERNELS_H
