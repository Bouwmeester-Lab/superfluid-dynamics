#pragma once
#ifndef RADIAL_BOUNDARY_NODES_H
#define RADIAL_BOUNDARY_NODES_H

#include "cuda_runtime.h"
#include <cmath>
#include <stdexcept>

#include "Core/CudaChecks.cuh"
#include "BoundaryIntegral/Axisymmetric/RadialModels.h"

/// <summary>
/// Device view of the collocation nodes on the boundary of an axisymmetric domain, in the meridional (r, z) plane.
/// Struct of arrays: every pointer is a device array of length total() = nSurface + nWall.
///
/// The boundary condition of a node is given by its index range, not stored per node:
///   [0, nSurface)               free surface: phi known, q = dphi/dn unknown
///   [nSurface, nSurface+nWall)  wall r = R:   q = 0 known, phi unknown
/// </summary>
struct BoundaryNodes
{
	double* r;          // radial coordinate of the node
	double* z;          // vertical coordinate of the node
	double* nr;         // r component of the unit outward normal
	double* nz;         // z component of the unit outward normal
	double* ds;         // arclength of the cell around the node (meridional, without the r weight)
	double* curvature;  // meridional curvature of the boundary at the node (used by the double-layer self term)

	size_t nSurface;
	size_t nWall;

	__host__ __device__ size_t total() const { return nSurface + nWall; }
};

/// <summary>
/// Fills the surface nodes [0, N) of `nodes` from the surface state z = eta(r).
/// Normal (-eta', 1)/T, ds = |dr| T and curvature eta''/T^3 with T = sqrt(1 + eta'^2).
/// The cell widths dr assume cell-centred points: the first cell starts at r = 0 and the last ends at r = R.
/// </summary>
static __global__ void buildSurfaceNodesKernel(RadialSurfaceState surface, double R, BoundaryNodes nodes)
{
	const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const size_t N = nodes.nSurface;
	if (idx >= N)
	{
		return;
	}

	const double etaPrime = surface.etaPrime[idx];
	const double T2 = 1.0 + etaPrime * etaPrime;
	const double T = sqrt(T2);

	double dr = 0.0;
	if (N > 1)
	{
		if (idx == 0)
		{
			// cell goes from r = 0 to the midpoint between r[0] and r[1]
			dr = 0.5 * (surface.r[0] + surface.r[1]);
		}
		else if (idx == N - 1)
		{
			// cell goes from the midpoint between the last two points to r = R
			dr = R - 0.5 * (surface.r[N - 2] + surface.r[N - 1]);
		}
		else
		{
			// cell goes from the left midpoint to the right midpoint
			dr = 0.5 * (surface.r[idx + 1] - surface.r[idx - 1]);
		}
	}

	nodes.r[idx] = surface.r[idx];
	nodes.z[idx] = surface.eta[idx];
	nodes.nr[idx] = -etaPrime / T;
	nodes.nz[idx] = 1.0 / T;
	nodes.ds[idx] = fabs(dr) * T;
	nodes.curvature[idx] = surface.etaPrimePrime[idx] / pow(T2, 1.5);
}

/// <summary>
/// Owns the device arrays behind a BoundaryNodes view. Sizes are runtime values.
/// </summary>
class BoundaryNodeBuffers
{
public:
	BoundaryNodeBuffers(size_t nSurface, size_t nWall)
		: nSurface(nSurface), nWall(nWall)
	{
		if (nSurface == 0)
		{
			throw std::invalid_argument("BoundaryNodeBuffers requires at least one surface node");
		}
		const size_t bytes = (nSurface + nWall) * sizeof(double);
		CHECK_CUDA(cudaMalloc((void**)&devR, bytes));
		CHECK_CUDA(cudaMalloc((void**)&devZ, bytes));
		CHECK_CUDA(cudaMalloc((void**)&devNr, bytes));
		CHECK_CUDA(cudaMalloc((void**)&devNz, bytes));
		CHECK_CUDA(cudaMalloc((void**)&devDs, bytes));
		CHECK_CUDA(cudaMalloc((void**)&devCurvature, bytes));
	}

	~BoundaryNodeBuffers()
	{
		cudaFree(devR);
		cudaFree(devZ);
		cudaFree(devNr);
		cudaFree(devNz);
		cudaFree(devDs);
		cudaFree(devCurvature);
	}

	BoundaryNodeBuffers(const BoundaryNodeBuffers&) = delete;
	BoundaryNodeBuffers& operator=(const BoundaryNodeBuffers&) = delete;

	BoundaryNodes view() const
	{
		return BoundaryNodes{ devR, devZ, devNr, devNz, devDs, devCurvature, nSurface, nWall };
	}

	size_t total() const { return nSurface + nWall; }

	/// <summary>
	/// Fills the surface nodes from the surface state. Wall nodes are not touched.
	/// </summary>
	void buildSurface(const RadialSurfaceState& surface, double R)
	{
		if (surface.n != nSurface)
		{
			throw std::invalid_argument("Surface state size does not match the number of surface nodes");
		}
		constexpr unsigned threads = 256;
		const unsigned blocks = static_cast<unsigned>((nSurface + threads - 1) / threads);
		buildSurfaceNodesKernel<<<blocks, threads>>>(surface, R, view());
		CHECK_CUDA(cudaGetLastError());
	}

private:
	size_t nSurface;
	size_t nWall;

	double* devR = nullptr;
	double* devZ = nullptr;
	double* devNr = nullptr;
	double* devNz = nullptr;
	double* devDs = nullptr;
	double* devCurvature = nullptr;
};

#endif // RADIAL_BOUNDARY_NODES_H
