#pragma once
#ifndef BESSEL_GREEN_FUNCTIONS_H
#define BESSEL_GREEN_FUNCTIONS_H

#include "../Common/CudaChecks.cuh"

#include "cuda_runtime.h"
#include <vector>
#include <stdexcept>
#include <boost/math/special_functions/bessel.hpp>
#include <iterator>

#include "../RadialModels.h"
#include "../Radial/BoundaryNodes.cuh"
#include "../Radial/SelfTerms.cuh"

/// <summary>
/// B[n * nNodes + j] = J0(kappa_n r_j)
/// </summary>
static __global__ void calculateBnMatrix(const double* dev_r, const double* devKappa, double* Bn, size_t Nb, size_t N_collocations)
{
	size_t k = blockIdx.x * blockDim.x + threadIdx.x; // bessel function n
	size_t j = blockIdx.y * blockDim.y + threadIdx.y; // collocation point j
	if (j < N_collocations && k < Nb)
	{
		Bn[k * N_collocations + j] = j0(devKappa[k] * dev_r[j]);
	}
}

/// <summary>
/// B'[n * nNodes + j] = J1(kappa_n r_j)
/// </summary>
static __global__ void calculateJ1CollocationMatrix(const double* dev_r, const double* devKappa, double* BPrimeN, size_t Nb, size_t N_collocations)
{
	size_t k = blockIdx.x * blockDim.x + threadIdx.x; // bessel function n
	size_t j = blockIdx.y * blockDim.y + threadIdx.y; // collocation point j
	if (j < N_collocations && k < Nb)
	{
		BPrimeN[k * N_collocations + j] = j1(devKappa[k] * dev_r[j]);
	}
}

/// <summary>
/// Device view of the Dirichlet (r = R) / Neumann (z = -depth) Bessel-series Green function
///   G = sum_n W_n J0(kappa_n r) J0(kappa_n rho) [exp(-kappa_n |z - eta|) + exp(-kappa_n (z + eta + 2 depth))],
/// with kappa_n = beta_n / R, J0(beta_n) = 0 and W_n = 1 / (R^2 J1(beta_n)^2 kappa_n).
/// Implements the radial kernel interface used by formRadialSDMatrices: G, dGdn, selfSingle, selfDouble.
/// k is the field node, j the source node.
/// </summary>
class DirichletNeumannBesselGreenFunctionsDeviceView
{
public:
	__host__ __device__ DirichletNeumannBesselGreenFunctionsDeviceView(const double* devKappa, const double* devWn, const double* devBn, const double* devJ1, size_t nModes, size_t nNodes, double depth)
		: devKappa(devKappa), devWn(devWn), devBn(devBn), devJ1(devJ1), nModes(nModes), nNodes(nNodes), depth(depth) {}

	/// <summary>
	/// Truncated series for G at k != j. The self term is handled by selfSingle.
	/// </summary>
	__device__ __inline__ double G(const BoundaryNodes& nodes, size_t k, size_t j) const
	{
		double sum = 0.0;
		for (size_t n = 0; n < nModes; ++n)
		{
			sum += getBnj(n, k) * getBnj(n, j) * devWn[n] * calculate_g(nodes, k, j, n);
		}
		return sum;
	}

	/// <summary>
	/// Truncated series for the normal derivative of G with respect to the source node j, at k != j.
	/// </summary>
	__device__ __inline__ double dGdn(const BoundaryNodes& nodes, size_t k, size_t j) const
	{
		double sumr = 0.0;
		double sumz = 0.0;
		for (size_t n = 0; n < nModes; ++n)
		{
			sumr -= devWn[n] * calculate_g(nodes, k, j, n) * getBPrimenj(n, j) * devKappa[n] * getBnj(n, k);
			sumz += devWn[n] * getBnj(n, k) * getBnj(n, j) * calculate_g_prime(nodes, k, j, n);
		}
		return nodes.nr[j] * sumr + nodes.nz[j] * sumz;
	}

	__device__ __inline__ double selfSingle(const BoundaryNodes& nodes, size_t k) const
	{
		return RadialSelfTerms::single(nodes, k);
	}

	__device__ __inline__ double selfDouble(const BoundaryNodes& nodes, size_t k) const
	{
		return RadialSelfTerms::doubleLayer(nodes, k);
	}

private:
	const double* devKappa; // kappa_n = beta_n / R
	const double* devWn;    // W_n = 1/(R^2 J1(beta_n)^2 kappa_n)
	const double* devBn;    // J0(kappa_n r_j), laid out n * nNodes + j
	const double* devJ1;    // J1(kappa_n r_j), laid out n * nNodes + j
	size_t nModes;
	size_t nNodes;
	double depth;

	__device__ __inline__ double getBnj(size_t n, size_t j) const { return devBn[n * nNodes + j]; }
	__device__ __inline__ double getBPrimenj(size_t n, size_t j) const { return devJ1[n * nNodes + j]; }

	/// <summary>
	/// Vertical factor of mode n: direct term plus the image below the bottom z = -depth.
	/// </summary>
	__device__ __inline__ double calculate_g(const BoundaryNodes& nodes, size_t k, size_t j, size_t n) const
	{
		return exp(-1.0 * devKappa[n] * fabs(nodes.z[k] - nodes.z[j])) + exp(-devKappa[n] * (nodes.z[k] + nodes.z[j] + 2.0 * depth));
	}

	/// <summary>
	/// Derivative of the vertical factor with respect to the source height z_j.
	/// </summary>
	__device__ __inline__ double calculate_g_prime(const BoundaryNodes& nodes, size_t k, size_t j, size_t n) const
	{
		const double kappa = devKappa[n];
		const double zk = nodes.z[k];
		const double zj = nodes.z[j];

		if (k == j)
		{
			return 0.0; // self term is handled by selfDouble
		}

		const double dz = zk - zj;
		const double direct = exp(-kappa * fabs(dz));
		const double image = exp(-kappa * (zk + zj + 2.0 * depth));

		if (dz > 0.0)
		{
			return kappa * (direct - image);
		}
		else if (dz < 0.0)
		{
			return -kappa * (direct + image);
		}
		else
		{
			// k != j but same height: symmetric / PV convention.
			return -kappa * image;
		}
	}
};

/// <summary>
/// Owns the mode data (kappa_n, W_n) and the per-node Bessel tables J0(kappa_n r_j), J1(kappa_n r_j).
/// The number of modes and of nodes are runtime values. The series encodes the wall r = R, so it only
/// supports boundaries without wall nodes.
/// </summary>
class DirichletNeumannBesselGreenFunctions
{
public:
	DirichletNeumannBesselGreenFunctions(size_t nModes, size_t nNodes)
		: nModes(nModes), nNodes(nNodes)
	{
		if (nModes == 0 || nNodes == 0)
		{
			throw std::invalid_argument("DirichletNeumannBesselGreenFunctions requires at least one mode and one node");
		}
		CHECK_CUDA(cudaMalloc((void**)&devKappa, nModes * sizeof(double)));
		CHECK_CUDA(cudaMalloc((void**)&devWn, nModes * sizeof(double)));
		CHECK_CUDA(cudaMalloc((void**)&devBn, nNodes * nModes * sizeof(double)));
		CHECK_CUDA(cudaMalloc((void**)&devJ1, nNodes * nModes * sizeof(double)));
		matrixBlocks = dim3(static_cast<unsigned>((nModes + 15) / 16), static_cast<unsigned>((nNodes + 15) / 16));
	}

	~DirichletNeumannBesselGreenFunctions()
	{
		cudaFree(devKappa);
		cudaFree(devWn);
		cudaFree(devBn);
		cudaFree(devJ1);
	}

	DirichletNeumannBesselGreenFunctions(const DirichletNeumannBesselGreenFunctions&) = delete;
	DirichletNeumannBesselGreenFunctions& operator=(const DirichletNeumannBesselGreenFunctions&) = delete;

	/// <summary>
	/// Computes kappa_n = beta_n / R from the zeros beta_n of J0, and W_n = 1/(R^2 J1(beta_n)^2 kappa_n).
	/// </summary>
	void initialize(RadialProperties properties)
	{
		const double R = properties.R;

		std::vector<double> zeros_J0_host;
		std::vector<double> kappa_host;
		std::vector<double> Wn_host;
		boost::math::cyl_bessel_j_zero(0.0, 1, static_cast<unsigned>(nModes), std::back_inserter(zeros_J0_host));

		for (size_t n = 0; n < nModes; ++n)
		{
			kappa_host.push_back(zeros_J0_host[n] / R);
			const double j1 = boost::math::cyl_bessel_j(1.0, zeros_J0_host[n]);
			Wn_host.push_back(1.0 / (R * R * j1 * j1 * kappa_host[n]));
		}

		CHECK_CUDA(cudaMemcpy(devKappa, kappa_host.data(), nModes * sizeof(double), cudaMemcpyHostToDevice));
		CHECK_CUDA(cudaMemcpy(devWn, Wn_host.data(), nModes * sizeof(double), cudaMemcpyHostToDevice));
	}

	/// <summary>
	/// Recomputes the Bessel tables at the current node radii.
	/// </summary>
	void calculateMatrices(const BoundaryNodes& nodes)
	{
		if (nodes.nWall != 0)
		{
			throw std::invalid_argument("The Bessel-series Green function encodes the wall: it does not support wall nodes");
		}
		if (nodes.total() != nNodes)
		{
			throw std::invalid_argument("Node count does not match the Bessel Green function tables");
		}
		calculateBnMatrix<<<matrixBlocks, matrixThreads>>>(nodes.r, devKappa, devBn, nModes, nNodes);
		CHECK_CUDA(cudaGetLastError());
		calculateJ1CollocationMatrix<<<matrixBlocks, matrixThreads>>>(nodes.r, devKappa, devJ1, nModes, nNodes);
		CHECK_CUDA(cudaGetLastError());
	}

	DirichletNeumannBesselGreenFunctionsDeviceView createDeviceView(RadialProperties properties) const
	{
		return DirichletNeumannBesselGreenFunctionsDeviceView(devKappa, devWn, devBn, devJ1, nModes, nNodes, properties.depth);
	}

	size_t modeCount() const { return nModes; }

private:
	size_t nModes;
	size_t nNodes;

	double* devKappa = nullptr;
	double* devWn = nullptr;
	double* devBn = nullptr;
	double* devJ1 = nullptr;

	dim3 matrixThreads = dim3(16, 16);
	dim3 matrixBlocks;
};

#endif
