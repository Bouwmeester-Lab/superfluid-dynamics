#pragma once

#ifndef RADIAL_BOUNDARY_INTEGRATOR_H
#define RADIAL_BOUNDARY_INTEGRATOR_H

#include "cuda_runtime.h"
#include "cublas_v2.h"

#include <iostream>
#include <vector>

#include "Common/CudaChecks.cuh"
#include "Common/DenseLuSolver.cuh"
#include "Math/BesselGreenFunctions.cuh"
#include "BoundaryKernels/RadialMatrixKernels.cuh"
#include "Radial/BoundaryNodes.cuh"
#include "RadialModels.h"


static __global__ void calculateTangentialVelocitiesKernel(const double* __restrict__ devEtaPrime, const double* __restrict__ devPhiPrime, double* __restrict__ devTangential, const size_t N_collocations)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N_collocations)
	{
		devTangential[idx] = devPhiPrime[idx] / sqrt(1.0 + devEtaPrime[idx] * devEtaPrime[idx]);
	}
}

static __global__ void calculateVelocitiesKernel(const double* __restrict__ devNormalVelocities, const double* __restrict__ devTangentialVelocities, const double* __restrict__ devEtaPrime, double* __restrict__ devVelocitiesR, double* __restrict__ devVelocitiesZ, const size_t N_collocations)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N_collocations)
	{
		double T = sqrt(1.0 + devEtaPrime[idx] * devEtaPrime[idx]);
		double etaPrime = devEtaPrime[idx];
		double normalVelocity = devNormalVelocities[idx];
		double tangentialVelocity = devTangentialVelocities[idx];
		devVelocitiesR[idx] = (-normalVelocity * etaPrime + tangentialVelocity) / T;
		devVelocitiesZ[idx] = (normalVelocity + tangentialVelocity * etaPrime) / T;
	}
}

/// <summary>
/// Computes the surface velocities (v_r, v_z) of an axisymmetric potential flow from the surface state,
/// using the Bessel-series Green function. Sizes (surface points, Bessel modes) are runtime values.
///
/// Pipeline: surface state -> BoundaryNodes -> S, D matrices -> solve S q = (1/2 I + D) phi for the normal
/// velocity q -> combine with the tangential velocity phi'/T into (v_r, v_z).
/// </summary>
class RadialVelocityCalculator
{
public:
	RadialVelocityCalculator(size_t nSurface, size_t nModes)
		: nSurface(nSurface),
		nodes(nSurface, 0),
		besselGreenFunctions(nModes, nSurface),
		solver(static_cast<int>(nSurface))
	{
		const size_t n = nodes.total();
		CHECK_CUDA(cudaMalloc((void**)&devS, n * n * sizeof(double)));
		CHECK_CUDA(cudaMalloc((void**)&devD, n * n * sizeof(double)));
		CHECK_CUDA(cudaMalloc((void**)&devb, n * sizeof(double)));
		CHECK_CUDA(cudaMalloc((void**)&devNormalVelocities, n * sizeof(double)));
		CHECK_CUDA(cudaMalloc((void**)&devTangentialVelocities, nSurface * sizeof(double)));
		checkCublas(cublasCreate(&cublasHandle));
		blocksPerGrid = static_cast<unsigned>((nSurface + threadsPerBlock - 1) / threadsPerBlock);
	}

	~RadialVelocityCalculator()
	{
		cublasDestroy(cublasHandle);
		cudaFree(devS);
		cudaFree(devD);
		cudaFree(devb);
		cudaFree(devNormalVelocities);
		cudaFree(devTangentialVelocities);
	}

	RadialVelocityCalculator(const RadialVelocityCalculator&) = delete;
	RadialVelocityCalculator& operator=(const RadialVelocityCalculator&) = delete;

	void initialize(RadialProperties properties)
	{
		besselGreenFunctions.initialize(properties);
	}

	/// <summary>
	/// Writes v_r and v_z at the N surface points into the device arrays devVelocitiesR, devVelocitiesZ.
	/// </summary>
	void calculateVelocities(double* devVelocitiesR, double* devVelocitiesZ, const RadialSurfaceState& surface, RadialProperties properties)
	{
		nodes.buildSurface(surface, properties.R);
		const BoundaryNodes nodeView = nodes.view();
		besselGreenFunctions.calculateMatrices(nodeView);

		// create the S, D matrices used in the BI equation to obtain the normal velocities
		launchFormRadialSDMatrices(devS, devD, nodeView, besselGreenFunctions.createDeviceView(properties));
#ifdef TEST
		dumpMatrices();
#endif
		// calculate the normal velocities using the S, D matrices and the potential
		calculateNormalVelocities(surface);
		// tangential velocity from the derivative of the potential and of the height
		calculateTangentialVelocitiesKernel<<<blocksPerGrid, threadsPerBlock>>>(surface.etaPrime, surface.phiPrime, devTangentialVelocities, nSurface);
		CHECK_CUDA(cudaGetLastError());
		// transform the normal and tangential velocities to the radial and vertical velocities
		calculateVelocitiesKernel<<<blocksPerGrid, threadsPerBlock>>>(devNormalVelocities, devTangentialVelocities, surface.etaPrime, devVelocitiesR, devVelocitiesZ, nSurface);
		CHECK_CUDA(cudaGetLastError());
	}

private:
	size_t nSurface;
	BoundaryNodeBuffers nodes;
	DirichletNeumannBesselGreenFunctions besselGreenFunctions;
	DenseLuSolver solver;
	cublasHandle_t cublasHandle = nullptr;

	double* devS = nullptr;
	double* devD = nullptr;
	double* devb = nullptr;
	double* devNormalVelocities = nullptr;
	double* devTangentialVelocities = nullptr;

	static constexpr unsigned threadsPerBlock = 256;
	unsigned blocksPerGrid = 1;

	/// <summary>
	/// b = D phi (D already carries the 1/2 jump on its diagonal), then S q = b.
	/// </summary>
	void calculateNormalVelocities(const RadialSurfaceState& surface)
	{
		const int n = static_cast<int>(nodes.total());
		const double alpha = 1.0;
		const double beta = 0.0;
		checkCublas(cublasDgemv(cublasHandle, CUBLAS_OP_N, n, n, &alpha, devD, n, surface.phi, 1, &beta, devb, 1));

		solver.solve(devS, devb, devNormalVelocities);
		CHECK_CUDA(cudaGetLastError());
	}

#ifdef TEST
	void dumpMatrices()
	{
		CHECK_CUDA(cudaDeviceSynchronize());
		const size_t n = nodes.total();
		std::vector<double> hostS(n * n);
		std::vector<double> hostD(n * n);
		CHECK_CUDA(cudaMemcpy(hostS.data(), devS, sizeof(double) * n * n, cudaMemcpyDeviceToHost));
		CHECK_CUDA(cudaMemcpy(hostD.data(), devD, sizeof(double) * n * n, cudaMemcpyDeviceToHost));

		std::cout << "Matrix S, D calculated" << std::endl;
		for (size_t i = 0; i < n; ++i)
		{
			std::cout << "[";
			for (size_t j = 0; j < n; ++j)
			{
				const size_t index = i + j * n;
				std::cout << "(" << hostS[index] << ", " << hostD[index] << "), ";
			}
			std::cout << "]" << std::endl;
		}
		std::cout << "In TEST block" << std::endl;
	}
#endif
};


#endif // RADIAL_BOUNDARY_INTEGRATOR_H
