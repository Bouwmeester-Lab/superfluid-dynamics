#pragma once

#ifndef RADIAL_BOUNDARY_INTEGRATOR_H
#define RADIAL_BOUNDARY_INTEGRATOR_H

#include "cuda_runtime.h"
#include "Math/BesselGreenFunctions.cuh"
#include "BoundaryKernels/BesselMatrixKernels.cuh"
#include "cublas_v2.h"
#include "MatrixSolver.cuh"
#include "RadialModels.h"


struct RadialWorkspacePointers {
	double* devS;
	double* devD;
	double* devDeltaS;
	double* devb;

	double* devNormalVelocities;
	double* devTangentialVelocities;

	__host__ void allocate(size_t N_collocations) {
		checkCuda(cudaMalloc((void**)&devS, N_collocations * N_collocations * sizeof(double)), __FUNCTION__, __FILE__, __LINE__);
		checkCuda(cudaMalloc((void**)&devD, N_collocations * N_collocations * sizeof(double)), __FUNCTION__, __FILE__, __LINE__);
		checkCuda(cudaMalloc((void**)&devDeltaS, N_collocations * sizeof(double)), __FUNCTION__, __FILE__, __LINE__);
		checkCuda(cudaMalloc((void**)&devb, N_collocations * sizeof(double)), __FUNCTION__, __FILE__, __LINE__);

		checkCuda(cudaMalloc((void**)&devNormalVelocities, N_collocations * sizeof(double)), __FUNCTION__, __FILE__, __LINE__);
		checkCuda(cudaMalloc((void**)&devTangentialVelocities, N_collocations * sizeof(double)), __FUNCTION__, __FILE__, __LINE__);
	}

	__host__ void free() {
		checkCuda(cudaFree(devS), __FUNCTION__, __FILE__, __LINE__);
		checkCuda(cudaFree(devD), __FUNCTION__, __FILE__, __LINE__);
		checkCuda(cudaFree(devDeltaS), __FUNCTION__, __FILE__, __LINE__);
		checkCuda(cudaFree(devb), __FUNCTION__, __FILE__, __LINE__);
		checkCuda(cudaFree(devNormalVelocities), __FUNCTION__, __FILE__, __LINE__);
		checkCuda(cudaFree(devTangentialVelocities), __FUNCTION__, __FILE__, __LINE__);
	}
};


__global__ void calculateTangentialVelocitiesKernel(const double* __restrict__ devEtaPrime, const double* __restrict__ devPhiPrime, double* __restrict__ devTangential, const size_t N_collocations)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N_collocations) 
	{
		devTangential[idx] = devPhiPrime[idx] / sqrt(1.0 + devEtaPrime[idx] * devEtaPrime[idx]);
	}
}

__global__ void calculateVelocitiesKernel(const double* __restrict__ devNormalVelocities, const double* __restrict__ devTangentialVelocities, const double* __restrict__ devEtaPrime, double* __restrict__ devVelocitiesR, double* __restrict__ devVelocitiesZ, const size_t N_collocations)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
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



template <size_t Nb, size_t N_collocations>
class RadialVelocityCalculator
{
public:
	RadialVelocityCalculator();
	~RadialVelocityCalculator();
	
	void calculateVelocities(double* devVelocitiesR, double* devVelocitiesZ, RadialPointers pointers, RadialProperties properties);

private:
	RadialWorkspacePointers workspacePointers;
	DirichletNeumannBesselGreenFunctions<Nb, N_collocations> besselGreenFunctions;
	
	size_t threadsPerBlock = 256;
	size_t blocksPerGrid = (N_collocations + threadsPerBlock - 1) / threadsPerBlock;

	dim3 matrixThreadsPerBlock = dim3(16, 16);
	dim3 matrixBlocksPerGrid = dim3((N_collocations + 15) / 16, (N_collocations + 15) / 16);
	
	

	cublasHandle_t cublasHandle;
	MatrixSolver<N_collocations, 1> matrixSolver;

	void createRadialMatrices(RadialPointers pointers, RadialProperties properties);
	void calculateNormalVelocities(double* devNormalVelocities, RadialProperties properties);
	void calculateTangentialVelocities(double* devTangentialVelocities, RadialPointers pointers, RadialProperties properties);
};

template <size_t Nb, size_t N_collocations>
RadialVelocityCalculator<Nb, N_collocations>::RadialVelocityCalculator()
{
	// allocate the memory used for the matrices S and D on the device
	workspacePointers.allocate(N_collocations);


	checkCublas(cublasCreate(&cublasHandle));
}

template <size_t Nb, size_t N_collocations>
RadialVelocityCalculator<Nb, N_collocations>::~RadialVelocityCalculator()
{
	checkCublas(cublasDestroy(cublasHandle));
	workspacePointers.free();
}

template<size_t Nb, size_t N_collocations>
void RadialVelocityCalculator<Nb, N_collocations>::calculateVelocities(double* devVelocitiesR, double* devVelocitiesZ, RadialPointers pointers, RadialProperties properties)
{
	// create the S, D matrices used in the BI equation to obtain the normal velocities
	this->createRadialMatrices(pointers, properties);
	// calculate the normal velocities using the S, D matrices and the potential
	calculateNormalVelocities(workspacePointers.devNormalVelocities, properties);
	// calculate the tangential velocities using the derivative of the potential and the derivative of the height
	calculateTangentialVelocities(workspacePointers.devTangentialVelocities, pointers, properties);
	// transform the normal and tangential velocities to the radial and vertical velocities
	calculateVelocitiesKernel << <blocksPerGrid, threadsPerBlock >> > (workspacePointers.devNormalVelocities, workspacePointers.devTangentialVelocities, pointers.dev_z_prime, devVelocitiesR, devVelocitiesZ, N_collocations);
}

template<size_t Nb, size_t N_collocations>
void RadialVelocityCalculator<Nb, N_collocations>::createRadialMatrices(RadialPointers pointers,  RadialProperties properties)
{
	formBesselSDMatrices<<<matrixBlocksPerGrid, matrixThreadsPerBlock>>>(workspacePointers.devS, workspacePointers.devD, besselGreenFunctions, pointers, workspacePointers.devDeltaS, properties, 1);
}

template<size_t Nb, size_t N_collocations>
void RadialVelocityCalculator<Nb, N_collocations>::calculateNormalVelocities(double* devNormalVelocities, RadialProperties properties)
{
	createRadialMatrices(properties);
	double alpha = -1.0;
	double beta = 0.0;
	// calculate the b vector: D * phi, to solve: Sq + D phi = 0 for q
	cublasDgemv(cublasHandle, CUBLAS_OP_N, N_collocations, N_collocations, &alpha, workspacePointers.devD, N_collocations, workspacePointers.devDeltaS, 1, &beta, workspacePointers.devb, 1);

	matrixSolver.solve(workspacePointers.devS, workspacePointers.devb, devNormalVelocities);
}

template<size_t Nb, size_t N_collocations>
void RadialVelocityCalculator<Nb, N_collocations>::calculateTangentialVelocities(double* devTangentialVelocities, RadialPointers pointers, RadialProperties properties)
{
	calculateTangentialVelocitiesKernel <<<blocksPerGrid, threadsPerBlock >>> (pointers.dev_z_prime, pointers.devPhiPrime, devTangentialVelocities, N_collocations);
}


#endif // RADIAL_BOUNDARY_INTEGRATOR_H