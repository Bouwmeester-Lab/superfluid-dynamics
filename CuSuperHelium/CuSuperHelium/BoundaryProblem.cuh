#pragma once
#ifndef BOUNDARY_PROBLEM_H

#include "ProblemProperties.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utilities.cuh"
#include "constants.cuh"
#include "Derivatives.cuh"
#include "WaterVelocities.cuh"
#include "Energies.cuh"

struct ProblemPointers
{
public:
	const std_complex* Z; ///< Device pointer to the Z array (complex representation of the boundary)
	const std_complex* Zp; ///< Device pointer to the Zp array (derivative of Z)
	const std_complex* Zpp; ///< Device pointer to the Zpp array (second derivative of Z)
	const std_complex* Phi; ///< Device pointer to the Phi array (potential function on the boundary)
	const std_complex* VelocitiesUpper; ///< Device pointer to the velocities array (calculated velocities on the boundary)
	const std_complex* VelocitiesLower; ///< Device pointer to the velocities array (calculated velocities on the boundary)
};

template<int N, size_t batchSize>
class BoundaryProblem {
protected:
	const dim3 matrix_threads;// (16, 16);     // 256 threads per block in 2D
	const dim3 matrix_blocks; // ((N + 15) / 16, (N + 15) / 16);
	const int threads = 256; ///< Number of threads per block for CUDA kernels
	const int blocks = (batchSize * N + threads - 1) / threads; ///< Number of blocks for CUDA kernels, ensuring all elements are covered
public:
	EnergyContainer<N> energyContainer; ///< Energy container for storing the energies calculated during the simulation

	BoundaryProblem(EnergyBase<N>* kinetic, EnergyBase<N>* potential, EnergyBase<N>* surface) : matrix_threads(16, 16, 1), matrix_blocks((N + 15) / 16, (N + 15) / 16, batchSize), energyContainer(kinetic, potential, surface)
	{
	}
	virtual ~BoundaryProblem() {}
	/// <summary>
	/// reates the M matrix for the boundary integral problem.
	/// </summary>
	/// <param name="M">Output device pointer</param>
	/// <param name="Z">Input Z device pointer</param>
	/// <param name="Zp">Input Zp</param>
	/// <param name="Zpp"></param>
	/// <param name="rho"></param>
	/// <param name="n"></param>
	virtual void CreateMMatrix(double* M, const std_complex* Z, const std_complex* Zp, const std_complex* Zpp, ProblemProperties& properties) = 0;
	virtual void CalculateVelocities(const std_complex* Z,
		const std_complex* Zp,
		const std_complex* Zpp,
		std_complex* a,
		std_complex* aprime,
		std_complex* V1,
		std_complex* V2,
		std_complex* velocities,
		ProblemProperties& properties,
		bool lower) = 0;
	virtual void CalculateRhsPhi(const ProblemPointers problemPointers, std_complex* result, ProblemProperties& properties) = 0;
	virtual void CalculateEnergy(const DevicePointers& devPointers, cudaStream_t stream)
	{
		energyContainer.CalculateEnergy(devPointers, stream); ///< Calculate the energies based on the device pointers containing the state variables
	};
};

#endif // !BOUNDARY_PROBLEM_H