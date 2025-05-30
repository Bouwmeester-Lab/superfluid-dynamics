#pragma once
#ifndef TIMESTEP_MANAGER_H
#define TIMESTEP_MANAGER_H


#include "ProblemProperties.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cufft.h"
#include "utilities.cuh"
#include "constants.cuh"
#include "Derivatives.cuh"
#include "createM.cuh"
#include "WaterVelocities.cuh"
#include "MatrixSolver.cuh"

template<int N>
class TimeStepManager
{
public:
	TimeStepManager(ProblemProperties& problemProperties);
	~TimeStepManager();

	/// <summary>
	/// Initializes the time step manager by copying the host data to the device memory for the initial conditions. Use this only once when setting up the problem from the host side.
	/// </summary>
	/// <param name="Z0"></param>
	/// <param name="Phi0"></param>
	void initialize_device(cufftDoubleComplex* Z0, cufftDoubleComplex* Phi0);
	void runTimeStep();
	cufftDoubleComplex* devVelocitiesLower; ///< Device pointer to the velocities array (lower fluid)
	cufftDoubleComplex* devVelocitiesUpper; ///< Device pointer to the velocities array (upper fluid)

private:
	cufftDoubleComplex* devZPhi; ///< Device pointer to the ZPhi array
	cufftDoubleComplex* devZPhiPrime; ///< Device pointer to the ZPhiPrime array
	double* devPhiPrime; ///< Device pointer to the PhiPrime array (derivative of Phi)
	cufftDoubleComplex* devZpp; ///< Device pointer to the Zpp array

	double* devM; ///< Device pointer to the matrix M (NxN, double precision)
	double* deva; ///< Device pointer to the solution vector a
	cufftDoubleComplex* devaComplex; ///< Device pointer to the solution vector a in complex form (for compatibility with velocity calculations)

	cufftDoubleComplex* devaprime; ///< Device pointer to the derivative of a

	cufftDoubleComplex* devV1; ///< Device pointer to the V1 matrix
	cufftDoubleComplex* devV2; ///< Device pointer to the V2 diagonal vector

	

	ProblemProperties& problemProperties; ///< Reference to the problem properties for configuration
	ZPhiDerivative<N> zPhiDerivative; ///< Derivative calculator for Z and Phi
	FftDerivative<N, 1> fftDerivative; ///< FFT derivative calculator for single batch
	MatrixSolver<N> matrixSolver; ///< Matrix solver for solving the vorticities.
	VelocityCalculator<N> velocityCalculator; ///< Velocity calculator for calculating the velocities based on the vorticities and matrices.
	const int threads = 256; ///< Number of threads per block for CUDA kernels
	const int blocks = (N + threads - 1) / threads; ///< Number of blocks for CUDA kernels, ensuring all elements are covered

	const dim3 matrix_threads;// (16, 16);     // 256 threads per block in 2D
	const dim3 matrix_blocks; // ((N + 15) / 16, (N + 15) / 16);


	
};

template<int N>
TimeStepManager<N>::TimeStepManager(ProblemProperties& problemProperties) : problemProperties(problemProperties), zPhiDerivative(problemProperties), 
matrix_threads(16, 16), matrix_blocks((N + 15) / 16, (N + 15) / 16)
{
	cudaMalloc(&devZPhi,2 * N * sizeof(cufftDoubleComplex));
	cudaMalloc(&devZPhiPrime, 2 * N * sizeof(cufftDoubleComplex));
	cudaMalloc(&devZpp, N * sizeof(cufftDoubleComplex));
	cudaMalloc(&devM, N * N * sizeof(double)); // Matrix M for solving the system
	cudaMalloc(&deva, N * sizeof(double));
	cudaMalloc(&devaComplex, N * sizeof(cufftDoubleComplex)); // Device pointer for the solution vector a in complex form
	cudaMalloc(&devaprime, N * sizeof(cufftDoubleComplex));
	cudaMalloc(&devV1, N * N * sizeof(cufftDoubleComplex));
	cudaMalloc(&devV2, N * sizeof(cufftDoubleComplex));
	cudaMalloc(&devVelocitiesLower, N * sizeof(cufftDoubleComplex));
	cudaMalloc(&devVelocitiesUpper, N * sizeof(cufftDoubleComplex));
}

template<int N>
TimeStepManager<N>::~TimeStepManager()
{
	cudaFree(devZPhi);
	cudaFree(devZPhiPrime);
	cudaFree(devZpp);
	cudaFree(deva);
	cudaFree(devaprime);
	cudaFree(devV1);
	cudaFree(devV2);
	cudaFree(devVelocitiesLower);
	cudaFree(devVelocitiesUpper);
}

template<int N>
inline void TimeStepManager<N>::initialize_device(cufftDoubleComplex* Z0, cufftDoubleComplex* Phi0)
{
	// Copy initial conditions to device memory
	cudaMemcpy(devZPhi, Z0, N * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
	cudaMemcpy(devZPhi + N, Phi0, N * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
	//// Initialize other device arrays as needed
	//cudaMemset(devZpp, 0, N * sizeof(cufftDoubleComplex));
	//cudaMemset(deva, 0, N * sizeof(double));
	//cudaMemset(devaprime, 0, N * sizeof(double));
	//cudaMemset(devV1, 0, N * N * sizeof(cufftDoubleComplex));
	//cudaMemset(devV2, 0, N * sizeof(cufftDoubleComplex));
	//cudaMemset(devVelocitiesLower, 0, N * sizeof(cufftDoubleComplex));
	//cudaMemset(devVelocitiesUpper, 0, N * sizeof(cufftDoubleComplex));
}

template<int N>
inline void TimeStepManager<N>::runTimeStep()
{
	// Here you would implement the logic to run a time step of the simulation.
	// This would typically involve:
	// 1. Calculating ZPhiPrime and Zpp from devZPhi.
	// 2. Create the M matrix using devZPhi, devZPhiPrime, and devZpp.
	// 3. Solve Ma = phi' to obtain the vorticities (a).
	// 4. Calculate the derivatives of a.
	// 5. Creating the V1 and V2 matrices using devZPhi, devZPhiPrime, and devZpp, a'
	// 6. Calculating the velocities for both lower and upper fluids.
	// 7. Updating the RHS of state variables (e.g., deva, devaprime) based on the calculated velocities.

	zPhiDerivative.exec(devZPhi, devZPhiPrime, devZpp); // Calculate derivatives of Z and Phi

	//cudaDeviceSynchronize();
	createMKernel << <matrix_blocks, matrix_threads >> > (devM, devZPhi, devZPhiPrime, devZpp, problemProperties.rho,  N); // Create the M matrix
	//cudaDeviceSynchronize(); // wait for the kernel to finish
	complex_to_real << <blocks, threads >> > (devZPhiPrime + N, devPhiPrime, N); // Convert ZPhiPrime to real PhiPrime (takes only the real part).
	//cudaDeviceSynchronize();
	matrixSolver.solve(devM, devPhiPrime, deva); // Solve the system Ma = phi' to get the vorticities (a)
	//cudaDeviceSynchronize(); // wait for the solver to finish
	real_to_complex << <blocks, threads >> > (deva, devaComplex, N); // Convert the real vorticities to complex form for velocity calculations
	//cudaDeviceSynchronize();
	fftDerivative.exec(devaComplex, devaprime); // Calculate the derivative of a (vorticities)
	
	velocityCalculator.calculateVelocities(devZPhi, devZPhiPrime, devZpp, devaComplex, devaprime, devV1, devV2, devVelocitiesLower, true); // Calculate the velocities based on the vorticities and matrices

	velocityCalculator.calculateVelocities(devZPhi, devZPhiPrime, devZpp, devaComplex, devaprime, devV1, devV2, devVelocitiesUpper, false); // Calculate the velocities for the upper fluid
	// 7. will be done by the caller after device synchronization.
}
#endif // TIMESTEP_MANAGER_H