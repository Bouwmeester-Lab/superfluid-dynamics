#pragma once
#ifndef SIMPLE_EULER_H
#define SIMPLE_EULER_H

#include "ProblemProperties.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "AutonomousProblem.h"
#include "cublas_v2.h"
#include "cuComplex.h"

template<int N>
class Euler
{
public:
	Euler(BoundaryIntegralCalculator<N>& timeStepManager, double time_step = 0.001);
	~Euler();
	void doTimeStep();
	/// <summary>
	/// Sets the device pointer to the state vector Z (X + i Y).
	/// </summary>
	/// <param name="devZ"></param>
	void setDevZ(cuDoubleComplex* devZ) { this->devZ = devZ; }
	/// <summary>
	/// Set the device pointer to the Phi vector, which is used in the time-stepping process.
	/// </summary>
	/// <param name="devPhi"></param>
	void setDevPhi(cuDoubleComplex* devPhi) { this->devPhi = devPhi; }

private:
	BoundaryIntegralCalculator<N>& timeStepManager; ///< Instance of the TimeStepManager to handle time-stepping operations

	cuDoubleComplex time_step;
	double time_step_real; ///< Real part of the time step for convenience

	cuDoubleComplex* devZ; ///< Device pointer to the state vector X + i Y
	cuDoubleComplex* devPhi; ///< Device pointer to the Phi vector

	cublasHandle_t handle; ///< CUBLAS handle for matrix operations
};

template <int N>
Euler<N>::Euler(BoundaryIntegralCalculator<N>& timeStepManager, double time_step) : timeStepManager(timeStepManager), time_step(make_cuDoubleComplex(time_step, 0)), time_step_real(time_step)
{
	cublasCreate(&handle);
	
}

template <int N>
Euler<N>::~Euler()
{
	cublasDestroy(handle);
	// Free device memory if allocated
	if (devZ) cudaFree(devZ);
}

template<int N>
void Euler<N>::doTimeStep()
{
	
	// Perform a single time step using the Euler method
	// x_n+1 = x_n + h * v_n -> x_n is a N dimensional vector, h is the time step, v_n is the velocity at time n
	timeStepManager.runTimeStep(); // calculate the velocities for the current time step based on the current positions and potential
	cublasZaxpy(handle, N, &time_step, timeStepManager.devVelocitiesLower, 1, devZ, 1); // here devZ get's overwritten with the new positions
	// Note: This assumes devZ is already allocated and initialized with the current state of the system.
	// advance Phi using the rhs of phi
	cublasZaxpy(handle, N, &time_step, timeStepManager.devRhsPhi, 1, devPhi, 1); // update Phi with the RHS of the phi equation
}

#endif