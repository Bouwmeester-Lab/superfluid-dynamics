#pragma once
#ifndef RK4_TIME_DEPENDENT_H

#include "TimedBoundaryIntegrator.cuh"
#include "BondaryIntegrator.cuh"
#include "OdeSolver.h"

struct RK4_Options {
	double timeStep = 1e-3;
	// Add more options as needed, such as error tolerances for adaptive time stepping
};;

template <int N, size_t batchSize>
class RK4TimeDependent : public OdeSolver 
{
private:
	TimedBoundaryIntegrator<N, batchSize>& integrator;
	RK4_Options options;

	// initial state
	std_complex* devInitialState; ///< Device pointer to the initial state array (Z array + phi)


	// arrays needed for calculating the stages of RK4
	std_complex* devK1; ///< Device pointer to the k1 array (RHS for stage 1)
	std_complex* devK2; ///< Device pointer to the k2 array (RHS for stage 2)
	std_complex* devK3; ///< Device pointer to the k3 array (RHS for stage 3)
	std_complex* devK4; ///< Device pointer to the k4 array (RHS for stage 4)
public:
	RK4TimeDependent(TimedBoundaryIntegrator<N, batchSize>& integrator) : integrator(integrator) 
	{
		cudaMalloc(&devK1, sizeof(std_complex) * 2 * N * batchSize); // Assuming the state consists of Z and phi, each of size N*batchSize
		cudaMalloc(&devK2, sizeof(std_complex) * 2 * N * batchSize);
		cudaMalloc(&devK3, sizeof(std_complex) * 2 * N * batchSize);
		cudaMalloc(&devK4, sizeof(std_complex) * 2 * N * batchSize);
	}
	~RK4TimeDependent() 
	{
		cudaFree(devK1);
		cudaFree(devK2);
		cudaFree(devK3);
		cudaFree(devK4);
	}

	virtual OdeSolverResult runEvolution(double startTime, double endTime) override;
	void setOptions(const RK4_Options& newOptions) {
		options = newOptions;
	}

	void initialize(const std_complex* initialState, bool onDevice) 
	{
		if(onDevice) {
			devInitialState = const_cast<std_complex*>(initialState); // If the initial state is already on the device, just set the pointer
			return;
		}
		// Allocate memory for the initial state on the device and copy the initial state from the host to the device
		cudaMalloc(&devInitialState, sizeof(std_complex) * 2 * N * batchSize); // Assuming the state consists of Z and phi, each of size N*batchSize
		cudaMemcpy(devInitialState, initialState, sizeof(std_complex) * 2 * N * batchSize, cudaMemcpyHostToDevice);
	}
};


template<int N, size_t batchSize>
OdeSolverResult RK4TimeDependent<N, batchSize>::runEvolution(double startTime, double endTime)
{
	// Implement the RK4 method here, using the integrator to calculate the rhs at each stage
	// set the current time in the integrator to the start time
	integrator.setCurrentTime(startTime);

	// RK4 is a method that uses multiple stages to calculate the next value
	// For each stage, we need to calculate the right-hand side (RHS) using the integrator
	// The basic idea is to take the current state, apply the integrator to get the RHS, and then use that to update the state
	// RK4 works as follows for a 1st order ODE: dy/dt = f(t, y) with timestep h:
	// k1 = f(t_n, y_n)
	// k2 = f(t_n + h/2, y_n + h/2 * k1)
	// k3 = f(t_n + h/2, y_n + h/2 * k2)
	// k4 = f(t_n + h, y_n + h * k3)
	// y_{n+1} = y_n + h/6 * (k1 + 2*k2 + 2*k3 + k4)

	// calculate the number of steps needed to reach the end time
	size_t numSteps = static_cast<int>(std::ceil(std::abs(endTime - startTime) / options.timeStep));
	size_t currentStep = 0;

	for(size_t step = 0; step < numSteps; step++) {
		// calculate k1, k2, k3, k4 using the integrator to get the RHS at each stage
		// for k1, we allow it to save progress since it's f(t_n, y_n), the RHS for the current state, which we want to save for the next stages, we want to
		// at this stage, to remember how the RHS looks like for tn, yn.
		integrator.setSaveProgress(true);
		// time was set already either before the loop or at the end of the previous iteration, so we can just call run to calculate k1
		integrator.run(devInitialState, devK1);

		// for k2, k3, k4, we don't want to save progress since they are just intermediate stages and we only want to save the final result at the end of the RK4 step, which is more consistent with the idea of a time step in an ODE solver. Indeed, if we allow saving progress for k2, k3, k4, then the results would change with the intermediate stages of the RK4 method, which is not what we want.
		integrator.setSaveProgress(false);
		integrator.setCurrentTime(startTime + options.timeStep / 2.0); // Set time for k2
		// For k2, we need to update the state to yn + h/2 * k1 before calling the integrator to get the RHS for k2


		// Here we would calculate k1, k2, k3, k4 using the integrator to get the RHS at each stage
		// This would involve calling the integrator's calculateRhs method with the appropriate state and time for each stage
		// After calculating k1, k2, k3, k4, we would update the state to get the new state at the next time step
		// We would also need to check for any conditions that might indicate stiffness or other issues that would require us to stop the evolution
	}

	return OdeSolverResult();
}

#endif