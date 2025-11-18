#pragma once

#include "GLCoefficients.hpp"
#include "JacobianCalculator.cuh"
#include "OdeSolver.h"
#include <math.h>

struct GaussLegendre2Options {
	double stepSize;
	double newtonTolerance;
	size_t maxNewtonIterations;
	bool simplified;
	bool returnTrajectory = true;

	GaussLegendre2Options() :
		stepSize(0.01),
		newtonTolerance(1e-10),
		maxNewtonIterations(20),
		simplified(false),
		returnTrajectory(true)
	{
	}
};


template <size_t N>
class GaussLegendre2 : public OdeSolver
{
public:
	GaussLegendre2(AutonomousProblem<std_complex, N>& problem, JacobianCalculator<N>& jacobianCalculator, GaussLegendre2Options options = GaussLegendre2Options());
	~GaussLegendre2();
	virtual OdeSolverResult runEvolution(double startTime, double endTime) override;
	/// <summary>
	/// Sets the initial state for the solver.
	/// </summary>
	/// <param name="initialState"></param>
	/// <param name="onDevice"></param>
	void initialize(double* initialState, bool onDevice = false);
	void setStream(cudaStream_t stream) { this->stream = stream; }
private:
	AutonomousProblem<std_complex, N>& problem;
	JacobianCalculator<N>& jacobianCalculator;
	GaussLegendre2Options options;
	void setOptions(const GaussLegendre2Options options) { this->options = options; }

	double* devState = nullptr;
	double* devTempState = nullptr;

	double* devTimes = nullptr;
	double* devYs = nullptr;

	cudaStream_t stream = cudaStreamPerThread;


};

template <size_t N>
GaussLegendre2<N>::GaussLegendre2(AutonomousProblem<std_complex, N>& problem, JacobianCalculator<N>& jacobianCalculator, GaussLegendre2Options options) : problem(problem), jacobianCalculator(jacobianCalculator), options(options)
{
}
template <size_t N>
GaussLegendre2<N>::~GaussLegendre2()
{
}

template<size_t N>
OdeSolverResult GaussLegendre2<N>::runEvolution(double startTime, double endTime)
{
	if (this->options.stepSize < 0.0) {
		throw new std::invalid_argument("Step size must be positive.");
	}

	int nSteps = static_cast<int>(std::ceil(std::abs(endTime - startTime) / this->options.stepSize));
	const double hEff = (endTime - startTime) / nSteps;

	if(this->devState == nullptr) {
		throw std::runtime_error("Initial state not set. Call initialize() before running evolution.");
	}
	// copy initial state to temp state
	cudaMemcpyAsync(this->devTempState, this->devState, 3 * N * sizeof(double), cudaMemcpyDeviceToDevice, this->stream);

	// create the trajectory storage
	if (options.returnTrajectory)
	{
		checkCuda(cudaMallocAsync(&this->devTimes, (nSteps + 1) * sizeof(double), this->stream), "GaussLegendre2<N>::runEvolution", "GaussLegendre.cuh", 85));
		linspace<<<(nSteps + 1 + 255) / 256, 256, 0, this->stream>>>(this->devTimes, startTime, endTime, nSteps + 1);

		checkCuda(cudaMallocAsync(&this->devYs, (nSteps + 1) * 3 * N * sizeof(double), this->stream), "GaussLegendre2<N>::runEvolution", "GaussLegendre.cuh", 88)));
		// copy initial state to first entry
		checkCuda(cudaMemcpyAsync(this->devYs, this->devTempState, 3 * N * sizeof(double), cudaMemcpyDeviceToDevice, this->stream), "GaussLegendre2<N>::runEvolution", "GaussLegendre.cuh", 90));
	}

	double currentTime = startTime;

	for(int step = 0; step < nSteps; ++step) {
		// Perform a single Gauss-Legendre step from currentTime to currentTime + hEff
		// Using Newton's method to solve the implicit equations
		// Pseudocode for Newton's method:
		// 1. Initialize guess for stage values
		// 2. Iterate until convergence or max iterations reached
		//    a. Evaluate the residuals
		//    b. Evaluate the Jacobian
		//    c. Update the guess
		// After convergence, update devTempState with the new state
		currentTime += hEff;
		if (options.returnTrajectory) {
			// Copy the new state into the trajectory storage
			checkCuda(cudaMemcpyAsync(this->devYs + (step + 1) * 3 * N, this->devTempState, 3 * N * sizeof(double), cudaMemcpyDeviceToDevice, this->stream), "GaussLegendre2<N>::runEvolution", "GaussLegendre.cuh", 123));
		}
	}
}

template<size_t N>
void GaussLegendre2<N>::initialize(double* initialState, bool onDevice)
{
	if (onDevice) {
		this->devState = initialState;
	}
	else {
		size_t stateSizeBytes = 3 * N * sizeof(double); // x, y, phi each of size N.
		cudaError_t err = cudaMalloc(&this->devState, stateSizeBytes);
		if (err != cudaSuccess) {
			throw std::runtime_error("Failed to allocate device memory for initial state.");
		}
		err = cudaMemcpy(this->devState, initialState, stateSizeBytes, cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			throw std::runtime_error("Failed to copy initial state to device memory.");
		}
	}
	

}
