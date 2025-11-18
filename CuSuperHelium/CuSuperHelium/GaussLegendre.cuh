#pragma once

#include "GLCoefficients.hpp"
#include "JacobianCalculator.cuh"
#include "OdeSolver.h"
#include <math.h>
#include <thrust/device_vector.h>
#include <cuda/std/array>
#include "VectorUtilities.cuh"
#include <indicators/progress_bar.hpp>

struct GaussLegendre2Options {
	double stepSize;
	double newtonTolerance;
	size_t maxNewtonIterations;
	bool allowSimplifiedFallback = true;
	bool returnTrajectory = true;
	double armijo_c = 1e-4;
	double backtrack = 0.5;
	double minAlpha = 1e-6;
	size_t maxStepsHalves = 6;

	GaussLegendre2Options() :
		stepSize(0.01),
		newtonTolerance(1e-10),
		maxNewtonIterations(20),
		allowSimplifiedFallback(false),
		returnTrajectory(true)
	{
	}
};

size_t to_percent(double t, double t0, double t1)
{
	if (t <= t0) return 0;
	if (t >= t1) return 100;
	double frac = (t - t0) / (t1 - t0);
	return static_cast<std::size_t>(std::round(frac * 100.0));
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
	double* devTempNextState = nullptr;

	thrust::device_vector<double> devTimes;
	thrust::device_vector<cuda::std::array<double, 3 * N>> devYs;

	double hmin = -1;

	cudaStream_t stream = cudaStreamPerThread;

	std::unique_ptr<indicators::ProgressBar> progressBar;

	struct StepResult {
		size_t numberIterations;
		bool converged;
		double residualNorm;
		bool simplifiedFallbackUsed;
	} stepRes;

	void gaussLegendreS2Step(double* devCurrentState, double* devDestination, const double stepSize, StepResult& result);
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
	double totalT = std::abs(endTime - startTime);

	if (hmin < 0) {
		hmin = totalT / (std::pow(2, 20));
	}


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

		appendToVector<double, 3 * N>(this->devYs, this->devTempState, this->stream);
	}

	double currentTime = startTime;
	double forward = (endTime >= startTime) ? 1.0 : -1.0;
	double heff;
	double htry;
	// create the progress bar
	progressBar = std::make_unique<indicators::ProgressBar>(
		indicators::option::BarWidth{ 50 },
		indicators::option::Start{"["},
		indicators::option::Fill{"="},
		indicators::option::Lead{">"},
		indicators::option::Remainder{" "},
		indicators::option::End{"]"},
		indicators::option::PrefixText{"Gauss-Legendre 2nd Order Integration Progress:"},
		indicators::option::ShowPercentage{true},
		indicators::option::ShowElapsedTime{true},
		indicators::option::ShowRemainingTime{true}
	);
	bool attempt_ok;

	

	while ((currentTime - endTime) * forward < 0.0)
	{
		heff = std::min(this->options.stepSize, std::abs(endTime - currentTime)) * forward;
		
		attempt_ok = false;
		htry = heff;
		for (int i = 0; i < this->options.maxStepsHalves + 1; ++i) 
		{
			this->gaussLegendreS2Step(this->devTempState, this->devTempNextState, htry, this->stepRes);
			if(this->stepRes.converged) 
			{
				attempt_ok = true;
				break;
			}
			if(htry <= hmin) 
			{
				break;
			}
			// Halve the step size and try again
			htry *= 0.5;
		}

		if(!attempt_ok) 
		{
			throw std::runtime_error(std::format("Gauss-Legendre 2nd Order method failed to converge at t≈{}; residual={:.3e}; last htry={:.3e}", currentTime, stepRes.residualNorm, htry);
		}

		// Step successful, update state and time
		cudaMemcpyAsync(this->devTempState, this->devTempNextState, 3 * N * sizeof(double), cudaMemcpyDeviceToDevice, this->stream);
		progressBar->set_progress(to_percent(currentTime, endTime, startTime));
		currentTime += htry;
		this->options.stepSize = std::abs(htry); // Update step size for next iteration

		if (this->options.returnTrajectory) 
		{
			this->devTimes.push_back(currentTime);
			appendToVector<double, 3 * N>(this->devYs, this->devTempState, this->stream);
		}
	}
	return OdeSolverResult::ReachedEndTime;
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
		checkCuda(cudaMallocAsync(&this->devTempState, stateSizeBytes, this->stream), "GaussLegendre2<N>::initialize", "GaussLegendre.cuh", 152));
		checkCuda(cudaMallocAsync(&this->devTempNextState, stateSizeBytes, this->stream), "GaussLegendre2<N>::initialize", "GaussLegendre.cuh", 153));

		err = cudaMemcpy(this->devState, initialState, stateSizeBytes, cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			throw std::runtime_error("Failed to copy initial state to device memory.");
		}
	}
	

}

template<size_t N>
void GaussLegendre2<N>::gaussLegendreS2Step(double* devCurrentState, double* devDestination, const double stepSize, StepResult& result)
{
	// k = np.vstack([f(y), f(y)])   # shape(2, m)
	// WIP...
	for (int iteration = 0; iteration < this->options.maxNewtonIterations; ++iteration)
	{

		//// Compute the function value and Jacobian at the current guess
		//// f(Y) = Y - y_n - h/2 * (f(t_n, y_n) + f(t_n + h, Y))
		//// where Y is the next state we are solving for
		//// Here we use devTempState as Y
		//problem.evaluateRHS(devCurrentState, this->devTempState, this->stream); // f(t_n, y_n)
		//problem.evaluateRHS(this->devTempState, this->devTempState, this->stream); // f(t_n + h, Y)
		//// Compute the residual
		//// res = Y - y_n - h/2 * (f(t_n, y_n) + f(t_n + h, Y))
	}
}
