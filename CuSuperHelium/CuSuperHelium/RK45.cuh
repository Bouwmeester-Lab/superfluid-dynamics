#pragma once

#include "ProblemProperties.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "AutonomousProblem.h"
#include "cuDoubleComplexOperators.cuh"
#include "cublas_v2.h"
#include "DataLogger.cuh"
#include <limits>
#include "utilities.cuh"
#include <algorithm> // std::clamp
#include <cmath>     // std::pow

/// <summary>
/// Defines the workspace memory needed for the RK45 method in the GPU.
/// </summary>
/// <typeparam name="T"></typeparam>
/// <typeparam name="N"></typeparam>
template <typename T, size_t N>
struct RK45WorkspaceGpu
{
	T* k1;
	T* k2;
	T* k3;
	T* k4;
	T* k5;
	T* k6;
	T* yAcceptedStep; // Y values for each accepted step. This is kept unchanged during the step and is only updated when the step is accepted.
	T* yTemp; // Temporary storage for intermediate y values

	T* rawMemory;

	__host__ RK45WorkspaceGpu()
	{
		CHECK_CUDA(cudaMalloc((void**)&rawMemory, sizeof(T) * (8 * N)));
		k1 = rawMemory;
		k2 = k1 + N;
		k3 = k2 + N;
		k4 = k3 + N;
		k5 = k4 + N;
		k6 = k5 + N;

		yAcceptedStep = k6 + N;
		yTemp = yAcceptedStep + N;
	}
	__host__ ~RK45WorkspaceGpu()
	{
		if (rawMemory) cudaFree(rawMemory);
	}

	// 2) Create a trivially-copyable device view
	struct View {
		T* __restrict__ k1, * __restrict__ k2, * __restrict__ k3,
			* __restrict__ k4, * __restrict__ k5, * __restrict__ k6,
			* __restrict__ y, * __restrict__ ytmp;
		size_t n;
	};

	__host__ View view() const noexcept {
		return View{ k1, k2, k3, k4, k5, k6, yAcceptedStep, yTemp, N };
	}

	// non-copyable and non-movable
	RK45WorkspaceGpu(const RK45WorkspaceGpu&) = delete;
	RK45WorkspaceGpu& operator=(const RK45WorkspaceGpu&) = delete;
	RK45WorkspaceGpu(RK45WorkspaceGpu&&) = delete;
	RK45WorkspaceGpu& operator=(RK45WorkspaceGpu&&) = delete;
};

enum class RK45_Result
{
	StepAccepted,
	StepRejected
};

template <typename T, size_t N>
class RK45
{
public:
	RK45(AutonomousProblem<T, N>& autonomousProblem, DataLogger<T, N>& logger, double tstep = 1e-2, double h_max = 1e10, double _h_min = -1.0);
	~RK45();

	[[nodiscard]] RK45_Result runStep(int i); // https://stackoverflow.com/questions/76489630/explanation-of-nodiscard-in-c17
	void setTolerance(double atol, double rtol) 
	{ 
		this->atol = atol;
		this->rtol = rtol;
	}
protected:
	RK45WorkspaceGpu<T, N> workspace;
	virtual inline void calculateTempY(uint16_t step, double timeStep) = 0;
	// Calculate the new y value after a successful step and saves it in workspace.yAcceptedStep
	virtual inline void calculateWeightedY(double timeStep) = 0;
	virtual inline double calculateScaledError(double atol, double rtol) = 0;
	virtual inline double calculateNewTimestep(double oldTimestep, double error, bool accepting);
	

	void initialize(T* initialState, bool onDevice = false);


	AutonomousProblem<T, N>& problem;
	DataLogger<T, N>& logger;

	double atol = 1e-6;
	double rtol = 1e-3;

	
	double currentTimeStep = 1e-3;


	struct StepTempValues {
		double timestepNew = 1e-3;
		double scaledError = 0.0;
		bool acceptedStep = true;
	} tempValues;

	// bounds for timestep
	const double h_min = 1e-16;
	const double h_max;
};
template <typename T, size_t N>
RK45<T, N>::RK45(AutonomousProblem<T, N>& autonomousProblem, DataLogger<T, N>& logger, double tstep, double _h_max, double _h_min) : problem(autonomousProblem), logger(logger), currentTimeStep(tstep), h_max(_h_max)
{
	if (_h_min > 0.0) // only set h_min if a positive value is given, otherwise use the default value
	{
		this->h_min = _h_min;
	}
}
template <typename T, size_t N>
RK45<T, N>::~RK45()
{
}

template <typename T, size_t N>
RK45_Result RK45<T, N>::runStep(int i)
{
	const double h = currentTimeStep;
	// calculate k1
	if (tempValues.acceptedStep) { // only recalculate k1 if the previous step was accepted, otherwise reuse the previous k1
		problem.run(workspace.yAcceptedStep, workspace.k1);
	}

	calculateTempY(1, h); // calculate y + a21*k1
	problem.run(workspace.yTemp, workspace.k2);

	calculateTempY(2, h); // calculate y + a31*k1 + a32*k2
	problem.run(workspace.yTemp, workspace.k3);

	calculateTempY(3, h); // calculate y + a41*k1 + a42*k2 + a43*k3
	problem.run(workspace.yTemp, workspace.k4);

	calculateTempY(4, h); // calculate y + a51*k1 + a52*k2 + a53*k3 + a54*k4
	problem.run(workspace.yTemp, workspace.k5);

	calculateTempY(5, h); // calculate y + a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5
	problem.run(workspace.yTemp, workspace.k6);

	
	tempValues.scaledError = calculateScaledError(atol, rtol);
	tempValues.acceptedStep = tempValues.scaledError <= 1.0;
	

	tempValues.timestepNew = calculateNewTimestep(h, tempValues.scaledError, tempValues.acceptedStep);

	if (tempValues.acceptedStep) {
		// accept step
		calculateWeightedY(h);
		currentTimeStep = tempValues.timestepNew;
		if (logger.shouldCopy(i))
			logger.setReadyToCopy(workspace.yAcceptedStep);
		return RK45_Result::StepAccepted;
	}
	else {
		// reject step
		currentTimeStep = tempValues.timestepNew;
		return RK45_Result::StepRejected;
	}
}

template<typename T, size_t N>
inline double RK45<T, N>::calculateNewTimestep(double oldTimestep, double error, bool accepting)
{
	constexpr double safety = 0.9;
	constexpr double minfac = 0.2;   // at most 5x shrink
	constexpr double maxfac = 5.0;   // at most 5x growth
	constexpr double expo = 1.0 / 5.0; // RK45 -> 1/(p+1), p=4

	if (error == 0.0) {
		// If we’re accepting and error is zero, push to max growth.
		double fac = accepting ? maxfac : 1.0;   // never grow on reject
		return oldTimestep * fac;
	}
	double fac = safety * std::pow(error, -expo);

	if (accepting) {
		fac = std::clamp(fac, minfac, maxfac);  // allow growth
	}
	else {
		fac = std::clamp(fac, minfac, 1.0);     // forbid growth on retry
	}
	return std::clamp(oldTimestep * fac, h_min, h_max);
}

template<typename T, size_t N>
void RK45<T, N>::initialize(T* initialState, bool onDevice)
{
	if (onDevice) {
		CHECK_CUDA(cudaMemcpy(workspace.yAcceptedStep, initialState, sizeof(T) * N, cudaMemcpyDeviceToDevice));
	}
	else {
		CHECK_CUDA(cudaMemcpy(workspace.yAcceptedStep, initialState, sizeof(T) * N, cudaMemcpyHostToDevice));
	}
}