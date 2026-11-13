#pragma once
#ifndef RUNGE_KUTTA_4_H
#define RUNGE_KUTTA_4_H

#include <stdexcept>
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "Core/CudaChecks.cuh"
#include "Core/DeviceBuffer.cuh"
#include "Core/VectorKernels.cuh"
#include "Core/cuDoubleComplexOperators.cuh"
#include "LinearAlgebra/Blas.cuh"
#include "Ode/OdeProblem.h"
#include "Ode/OdeSolver.h"
#include "Ode/RK4Options.h"

/// <summary>
/// Classical fixed-step fourth-order Runge-Kutta for any OdeProblem&lt;T&gt;, T = std_complex, cuDoubleComplex or double.
/// The state size is read from the problem at construction.
///
/// Stage evaluations: k1 = f(y, t) [AcceptedPoint], k2 = f(y + h/2 k1, t + h/2), k3 = f(y + h/2 k2, t + h/2),
/// k4 = f(y + h k3, t + h) [all Stage]; then y += h/6 (k1 + 2 k2 + 2 k3 + k4).
/// </summary>
template <typename T>
class RungeKutta4 final : public OdeSolver<T>
{
public:
	RungeKutta4(OdeProblem<T>& problem, double tstep = 1e-2);
	~RungeKutta4() override;

	RungeKutta4(const RungeKutta4&) = delete;
	RungeKutta4& operator=(const RungeKutta4&) = delete;

	void setTimeStep(double tstep)
	{
		timeStep = CastTo<T>(tstep);
		halfTimeStep = CastTo<T>(tstep * 0.5);
		sixthTimeStep = CastTo<T>(tstep / 6.0);
	}

	void setOptions(const RK4Options& options)
	{
		setTimeStep(options.initial_timestep);
	}

	/// With onDevice = true the solver evolves the given device buffer in place.
	void initialize(T* devY0, bool onDevice = false) override;
	OdeSolverResult runEvolution(double startTime, double endTime) override;
	void runStep();

	const T* state() const override { return devY0; }
	double time() const override { return currentTime; }

private:
	OdeProblem<T>& problem;
	const size_t n;
	const int threads = 256;
	const int blocks;
	cudaStream_t stream = cudaStreamPerThread; ///< Stream handed to observers (the stepper's own work uses the default stream)

	DeviceBuffer<T> k1, k2, k3, k4;
	DeviceBuffer<T> devY1, devY2, devY3;
	DeviceBuffer<T> ownedY0; ///< Holds the state when initialize() copies it from the host
	T* devY0 = nullptr;

	cublasHandle_t handle = nullptr;

	T timeStep;
	T halfTimeStep;
	T sixthTimeStep;

	double currentTime = 0.0;

	void resetStages();
};

template <typename T>
RungeKutta4<T>::RungeKutta4(OdeProblem<T>& problem, double tstep) : problem(problem), n(problem.size()), blocks(static_cast<int>((problem.size() + 255) / 256)),
	k1(n), k2(n), k3(n), k4(n), devY1(n), devY2(n), devY3(n)
{
	if (n == 0)
	{
		throw std::invalid_argument("RungeKutta4 requires a problem with a non-empty state");
	}
	CUSH_CHECK(cublasCreate(&handle));
	setTimeStep(tstep);
}

template <typename T>
RungeKutta4<T>::~RungeKutta4()
{
	if (handle) CUSH_CHECK_NOTHROW(cublasDestroy(handle));
}

template <typename T>
void RungeKutta4<T>::initialize(T* y0, bool onDevice)
{
	if (onDevice)
	{
		devY0 = y0;
	}
	else
	{
		if (ownedY0.size() != n)
		{
			ownedY0 = DeviceBuffer<T>(n);
		}
		devY0 = ownedY0.get();
		CUSH_CHECK(cudaMemcpy(devY0, y0, n * sizeof(T), cudaMemcpyHostToDevice));
	}
	resetStages();
}

template <typename T>
void RungeKutta4<T>::resetStages()
{
	CUSH_CHECK(cudaMemcpy(devY1.get(), devY0, n * sizeof(T), cudaMemcpyDeviceToDevice));
	CUSH_CHECK(cudaMemcpy(devY2.get(), devY0, n * sizeof(T), cudaMemcpyDeviceToDevice));
	CUSH_CHECK(cudaMemcpy(devY3.get(), devY0, n * sizeof(T), cudaMemcpyDeviceToDevice));
}

template <typename T>
void RungeKutta4<T>::runStep()
{
	const double t = currentTime;
	const int count = static_cast<int>(n);

	problem.rhs(devY0, t, k1.get(), Evaluation::AcceptedPoint);
	CUSH_CHECK(cush::blas::axpy(handle, count, &halfTimeStep, k1.get(), devY1.get()));

	problem.rhs(devY1.get(), t + CastFrom<T>(halfTimeStep), k2.get(), Evaluation::Stage);
	CUSH_CHECK(cush::blas::axpy(handle, count, &halfTimeStep, k2.get(), devY2.get()));

	problem.rhs(devY2.get(), t + CastFrom<T>(halfTimeStep), k3.get(), Evaluation::Stage);
	CUSH_CHECK(cush::blas::axpy(handle, count, &timeStep, k3.get(), devY3.get()));

	problem.rhs(devY3.get(), t + CastFrom<T>(timeStep), k4.get(), Evaluation::Stage);

	add_k_vectors << <blocks, threads >> > (k1.get(), k2.get(), k3.get(), k4.get(), k1.get(), count);
	CUSH_CHECK_LAUNCH();

	// k1 now holds k1 + 2 k2 + 2 k3 + k4
	CUSH_CHECK(cush::blas::axpy(handle, count, &sixthTimeStep, k1.get(), devY0));

	resetStages();
}

template <typename T>
OdeSolverResult RungeKutta4<T>::runEvolution(double startTime, double endTime)
{
	if (devY0 == nullptr)
	{
		throw std::logic_error("RungeKutta4::runEvolution: call initialize() first");
	}
	currentTime = startTime;
	const size_t steps = static_cast<size_t>((endTime - startTime) / CastFrom<T>(timeStep));
	problem.start(startTime);
	this->notifyStart(startTime, devY0, n, stream);
	for (size_t step = 0; step < steps; step++)
	{
		runStep();
		currentTime = startTime + (step + 1) * CastFrom<T>(timeStep);
		this->notifyAcceptedStep(currentTime, devY0, n, stream);
	}
	if (currentTime >= endTime)
		return OdeSolverResult::ReachedEndTime;
	return OdeSolverResult::UnknownError;
}

#endif // !RUNGE_KUTTA_4_H
