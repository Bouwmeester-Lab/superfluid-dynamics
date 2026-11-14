#pragma once
#ifndef ODE_SOLVER_H
#define ODE_SOLVER_H

#include <memory>
#include <vector>
#include "cuda_runtime.h"
#include "IO/StepObserver.h"

enum class OdeSolverResult {
	ReachedEndTime,
	StiffnessDetected,
	UnknownError
};

/// <summary>
/// The interface shared by every time integrator. A solver integrates an OdeProblem&lt;T&gt;; what is recorded
/// during the integration is decided by the observers the caller attaches, not by the solver.
/// Options stay specific to each scheme (RK4Options, RK45_Options, GaussLegendre2Options).
/// </summary>
template <typename T>
class OdeSolver
{
public:
	virtual ~OdeSolver() = default;

	/// Sets the initial state. Every solver copies it into a buffer it owns, from device memory (onDevice = true)
	/// or host memory (onDevice = false); the caller's buffer is never modified. Read the result with state().
	virtual void initialize(const T* initialState, bool onDevice = false) = 0;

	virtual OdeSolverResult runEvolution(double startTime, double endTime) = 0;

	/// Device pointer to the current state.
	virtual const T* state() const = 0;

	/// Current integration time.
	virtual double time() const = 0;

	void addObserver(std::shared_ptr<StepObserver<T>> observer)
	{
		if (observer)
		{
			observers.push_back(std::move(observer));
		}
	}

protected:
	std::vector<std::shared_ptr<StepObserver<T>>> observers;

	void notifyStart(double t0, const T* y0, size_t n, cudaStream_t stream)
	{
		for (auto& observer : observers)
		{
			observer->onStart(t0, y0, n, stream);
		}
	}

	void notifyAcceptedStep(double t, const T* y, size_t n, cudaStream_t stream)
	{
		for (auto& observer : observers)
		{
			observer->onAcceptedStep(t, y, n, stream);
		}
	}
};

#endif // !ODE_SOLVER_H
