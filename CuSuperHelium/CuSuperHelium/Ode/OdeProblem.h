#pragma once
#ifndef ODE_PROBLEM_H
#define ODE_PROBLEM_H

#include <cstddef>
#include "cuda_runtime.h"

/// <summary>
/// Tells a problem whether rhs() is evaluated at a state the solver has accepted (the start of a step)
/// or at a trial stage inside a step. Most problems ignore it. The optomechanical drive uses it to update
/// its delayed-intensity history only at accepted states. Phase 8 replaces it with an acceptStep hook.
/// </summary>
enum class Evaluation
{
	AcceptedPoint,
	Stage
};

/// <summary>
/// The one interface every time integrator (OdeSolver) uses: dy/dt = f(y, t).
/// </summary>
template <typename T>
class OdeProblem
{
public:
	virtual ~OdeProblem() = default;

	/// Number of elements of type T in the state vector.
	virtual size_t size() const = 0;

	/// Writes f(y, t) into dydt. Both are device pointers of size() elements and never overlap.
	/// Work is queued on the problem's stream; the call does not synchronize.
	virtual void rhs(const T* y, double t, T* dydt, Evaluation kind) = 0;

	/// Called once by the solver before the first step of runEvolution.
	virtual void start(double t0) {}

	virtual void setStream(cudaStream_t stream) = 0;
};

#endif // !ODE_PROBLEM_H
