#pragma once
#ifndef ODE_PROBLEM_H
#define ODE_PROBLEM_H

#include <cstddef>
#include "cuda_runtime.h"

/// <summary>
/// The one interface every time integrator (OdeSolver) uses: dy/dt = f(y, t).
///
/// rhs() is a pure function of (y, t): it must not keep state between calls. Solvers evaluate it at trial stages,
/// reject steps and iterate (Newton), so a problem that remembered anything from one call to the next would depend on
/// the scheme. Anything that evolves in time belongs in the state y (the delayed intensity, for example).
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
	virtual void rhs(const T* y, double t, T* dydt) = 0;

	/// Called once by the solver before the first step of runEvolution.
	virtual void start(double t0) {}

	virtual void setStream(cudaStream_t stream) = 0;
};

#endif // !ODE_PROBLEM_H
