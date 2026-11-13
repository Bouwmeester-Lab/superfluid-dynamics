#pragma once
#ifndef STEP_OBSERVER_H
#define STEP_OBSERVER_H

#include <cstddef>
#include "cuda_runtime.h"

/// <summary>
/// Receives the state of a time integration. Every OdeSolver calls its observers the same way, so the
/// caller chooses what is recorded (a trajectory, a progress bar, energies) independently of the scheme.
/// All pointers are device pointers to a state of n elements; they are only valid during the call, and
/// work queued on them must use the given stream.
/// </summary>
template <typename T>
class StepObserver
{
public:
	virtual ~StepObserver() = default;

	/// Called once by runEvolution, before the first step, with the initial state.
	virtual void onStart(double t0, const T* y0, size_t n, cudaStream_t stream) {}

	/// Called once after every accepted step, with the new time and state.
	virtual void onAcceptedStep(double t, const T* y, size_t n, cudaStream_t stream) = 0;
};

#endif // !STEP_OBSERVER_H
