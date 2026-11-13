#pragma once
#include "BoundaryIntegral/Periodic2D/InterfaceParameters.h"
#ifndef TIMED_BOUNDARY_INTEGRATOR_H
#define TIMED_BOUNDARY_INTEGRATOR_H

#include "BoundaryIntegral/Periodic2D/BaseBoundaryIntegrator.cuh"
#include "BoundaryIntegral/Periodic2D/TimedBoundaryProblem.cuh"

/// <summary>
/// The periodic boundary integral with a time-dependent term added to the potential's right-hand side.
/// It is an OdeProblem like the autonomous calculator; rhs() receives the time and the evaluation kind.
/// </summary>
class TimedBoundaryIntegrator : public BaseBoundaryIntegralCalculator
{
	TimedBoundaryProblem& timedBoundaryProblem; ///< Reference to the timed boundary problem for calculating the time-dependent part of the rhs

	double currentTime = 0.0; ///< Time of the evaluation in progress
	/// True only when the solver evaluates at an accepted state. The drive term then updates its delayed-intensity
	/// history, so the history follows the time step and not the intermediate stages.
	bool saveProgress = true;
public:
	TimedBoundaryIntegrator(InterfaceParameters& problemProperties, TimedBoundaryProblem& boundaryProblem) : BaseBoundaryIntegralCalculator(problemProperties, boundaryProblem), timedBoundaryProblem(boundaryProblem)
	{
	}
	virtual ~TimedBoundaryIntegrator() {}

	void rhs(const std_complex* y, double t, std_complex* dydt, Evaluation kind) override
	{
		currentTime = t;
		saveProgress = (kind == Evaluation::AcceptedPoint);
		this->run(const_cast<std_complex*>(y), dydt);
	}

	void start(double t0) override
	{
		currentTime = t0;
		timedBoundaryProblem.SetStartingTime(t0);
	}

	virtual void calculateRhsPhi(const ProblemPointers& problemPointers, const std_complex* initialState, std_complex* rhs) override
	{
		BaseBoundaryIntegralCalculator::calculateRhsPhi(problemPointers, initialState, rhs); // Call the base class implementation to calculate the standard rhs
		// Now calculate the time-dependent part of the rhs using the boundary problem's method
		timedBoundaryProblem.CalculateTimeDependentRhsPhi(problemPointers, rhs, this->problemProperties, this->currentTime, this->saveProgress);
	}
};

#endif // !TIMED_BOUNDARY_INTEGRATOR_H
