#pragma once
#ifndef TIMED_BOUNDARY_INTEGRATOR_H

#include "BaseBoundaryIntegrator.cuh"
#include "TimedBoundaryProblem.cuh"

template<int N, size_t batchSize>
class TimedBoundaryIntegrator : public BaseBoundaryIntegralCalculator<N, batchSize>
{
	TimedBoundaryProblem<N, batchSize>& timedBoundaryProblem; ///< Reference to the timed boundary problem for calculating the time-dependent part of the rhs
	
	double currentTime = 0.0; ///< Current time in the simulation, used for calculating the time-dependent rhs
	bool saveProgress = true; ///< Flag used to know if values can be saved by the inside methods. This is useful to coordinate with an outside RK4 integrator. Indeed, if this is false, the RK4 integrator can call the calculateRhs method multiple times without saving intermediate results, and only save the final result at the end of the RK4 step, allowing for results to only change with the RK4 time step and not with the intermediate stages of the RK4 method, which is more consistent with the idea of a time step in an ODE solver.
public:
	TimedBoundaryIntegrator(ProblemProperties& problemProperties, TimedBoundaryProblem<N, batchSize>& boundaryProblem) : BaseBoundaryIntegralCalculator<N, batchSize>(problemProperties, boundaryProblem), timedBoundaryProblem(boundaryProblem)
	{
	}
	virtual ~TimedBoundaryIntegrator() {}

	void setCurrentTime(double time)
	{
		currentTime = time;
	}

	void setSaveProgress(bool save)
	{
		saveProgress = save;
	}

	virtual void calculateRhs(const ProblemPointers& problemPointers, const std_complex* initialState, std_complex* rhs) override
	{
		BaseBoundaryIntegralCalculator<N, batchSize>::calculateRhs(problemPointers, initialState, rhs); // Call the base class implementation to calculate the standard rhs
		// Now calculate the time-dependent part of the rhs using the boundary problem's method
		timedBoundaryProblem.CalculateTimeDependentRhsPhi(problemPointers, rhs, this->problemProperties, currentTime, saveProgress);
	}

};

#endif // !TIMED_BOUNDARY_INTEGRATOR_H