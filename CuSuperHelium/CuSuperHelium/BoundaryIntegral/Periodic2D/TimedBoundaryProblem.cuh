#pragma once
#include "BoundaryIntegral/Periodic2D/InterfaceParameters.h"
#ifndef TIMED_BOUNDARY_PROBLEM_H

#include "BoundaryIntegral/Periodic2D/BoundaryProblem.cuh"

class TimedBoundaryProblem : public BoundaryProblem
{
	/*TimedBoundaryProblem(EnergyBase<N>* kinetic, EnergyBase<N>* potential, EnergyBase<N>* surface) : matrix_threads(16, 16, 1), matrix_blocks((N + 15) / 16, (N + 15) / 16, batchSize), energyContainer(kinetic, potential, surface)
	{
	}*/
public:
	using BoundaryProblem::BoundaryProblem; // Inherit the constructor from BoundaryProblem

	virtual ~TimedBoundaryProblem() {}

	virtual void CalculateTimeDependentRhsPhi(const ProblemPointers problemPointers, std_complex* result, InterfaceParameters& properties, double time, bool saveProgress = true) = 0;

	virtual void SetStartingTime(double time) = 0;
};

#endif // !TIMED_BOUNDARY_PROBLEM_H