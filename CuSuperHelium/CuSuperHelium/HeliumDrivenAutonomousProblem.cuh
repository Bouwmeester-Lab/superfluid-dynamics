#pragma once
#ifndef HeliumDrivenAutonomusProblem_H
#define HeliumDrivenAutonomusProblem_H

#include "HeliumBoundaryProblem.cuh"
#include "OptomechanicalVariables.h"
#include "createM.cuh"

template <int N, size_t batchSize>
class HeliumDrivenAutonomousProblem : public HeliumBoundaryProblem<N, batchSize>
{
protected:
	OptomechanicalVariables& variables; ///< Reference to the optomechanical variables containing the parameters for the driving terms in the simulation
public:
	HeliumDrivenAutonomousProblem(ProblemProperties& properties, OptomechanicalVariables& variables) : HeliumBoundaryProblem<N, batchSize>(properties), variables(variables)
	{
		// Constructor for the helium driven autonomous problem, initializing the base class with the problem properties
	}

	virtual void CalculateRhsPhi(const ProblemPointers problemPointers, std_complex* result, ProblemProperties& properties) override
	{
		HeliumBoundaryProblem<N, batchSize>::CalculateRhsPhi(problemPointers, result, properties); // Call the base class implementation to calculate the right-hand side for the potential function on the boundary
		// add the driving terms that do not depend on time explecitly:
		add_optical_field_drive_terms_no_time_depence<N*batchSize><<<this->blocks, this->threads>>> (result, problemPointers.Z, problemPointers.VelocitiesLower, variables, properties);
		
	}
};


#endif // !HeliumDrivenAutonomusProblem_H