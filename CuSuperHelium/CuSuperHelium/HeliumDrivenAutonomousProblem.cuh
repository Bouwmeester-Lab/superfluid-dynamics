#pragma once
#ifndef HeliumDrivenAutonomusProblem_H
#define HeliumDrivenAutonomusProblem_H

#include "HeliumBoundaryProblem.cuh"
#include "OptomechanicalVariables.h"
#include "createM.cuh"
#include "LightIntensity.cuh"

template <int N, size_t batchSize>
class HeliumDrivenAutonomousProblem : public HeliumBoundaryProblem<N, batchSize>
{
protected:
	OptomechanicalVariables& variables; ///< Reference to the optomechanical variables containing the parameters for the driving terms in the simulation
	std::shared_ptr<LightIntensity> intensity;
	GaussianDistribution lightShape;
public:
	HeliumDrivenAutonomousProblem(ProblemProperties& properties, OptomechanicalVariables& variables, std::shared_ptr<LightIntensity> intensity) : HeliumBoundaryProblem<N, batchSize>(properties), variables(variables), intensity(intensity)
	{
		// Constructor for the helium driven autonomous problem, initializing the base class with the problem properties
		intensity->allocate(N * batchSize); // Allocate memory for the intensity weights based on the number of points in the simulation
	}

	virtual void CalculateRhsPhi(const ProblemPointers problemPointers, std_complex* result, ProblemProperties& properties) override
	{
		HeliumBoundaryProblem<N, batchSize>::CalculateRhsPhi(problemPointers, result, properties); // Call the base class implementation to calculate the right-hand side for the potential function on the boundary
		// add the driving terms that do not depend on time explecitly:
		// calculate the weights of the optical field, based on its shape:
		lightShape.x0 = variables.location_x0_mode;
		lightShape.sigma = variables.sigma_optical_mode;
		intensity->compute_weights<GaussianDistribution>(problemPointers.Z, variables, lightShape);
		// calculate the frequency shift due to the optomechanical shift
		double* frequencyShiftResult = intensity->compute_frequency_shift(problemPointers.Z, variables);

		add_optical_field_drive_terms_no_time_depence<N*batchSize><<<this->blocks, this->threads>>> (result, frequencyShiftResult, problemPointers.Z, problemPointers.VelocitiesLower, variables, properties);

	}
};


#endif // !HeliumDrivenAutonomusProblem_H