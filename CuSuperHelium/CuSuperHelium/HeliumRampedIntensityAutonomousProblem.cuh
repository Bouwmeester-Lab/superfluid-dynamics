#pragma once
#ifndef HeliumRampedIntensityAutonomousProblem_H
#define HeliumRampedIntensityAutonomousProblem_H

#include "HeliumBoundaryProblem.cuh"
#include "OptomechanicalVariables.h"
#include "LightIntensity.cuh"
#include "Optomechanics/OptomechanicalDriveKernels.cuh"

template <int N, size_t batchSize>
class HeliumRampedIntensityAutonomousProblem : public HeliumBoundaryProblem<N, batchSize>
{
protected:
	OptomechanicalVariables& variables; ///< Reference to the optomechanical variables containing the parameters for the driving terms in the simulation
	std::shared_ptr<LightIntensity> intensity;
	GaussianDistribution lightShape;
public:
	HeliumRampedIntensityAutonomousProblem(ProblemProperties& properties, OptomechanicalVariables& variables, std::shared_ptr<LightIntensity> intensity) : HeliumBoundaryProblem<N, batchSize>(properties), variables(variables), intensity(intensity)
	{
		// Constructor for the helium ramped intensity autonomous problem, initializing the base class with the problem properties
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
		const std_complex* devRampedIntensity = problemPointers.Z + 3 * N * batchSize; // the ramped intensity is the fourth variable of the state vector: Z ( 0 - N), Phi (N, 2*N), DelayedIntensity (2*N, 3*N), RampedIntensity (3*N single variable)

		// calculate the frequency shift due to the optomechanical shift
		double* frequencyShiftResult = intensity->compute_frequency_shift(problemPointers.Z, variables, properties);

		add_optical_field_drive_terms_no_time_depence<N* batchSize> << <this->blocks, this->threads >> > (result, devRampedIntensity, frequencyShiftResult, problemPointers.Z, problemPointers.VelocitiesLower, variables, properties);

	}
};


#endif // !HeliumRampedIntensityAutonomousProblem_H
