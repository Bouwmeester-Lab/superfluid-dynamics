#pragma once

#ifndef HELIUM_DRIVING_PROBLEM_H


#include "BoundaryIntegral/Periodic2D/TimedBoundaryProblem.cuh"
#include "BoundaryIntegral/Periodic2D/MatrixKernels.cuh"
#include "BoundaryIntegral/Periodic2D/PhiRhsKernels.cuh"
#include "Optomechanics/OptomechanicalDriveKernels.cuh"

//
class HeliumWithOptomechanicalDrivingProblem : public TimedBoundaryProblem
{
	VelocityCalculator velocityCalculator; ///< Velocity calculator for calculating the velocities based on the vorticities and matrices.
	DelayedIntensityTerm delayedIntensityTerm; ///< Delayed intensity term for modeling the optomechanical driving in the helium boundary problem
	OptomechanicalVariables variables; ///< Optomechanical variables for configuring the optomechanical driving in the helium boundary problem

	std::unique_ptr<LightIntensity> intensity;
	GaussianDistribution lightShape;
	DimensionlessParameters parameters; ///< Full parameters (with scales) for the drive terms
public:
	HeliumWithOptomechanicalDrivingProblem(DimensionlessParameters& properties, OptomechanicalVariables optomechanicalVariables, std::unique_ptr<LightIntensity> lightIntensity, int n) : TimedBoundaryProblem(n, 1, new KineticEnergy(properties, n), new VanDerWaalsEnergy(properties, n), new SurfaceEnergy(properties, n)),
		velocityCalculator(n, 1), delayedIntensityTerm(optomechanicalVariables, n), variables(optomechanicalVariables), intensity(std::move(lightIntensity)), parameters(properties)
	{
		// Constructor for the helium boundary problem, initializing the velocity calculator with the problem properties
		intensity->allocate(N);
	}

	virtual void CreateMMatrix(double* M, const std_complex* Z, const std_complex* Zp, const std_complex* Zpp, InterfaceParameters& properties) override
	{
		createFiniteDepthMKernel << <this->matrix_blocks, this->matrix_threads >> > (M, Z, Zp, Zpp, properties.depth, N, 1, properties.infinite_depth);
		CUSH_CHECK_LAUNCH();
	}

	virtual void CalculateVelocities(const std_complex* Z,
		const std_complex* Zp,
		const std_complex* Zpp,
		std_complex* a,
		std_complex* aprime,
		std_complex* V1,
		std_complex* V2,
		std_complex* velocities,
		InterfaceParameters& properties,
		bool lower) override
	{
		// create the V1 matrix and V2 diagonal vector
		createHeliumVelocityMatrices << <this->matrix_blocks, this->matrix_threads >> > (Z, Zp, Zpp, properties.depth, N, V1, V2, lower, 1, properties.infinite_depth);
		CUSH_CHECK_LAUNCH();
		velocityCalculator.calculateVelocities(Z, Zp, Zpp, a, aprime, V1, V2, velocities, lower);
	}

	virtual void CalculateTimeDependentRhsPhi(const ProblemPointers problemPointers, std_complex* result, InterfaceParameters& properties, double time, bool saveProgress = true) 
	{
		/// add the driving terms
		/// TODO: N here witll break when using batches.
		// calculate the weights of the optical field, based on its shape:
		lightShape.x0 = variables.location_x0_mode;
		lightShape.sigma = variables.sigma_optical_mode;
		intensity->compute_weights<GaussianDistribution>(problemPointers.Z, variables, lightShape);
		// calculate the frequency shift due to the optomechanical shift
		double* frequencyShiftResult = intensity->compute_frequency_shift(problemPointers.Z, variables, this->parameters);
		

		add_optical_field_drive_terms<<<this->blocks, this->threads>>> (N, result, time, frequencyShiftResult, problemPointers.Z, problemPointers.VelocitiesLower, delayedIntensityTerm.device_view(), variables, this->parameters, saveProgress);
		CUSH_CHECK_LAUNCH();
	}

	virtual void CalculateRhsPhi(const ProblemPointers problemPointers, std_complex* result, InterfaceParameters& properties) override
	{
		// do all the calculations as it was without the driving or loss terms.
		if (properties.use_expansions)
		{
			compute_rhs_helium_phi_expression_expansion_terms << <this->blocks, this->threads >> > (problemPointers.Z, problemPointers.VelocitiesLower, result, properties.depth, 1 * N, properties.expansion_order);
			CUSH_CHECK_LAUNCH();
			return;
		}
		if (properties.kappa != 0.0)
			{ compute_rhs_helium_phi_expression_with_surface_tension << <this->blocks, this->threads >> > (problemPointers.Z, problemPointers.Zp, problemPointers.Zpp, problemPointers.VelocitiesLower, result, properties.depth, properties.kappa, 1 * N); CUSH_CHECK_LAUNCH(); }
		else
			{ compute_rhs_helium_phi_expression << <this->blocks, this->threads >> > (problemPointers.Z, problemPointers.VelocitiesLower, result, properties.depth, 1 * N); CUSH_CHECK_LAUNCH(); }
	}

	virtual void SetStartingTime(double time) override
	{
		// set the starting time for the delayed intensity term, this is used to calculate the delayed intensity term strength in the first iteration of the simulation
		delayedIntensityTerm.setInitialTime(time);
	}
};

#endif // !HELIUM_DRIVING_PROBLEM_H
