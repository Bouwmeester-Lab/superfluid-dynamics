#pragma once

#ifndef HELIUM_DRIVING_PROBLEM_H


#include "TimedBoundaryProblem.cuh"
#include "createM.cuh"

//
template <int N>
class HeliumWithOptomechanicalDrivingProblem : public TimedBoundaryProblem<N, 1>
{
	VelocityCalculator<N, 1> velocityCalculator; ///< Velocity calculator for calculating the velocities based on the vorticities and matrices.
	DelayedIntensityTerm<N> delayedIntensityTerm; ///< Delayed intensity term for modeling the optomechanical driving in the helium boundary problem
	OptomechanicalVariables variables; ///< Optomechanical variables for configuring the optomechanical driving in the helium boundary problem

public:
	HeliumWithOptomechanicalDrivingProblem(ProblemProperties& properties, OptomechanicalVariables optomechanicalVariables) : TimedBoundaryProblem<N, 1>(new KineticEnergy<N>(properties), new VanDerWaalsEnergy<N>(properties), new SurfaceEnergy<N>(properties)),
		velocityCalculator(), delayedIntensityTerm(optomechanicalVariables), variables(optomechanicalVariables)
	{
		// Constructor for the helium boundary problem, initializing the velocity calculator with the problem properties
	}

	virtual void CreateMMatrix(double* M, const std_complex* Z, const std_complex* Zp, const std_complex* Zpp, ProblemProperties& properties) override
	{
		createFiniteDepthMKernel << <this->matrix_blocks, this->matrix_threads >> > (M, Z, Zp, Zpp, properties.depth, N, 1, properties.infinite_depth);
	}

	virtual void CalculateVelocities(const std_complex* Z,
		const std_complex* Zp,
		const std_complex* Zpp,
		std_complex* a,
		std_complex* aprime,
		std_complex* V1,
		std_complex* V2,
		std_complex* velocities,
		ProblemProperties& properties,
		bool lower) override
	{
		// create the V1 matrix and V2 diagonal vector
		createHeliumVelocityMatrices << <this->matrix_blocks, this->matrix_threads >> > (Z, Zp, Zpp, properties.depth, N, V1, V2, lower, 1, properties.infinite_depth);
		velocityCalculator.calculateVelocities(Z, Zp, Zpp, a, aprime, V1, V2, velocities, lower);
	}

	virtual void CalculateTimeDependentRhsPhi(const ProblemPointers problemPointers, std_complex* result, ProblemProperties& properties, double time, bool saveProgress = true) 
	{
		/// add the driving terms
		/// TODO: N here witll break when using batches.
		add_optical_field_drive_terms<N><<<this->blocks, this->threads>>> (result, time, problemPointers.Z, problemPointers.VelocitiesLower, delayedIntensityTerm.device_view(), variables, saveProgress);
	}

	virtual void CalculateRhsPhi(const ProblemPointers problemPointers, std_complex* result, ProblemProperties& properties) override
	{
		// do all the calculations as it was without the driving or loss terms.
		if (properties.use_expansions)
		{
			compute_rhs_helium_phi_expression_expansion_terms << <this->blocks, this->threads >> > (problemPointers.Z, problemPointers.VelocitiesLower, result, properties.depth, 1 * N, properties.expansion_order);
			return;
		}
		if (properties.kappa != 0.0)
			compute_rhs_helium_phi_expression_with_surface_tension << <this->blocks, this->threads >> > (problemPointers.Z, problemPointers.Zp, problemPointers.Zpp, problemPointers.VelocitiesLower, result, properties.depth, properties.kappa, 1 * N);
		else
			compute_rhs_helium_phi_expression << <this->blocks, this->threads >> > (problemPointers.Z, problemPointers.VelocitiesLower, result, properties.depth, 1 * N);
	}

	virtual void SetStartingTime(double time) override
	{
		// set the starting time for the delayed intensity term, this is used to calculate the delayed intensity term strength in the first iteration of the simulation
		delayedIntensityTerm.setInitialTime(time);
	}
};

#endif // !HELIUM_DRIVING_PROBLEM_H