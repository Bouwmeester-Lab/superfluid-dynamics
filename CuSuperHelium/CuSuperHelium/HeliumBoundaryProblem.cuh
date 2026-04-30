#pragma once

#include "BoundaryProblem.cuh"


template<int N, size_t batchSize>
class HeliumBoundaryProblem : public BoundaryProblem<N, batchSize>
{
	VelocityCalculator<N, batchSize> velocityCalculator; ///< Velocity calculator for calculating the velocities based on the vorticities and matrices.
public:
	HeliumBoundaryProblem(ProblemProperties& properties) : BoundaryProblem<N, batchSize>(new KineticEnergy<N>(properties), new VanDerWaalsEnergy<N>(properties), new SurfaceEnergy<N>(properties)), velocityCalculator()
	{
		// Constructor for the helium boundary problem, initializing the velocity calculator with the problem properties
	}
	virtual void CreateMMatrix(double* M, const std_complex* Z, const std_complex* Zp, const std_complex* Zpp, ProblemProperties& properties) override
	{
		createFiniteDepthMKernel << <this->matrix_blocks, this->matrix_threads >> > (M, Z, Zp, Zpp, properties.depth, N, batchSize, properties.infinite_depth);
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
		createHeliumVelocityMatrices << <this->matrix_blocks, this->matrix_threads >> > (Z, Zp, Zpp, properties.depth, N, V1, V2, lower, batchSize, properties.infinite_depth);
		velocityCalculator.calculateVelocities(Z, Zp, Zpp, a, aprime, V1, V2, velocities, lower);
	}
	virtual void CalculateRhsPhi(const ProblemPointers problemPointers, std_complex* result, ProblemProperties& properties) override
	{
		if (properties.use_expansions)
		{
			compute_rhs_helium_phi_expression_expansion_terms << <this->blocks, this->threads >> > (problemPointers.Z, problemPointers.VelocitiesLower, result, properties.depth, batchSize * N, properties.expansion_order);
			return;
		}
		if (properties.kappa != 0.0)
			compute_rhs_helium_phi_expression_with_surface_tension << <this->blocks, this->threads >> > (problemPointers.Z, problemPointers.Zp, problemPointers.Zpp, problemPointers.VelocitiesLower, result, properties.depth, properties.kappa, batchSize * N);
		else
			compute_rhs_helium_phi_expression << <this->blocks, this->threads >> > (problemPointers.Z, problemPointers.VelocitiesLower, result, properties.depth, batchSize * N);
	}
};


template <int N, size_t batchSize>
class HeliumInfiniteDepthBoundaryProblem : public BoundaryProblem<N, batchSize>
{
	VelocityCalculator<N, batchSize> velocityCalculator; ///< Velocity calculator for calculating the velocities based on the vorticities and matrices.
public:
	HeliumInfiniteDepthBoundaryProblem(ProblemProperties& properties) : BoundaryProblem<N, batchSize>(new KineticEnergy<N>(properties), new VanDerWaalsEnergy<N>(properties), new SurfaceEnergy<N>(properties)), velocityCalculator()
	{
		// Constructor for the helium infinite depth boundary problem, initializing the velocity calculator with the problem properties
	}
	virtual void CreateMMatrix(double* M, const std_complex* Z, const std_complex* Zp, const std_complex* Zpp, ProblemProperties& properties) override
	{
		createMKernel << <this->matrix_blocks, this->matrix_threads >> > (M, Z, Zp, Zpp, properties.depth, N, batchSize);
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
		createVelocityMatrices << <this->matrix_blocks, this->matrix_threads >> > (Z, Zp, Zpp, N, V1, V2, lower);
		velocityCalculator.calculateVelocities(Z, Zp, Zpp, a, aprime, V1, V2, velocities, lower);
	}
	virtual void CalculateRhsPhi(const ProblemPointers problemPointers, std_complex* result, ProblemProperties& properties) override
	{
		compute_rhs_helium_phi_expression << <this->blocks, this->threads >> > (problemPointers.Z, problemPointers.VelocitiesLower, result, properties.depth, batchSize * N);
	}
};