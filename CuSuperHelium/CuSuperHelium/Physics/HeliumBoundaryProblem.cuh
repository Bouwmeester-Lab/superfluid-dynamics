#pragma once

#include "BoundaryIntegral/Periodic2D/BoundaryProblem.cuh"
#include "BoundaryIntegral/Periodic2D/MatrixKernels.cuh"
#include "BoundaryIntegral/Periodic2D/PhiRhsKernels.cuh"


class HeliumBoundaryProblem : public BoundaryProblem
{
	VelocityCalculator velocityCalculator; ///< Velocity calculator for calculating the velocities based on the vorticities and matrices.
protected:
	DimensionlessParameters parameters; ///< Full parameters (with scales) for derived problems that need more than InterfaceParameters
public:
	HeliumBoundaryProblem(DimensionlessParameters& properties, int n, size_t batchSize = 1) : BoundaryProblem(n, batchSize, new KineticEnergy(properties, n), new VanDerWaalsEnergy(properties, n), new SurfaceEnergy(properties, n)), velocityCalculator(n, batchSize), parameters(properties)
	{
		// Constructor for the helium boundary problem, initializing the velocity calculator with the problem properties
	}
	virtual void CreateMMatrix(double* M, const std_complex* Z, const std_complex* Zp, const std_complex* Zpp, InterfaceParameters& properties) override
	{
		createFiniteDepthMKernel << <this->matrix_blocks, this->matrix_threads >> > (M, Z, Zp, Zpp, properties.depth, N, batchSize, properties.infinite_depth);
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
		createHeliumVelocityMatrices << <this->matrix_blocks, this->matrix_threads >> > (Z, Zp, Zpp, properties.depth, N, V1, V2, lower, batchSize, properties.infinite_depth);
		CUSH_CHECK_LAUNCH();
		velocityCalculator.calculateVelocities(Z, Zp, Zpp, a, aprime, V1, V2, velocities, lower);
	}
	virtual void CalculateRhsPhi(const ProblemPointers problemPointers, std_complex* result, InterfaceParameters& properties) override
	{
		if (properties.use_expansions)
		{
			compute_rhs_helium_phi_expression_expansion_terms << <this->blocks, this->threads >> > (problemPointers.Z, problemPointers.VelocitiesLower, result, properties.depth, batchSize * N, properties.expansion_order);
			CUSH_CHECK_LAUNCH();
			return;
		}
		if (properties.kappa != 0.0)
			{ compute_rhs_helium_phi_expression_with_surface_tension << <this->blocks, this->threads >> > (problemPointers.Z, problemPointers.Zp, problemPointers.Zpp, problemPointers.VelocitiesLower, result, properties.depth, properties.kappa, batchSize * N); CUSH_CHECK_LAUNCH(); }
		else
			{ compute_rhs_helium_phi_expression << <this->blocks, this->threads >> > (problemPointers.Z, problemPointers.VelocitiesLower, result, properties.depth, batchSize * N); CUSH_CHECK_LAUNCH(); }
	}
};


class HeliumInfiniteDepthBoundaryProblem : public BoundaryProblem
{
	VelocityCalculator velocityCalculator; ///< Velocity calculator for calculating the velocities based on the vorticities and matrices.
public:
	HeliumInfiniteDepthBoundaryProblem(DimensionlessParameters& properties, int n, size_t batchSize = 1) : BoundaryProblem(n, batchSize, new KineticEnergy(properties, n), new VanDerWaalsEnergy(properties, n), new SurfaceEnergy(properties, n)), velocityCalculator(n, batchSize)
	{
		// Constructor for the helium infinite depth boundary problem, initializing the velocity calculator with the problem properties
	}
	virtual void CreateMMatrix(double* M, const std_complex* Z, const std_complex* Zp, const std_complex* Zpp, InterfaceParameters& properties) override
	{
		createMKernel << <this->matrix_blocks, this->matrix_threads >> > (M, Z, Zp, Zpp, properties.depth, N, batchSize);
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
		createVelocityMatrices << <this->matrix_blocks, this->matrix_threads >> > (Z, Zp, Zpp, N, V1, V2, lower);
		CUSH_CHECK_LAUNCH();
		velocityCalculator.calculateVelocities(Z, Zp, Zpp, a, aprime, V1, V2, velocities, lower);
	}
	virtual void CalculateRhsPhi(const ProblemPointers problemPointers, std_complex* result, InterfaceParameters& properties) override
	{
		compute_rhs_helium_phi_expression << <this->blocks, this->threads >> > (problemPointers.Z, problemPointers.VelocitiesLower, result, properties.depth, batchSize * N);
		CUSH_CHECK_LAUNCH();
	}
};
