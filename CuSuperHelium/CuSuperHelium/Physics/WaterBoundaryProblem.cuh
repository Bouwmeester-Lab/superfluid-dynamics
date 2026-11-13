#pragma once

#include "BoundaryIntegral/Periodic2D/BoundaryProblem.cuh"
#include "BoundaryIntegral/Periodic2D/MatrixKernels.cuh"
#include "BoundaryIntegral/Periodic2D/PhiRhsKernels.cuh"

class WaterBoundaryProblem : public BoundaryProblem
{
	VelocityCalculator velocityCalculator; ///< Velocity calculator for calculating the velocities based on the vorticities and matrices.
public:
	WaterBoundaryProblem(DimensionlessParameters& properties, int n, size_t batchSize = 1) : BoundaryProblem(n, batchSize, new KineticEnergy(properties, n), new GravitationalEnergy(properties, n), new SurfaceEnergy(properties, n)), velocityCalculator(n, batchSize)
	{
		// Constructor for the water boundary problem, initializing the velocity calculator with the problem properties
	}

	virtual void CreateMMatrix(double* M, const std_complex* Z, const std_complex* Zp, const std_complex* Zpp, InterfaceParameters& properties) override
	{
		createMKernel << <this->matrix_blocks, this->matrix_threads >> > (M, Z, Zp, Zpp, properties.rho, N, batchSize);
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
		createVelocityMatrices << <this->matrix_blocks, this->matrix_threads >> > (Z, Zp, Zpp, N, V1, V2, lower, batchSize);
		CUSH_CHECK_LAUNCH();
		velocityCalculator.calculateVelocities(Z, Zp, Zpp, a, aprime, V1, V2, velocities, lower);
	}
	virtual void CalculateRhsPhi(const ProblemPointers problemPointers, std_complex* result, InterfaceParameters& properties) override
	{
		compute_rhs_phi_expression << <this->blocks, this->threads >> > (problemPointers.Z, problemPointers.VelocitiesLower, problemPointers.VelocitiesUpper, result, properties.rho, batchSize * N);
		CUSH_CHECK_LAUNCH();
	}
};
