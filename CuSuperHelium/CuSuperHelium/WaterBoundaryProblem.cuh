#pragma once

#include "BoundaryProblem.cuh"
#include "utilities.cuh"
#include "createM.cuh"

template<int N, size_t batchSize>
class WaterBoundaryProblem : public BoundaryProblem<N, batchSize>
{
	VelocityCalculator<N, batchSize> velocityCalculator; ///< Velocity calculator for calculating the velocities based on the vorticities and matrices.
public:
	WaterBoundaryProblem(ProblemProperties& properties) : BoundaryProblem<N, batchSize>(new KineticEnergy<N>(properties), new GravitationalEnergy<N>(properties), new SurfaceEnergy<N>(properties)), velocityCalculator()
	{
		// Constructor for the water boundary problem, initializing the velocity calculator with the problem properties
	}

	virtual void CreateMMatrix(double* M, const std_complex* Z, const std_complex* Zp, const std_complex* Zpp, ProblemProperties& properties) override
	{
		createMKernel << <this->matrix_blocks, this->matrix_threads >> > (M, Z, Zp, Zpp, properties.rho, N, batchSize);
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
		createVelocityMatrices << <this->matrix_blocks, this->matrix_threads >> > (Z, Zp, Zpp, N, V1, V2, lower, batchSize);
		velocityCalculator.calculateVelocities(Z, Zp, Zpp, a, aprime, V1, V2, velocities, lower);
	}
	virtual void CalculateRhsPhi(const ProblemPointers problemPointers, std_complex* result, ProblemProperties& properties) override
	{
		compute_rhs_phi_expression << <this->blocks, this->threads >> > (problemPointers.Z, problemPointers.VelocitiesLower, problemPointers.VelocitiesUpper, result, properties.rho, batchSize * N);
	}
};