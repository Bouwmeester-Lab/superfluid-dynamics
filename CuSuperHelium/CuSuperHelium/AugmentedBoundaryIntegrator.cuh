#pragma once
#ifndef AUGMENTED_BOUNDARY_INTEGRATOR_CUH
#define AUGMENTED_BOUNDARY_INTEGRATOR_CUH

#include "BaseBoundaryIntegrator.cuh"
#include "DelayedIntensityIntegrator.cuh"
#include "AutonomousProblem.h"

template<int N, size_t batchSize>
class AugmentedBoundaryIntegrator : public AutonomousProblem<std_complex, 3 * N * batchSize>
{
private:
	std::unique_ptr<BaseBoundaryIntegralCalculator<N, batchSize>> m_integrator;
	std::unique_ptr<DelayedIntensityIntegrator<N, batchSize>> m_delayedIntensityIntegrator;
public:
	AugmentedBoundaryIntegrator(std::unique_ptr<BaseBoundaryIntegralCalculator<N, batchSize>> integrator, 
								std::unique_ptr<DelayedIntensityIntegrator<N, batchSize>> delayedIntegrator) : m_integrator(std::move(integrator)), m_delayedIntensityIntegrator(std::move(delayedIntegrator))
	{
	}

	virtual ~AugmentedBoundaryIntegrator()
	{
	}
	/// <summary>
	/// Runs the augmented boundary integrator, which includes the original boundary integral calculation and additional calculations for the augmented part. 
	/// The result is stored in the rhs vector, which should have a size of 3 * N * batchSize to accommodate the additional calculations.
	/// The additional variable is linked to the delayed intensity term which becomes it's own variable with an ODE associated to it.
	/// This allows us to drop the explicit time dependence.
	/// </summary>
	/// <param name="initialState"></param>
	/// <param name="rhs"></param>
	virtual void run(std_complex* initialState, std_complex* rhs) override
	{
		m_integrator->run(initialState, rhs);
		m_delayedIntensityIntegrator->run(initialState, rhs);
	}
	virtual void setStream(cudaStream_t stream) override
	{
		m_integrator->setStream(stream);
		m_delayedIntensityIntegrator->setStream(stream);
	}
};


#endif // !AUGMENTED_BOUNDARY_INTEGRATOR_CUH
