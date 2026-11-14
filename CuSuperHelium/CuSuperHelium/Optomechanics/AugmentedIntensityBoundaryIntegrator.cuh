#pragma once
#ifndef AUGMENTED_INTENSITY_BOUNDARY_INTEGRATOR_CUH
#define AUGMENTED_INTENSITY_BOUNDARY_INTEGRATOR_CUH

#include "BoundaryIntegral/Periodic2D/BaseBoundaryIntegrator.cuh"
#include "Optomechanics/DelayedIntensityIntegrator.cuh"
#include "Ode/AutonomousProblem.h"

class AugmentedIntensityBoundaryIntegrator : public AutonomousProblem<std_complex>
{
private:
	std::unique_ptr<BaseBoundaryIntegralCalculator> m_integrator;
	std::unique_ptr<DelayedRampedIntensityIntegrator> m_delayedIntensityIntegrator;
public:
	AugmentedIntensityBoundaryIntegrator(std::unique_ptr<BaseBoundaryIntegralCalculator> integrator,
		std::unique_ptr<DelayedRampedIntensityIntegrator> delayedIntegrator) : AutonomousProblem<std_complex>(delayedIntegrator->size()), m_integrator(std::move(integrator)), m_delayedIntensityIntegrator(std::move(delayedIntegrator))
	{}

	virtual ~AugmentedIntensityBoundaryIntegrator()
	{}
	/// <summary>
	/// Runs the augmented boundary integrator, which includes the original boundary integral calculation and additional calculations for the augmented part. 
	/// The result is stored in the rhs vector, which should have a size of 3 * N * batchSize + 1 to accommodate the delayed intensity field and scalar ramped intensity.
	/// The additional variables are linked to the delayed intensity term and the linear intensity ramp, each with an ODE associated to it.
	/// This allows us to drop the explicit time dependence.
	/// </summary>
	/// <param name="initialState"></param>
	/// <param name="rhs"></param>
	virtual void run(const std_complex* initialState, std_complex* rhs) override
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


#endif // !AUGMENTED_INTENSITY_BOUNDARY_INTEGRATOR_CUH
