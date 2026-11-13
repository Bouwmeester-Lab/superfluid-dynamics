#pragma once
#ifndef GENERAL_AUGMENTED_BOUNDARY_INTEGRATOR_CUH
#define GENERAL_AUGMENTED_BOUNDARY_INTEGRATOR_CUH

#include "BoundaryIntegral/Periodic2D/BaseBoundaryIntegrator.cuh"
#include "Optomechanics/DelayedIntensityIntegrator.cuh"
#include "Ode/AutonomousProblem.h"

/// Runs a base boundary integral (Z, Phi) and an augmented problem over the full state (Z, Phi, extra variables).
/// The augmented problem defines the state size.
class GeneralAugmentedBoundaryIntegrator : public AutonomousProblem<std_complex>
{
private:
	std::unique_ptr<AutonomousProblem<std_complex>> m_integrator_base;
	std::unique_ptr<AutonomousProblem<std_complex>> m_integrator_augmented;
public:
	GeneralAugmentedBoundaryIntegrator(std::unique_ptr<AutonomousProblem<std_complex>> integrator_base,
		std::unique_ptr<AutonomousProblem<std_complex>> integrator_augmented) : AutonomousProblem<std_complex>(integrator_augmented->size()), m_integrator_base(std::move(integrator_base)), m_integrator_augmented(std::move(integrator_augmented))
	{}

	virtual ~GeneralAugmentedBoundaryIntegrator()
	{}

	virtual void run(std_complex* initialState, std_complex* rhs) override
	{
		m_integrator_base->run(initialState, rhs);
		m_integrator_augmented->run(initialState, rhs);
	}

	virtual void setStream(cudaStream_t stream) override
	{
		m_integrator_base->setStream(stream);
		m_integrator_augmented->setStream(stream);
	}
};

#endif // !GENERAL_AUGMENTED_BOUNDARY_INTEGRATOR_CUH
