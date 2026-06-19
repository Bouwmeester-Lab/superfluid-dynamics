#pragma once
#ifndef GENERAL_AUGMENTED_BOUNDARY_INTEGRATOR_CUH
#define GENERAL_AUGMENTED_BOUNDARY_INTEGRATOR_CUH

#include "BaseBoundaryIntegrator.cuh"
#include "DelayedIntensityIntegrator.cuh"
#include "AutonomousProblem.h"

template <int N, size_t batchSize, size_t extra_variables>
class GeneralAugmentedBoundaryIntegrator : public AutonomousProblem<std_complex, 3 * N * batchSize + extra_variables>
{
private:
	std::unique_ptr< AutonomousProblem<std_complex, 2 * N * batchSize>> m_integrator_base;
	std::unique_ptr<AutonomousProblem<std_complex, 3 * N * batchSize + extra_variables>> m_integrator_augmented;
public:
	GeneralAugmentedBoundaryIntegrator(std::unique_ptr<AutonomousProblem<std_complex, 2 * N * batchSize>> integrator_base,
		std::unique_ptr<AutonomousProblem<std_complex, 3 * N * batchSize + extra_variables>> integrator_augmented) : m_integrator_base(std::move(integrator_base)), m_integrator_augmented(std::move(integrator_augmented))
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
