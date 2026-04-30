#pragma once
#ifndef DELAYED_INTENSITY_INTEGRATOR_H
#define DELAYED_INTENSITY_INTEGRATOR_H
#include "AutonomousProblem.h"
#include "constants.cuh"
#include "OptomechanicalVariables.h"

template <int N, size_t batchSize>
class DelayedIntensityIntegrator : public AutonomousProblem<std_complex, 3 * N* batchSize>
{
private:
	OptomechanicalVariables& variables;
	const int threads = 256; ///< Number of threads per block for CUDA kernels
	const int blocks = (batchSize * N + threads - 1) / threads; ///< Number of blocks for CUDA kernels, ensuring all elements are covered
public:
	DelayedIntensityIntegrator(OptomechanicalVariables& variables) : variables(variables)
	{
	}

	virtual void run(std_complex* initialState, std_complex* rhs) override
	{
		std_complex* const devZ = initialState;
		std_complex* const devRhsPhi = rhs + N * batchSize; // The rhs of phi is stored in the middle of the rhs vector (dZdt, dPhidt, dDelayedIntensity) each of size N*batchSize
		
		std_complex* const devDelayedIntensity = initialState + 2 * N * batchSize; // The delayed intensity is stored at the end of the state vector (Z , Phi, DelayedIntensity) each of size N*batchSize
		std_complex* const devRhsDelayedIntensity = rhs + 2 * N * batchSize; // The rhs of the delayed intensity is stored at the end of the rhs vector (dZdt, dPhidt, dDelayedIntensity) each of size N*batchSize

		// we need to calculate the rhs of the delayed intesity: d/dt DelayedIntensity = beta * (I(real(Z), imag(Z)) - DelayedIntensity) where I is the intensity of the optical mode
		calculate_intensity_delayed_rhs<N* batchSize><<<this->blocks, this->threads>>>(devRhsDelayedIntensity, devZ, devDelayedIntensity, variables);
		// add the delayed intensity term to the rhs of phi:
		add_delayed_intensity_phi_rhs<N* batchSize><<<this->blocks, this->threads>>>(devRhsPhi,devDelayedIntensity, variables);

	}

	virtual void setStream(cudaStream_t stream) override
	{
		
	}
};

#endif
