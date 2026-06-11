#pragma once
#ifndef DELAYED_INTENSITY_INTEGRATOR_H
#define DELAYED_INTENSITY_INTEGRATOR_H
#include "AutonomousProblem.h"
#include "constants.cuh"
#include "OptomechanicalVariables.h"
#include "ProblemProperties.hpp"

template <int N, size_t batchSize>
class DelayedIntensityIntegrator : public AutonomousProblem<std_complex, 3 * N* batchSize>
{
private:
	ProblemProperties& properties;
	OptomechanicalVariables& variables;
	std::shared_ptr<LightIntensity> intensity;
	const int threads = 256; ///< Number of threads per block for CUDA kernels
	const int blocks = (batchSize * N + threads - 1) / threads; ///< Number of blocks for CUDA kernels, ensuring all elements are covered
	cudaStream_t stream = cudaStreamPerThread;
public:
	DelayedIntensityIntegrator(OptomechanicalVariables& variables, ProblemProperties& properties, std::shared_ptr<LightIntensity> intensity) : variables(variables), properties(properties), intensity(intensity)
	{
	}

	virtual void run(std_complex* initialState, std_complex* rhs) override
	{
		std_complex* const devZ = initialState;
		std_complex* const devRhsPhi = rhs + N * batchSize; // The rhs of phi is stored in the middle of the rhs vector (dZdt, dPhidt, dDelayedIntensity) each of size N*batchSize
		
		std_complex* const devDelayedIntensity = initialState + 2 * N * batchSize; // The delayed intensity is stored at the end of the state vector (Z , Phi, DelayedIntensity) each of size N*batchSize
		std_complex* const devRhsDelayedIntensity = rhs + 2 * N * batchSize; // The rhs of the delayed intensity is stored at the end of the rhs vector (dZdt, dPhidt, dDelayedIntensity) each of size N*batchSize
		double* devFrequencyShift = intensity->get_dev_frequency_shift();
		// we need to calculate the rhs of the delayed intesity: d/dt DelayedIntensity = beta * (I(real(Z), imag(Z)) - DelayedIntensity) where I is the intensity of the optical mode
		calculate_intensity_delayed_rhs<N* batchSize><<<this->blocks, this->threads, 0, this->stream>>>(devRhsDelayedIntensity, devFrequencyShift, devZ, devDelayedIntensity, variables);
		// add the delayed intensity term to the rhs of phi:
		add_delayed_intensity_phi_rhs<N* batchSize><<<this->blocks, this->threads, 0, this->stream>>>(devRhsPhi,devDelayedIntensity, variables, properties);

	}

	virtual void setStream(cudaStream_t stream) override
	{
		this->stream = stream;
	}
};

template <int N, size_t batchSize>
class DelayedRampedIntensityIntegrator : public AutonomousProblem<std_complex, 3 * N * batchSize + 1>
{
private:
	ProblemProperties& properties;
	OptomechanicalVariables& variables;
	std::shared_ptr<LightIntensity> intensity;
	const int threads = 256; ///< Number of threads per block for CUDA kernels
	const int blocks = (batchSize * N + threads - 1) / threads; ///< Number of blocks for CUDA kernels, ensuring all elements are covered
	cudaStream_t stream = cudaStreamPerThread;
public:
	DelayedRampedIntensityIntegrator(OptomechanicalVariables& variables, ProblemProperties& properties, std::shared_ptr<LightIntensity> intensity) : variables(variables), properties(properties), intensity(intensity)
	{}

	virtual void run(std_complex* initialState, std_complex* rhs) override
	{
		std_complex* const devZ = initialState;
		std_complex* const devRhsPhi = rhs + N * batchSize; // The rhs of phi is stored after dZdt.

		std_complex* const devDelayedIntensity = initialState + 2 * N * batchSize; // The delayed intensity is stored at the end of the state vector (Z , Phi, DelayedIntensity) each of size N*batchSize
		std_complex* const devRhsDelayedIntensity = rhs + 2 * N * batchSize; // The rhs of the delayed intensity is stored at the end of the rhs vector (dZdt, dPhidt, dDelayedIntensity) each of size N*batchSize
		std_complex* const devRhsRampedIntensity = rhs + 3 * N * batchSize; // The scalar ramped intensity rhs is stored after the field components.
		double* devFrequencyShift = intensity->get_dev_frequency_shift();
		std_complex* const devRampedIntensity = initialState + 3 * N * batchSize; // The scalar ramped intensity is stored after Z, Phi, and DelayedIntensity.

		// we need to calculate the rhs of the delayed intesity: d/dt DelayedIntensity = beta * (I(real(Z), imag(Z)) - DelayedIntensity) where I is the intensity of the optical mode
		calculate_ramped_intensity_delayed_rhs<N* batchSize> << <this->blocks, this->threads, 0, this->stream >> > (devRhsDelayedIntensity, devRampedIntensity, devFrequencyShift, devZ, devDelayedIntensity, variables);
		// add the delayed intensity term to the rhs of phi:
		add_delayed_intensity_phi_rhs<N* batchSize> << <this->blocks, this->threads, 0, this->stream >> > (devRhsPhi, devDelayedIntensity, variables, properties);
		// add the ramped intensity rhs of d/dt RampedIntensity = ramp_rate // since I = ramp_rate * t
		
		set_growth_intensity_rate << <1, 1, 0, this->stream >> > (devRhsRampedIntensity, variables.ramp_rate);
	}

	virtual void setStream(cudaStream_t stream) override
	{
		this->stream = stream;
	}
};

#endif
