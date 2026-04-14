#pragma once
#ifndef DELAYED_INTENSITY_TERM_H

#include "cuda_runtime.h"
#include "cuda.h"
#include <cmath>
#include "OptomechanicalVariables.h"
/// <summary>
/// Delayed intensity integration.
/// This class is responsible for calculating the delayed intensity term in the optomechanical equations, which accounts for the finite response time of the optical field on the superfluid. The delayed intensity is computed by integrating the intensity over a time window defined by Tau, and it is stored in a device pointer for use in CUDA kernels.
/// This works only with fixed time steps!
/// </summary>
/// <typeparam name="N">Number of collacation points on the superfluid film</typeparam>
template <size_t N>
struct DelayedIntensityTermDevice
{
    const OptomechanicalVariables variables;   // device pointer
    double* delayed_intensity;                  // device pointer
    double* prev_time;

    __device__ double calculate_new_delayed_intensity(double currentTime,
        double lightIntensityFelt,
        size_t index)
    {
        return exp(-(currentTime - *prev_time) / variables.Tau) * delayed_intensity[index]
            + lightIntensityFelt;
    }

    __device__ void save_value(double value, double time, size_t index)
    {
        delayed_intensity[index] = value;
		*prev_time = time;
	}
};

template <size_t N>
class DelayedIntensityTerm
{
public:
    double* delayed_intensity = nullptr;
    double* prev_time = nullptr;
    OptomechanicalVariables variables;

	DelayedIntensityTerm(OptomechanicalVariables h_variables) : variables(h_variables)
    {
        cudaMalloc(&delayed_intensity, N * sizeof(double));
		cudaMalloc(&prev_time, sizeof(double));
		double initial_time = variables.initial_time;
		// copy the intial time to the device
		cudaMemcpy(prev_time, &initial_time, sizeof(double), cudaMemcpyHostToDevice);
    }

    ~DelayedIntensityTerm()
    {
        cudaFree(delayed_intensity);
		cudaFree(prev_time);
    }

    DelayedIntensityTermDevice<N> device_view() const
    {
        return DelayedIntensityTermDevice<N>{ variables, delayed_intensity, prev_time };
    }
};

#endif // !DELAYED_INTENSITY_TERM_H
