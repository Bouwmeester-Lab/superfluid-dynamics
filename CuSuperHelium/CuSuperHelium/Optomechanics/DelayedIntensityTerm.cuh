#pragma once
#ifndef DELAYED_INTENSITY_TERM_H

#include "cuda_runtime.h"
#include "cuda.h"
#include <cmath>
#include "Optomechanics/OptomechanicalVariables.h"
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
        if(currentTime == *prev_time) {
            return 0.0;
		}
        double a = exp(-(currentTime - *prev_time) / variables.Tau);
        return a * delayed_intensity[index]
            + (1 - a) * lightIntensityFelt;
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
        CUSH_CHECK(cudaMalloc(&delayed_intensity, N * sizeof(double)));
		CUSH_CHECK(cudaMalloc(&prev_time, sizeof(double)));
		double initial_time = variables.initial_time;
		// copy the intial time to the device
		CUSH_CHECK(cudaMemcpy(prev_time, &initial_time, sizeof(double), cudaMemcpyHostToDevice));
    }

    ~DelayedIntensityTerm()
    {
        CUSH_CHECK_NOTHROW(cudaFree(delayed_intensity));
		CUSH_CHECK_NOTHROW(cudaFree(prev_time));
    }

    DelayedIntensityTermDevice<N> device_view()
    {
        return DelayedIntensityTermDevice<N>{ variables, delayed_intensity, prev_time };
    }

    void setInitialTime(double initial_time)
    {
        CUSH_CHECK(cudaMemcpy(prev_time, &initial_time, sizeof(double), cudaMemcpyHostToDevice));
	}
};

#endif // !DELAYED_INTENSITY_TERM_H
