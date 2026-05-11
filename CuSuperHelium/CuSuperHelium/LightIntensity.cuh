
#pragma once
#ifndef LIGHT_INTENSITY_H
#define LIGHT_INTENSITY_H

#include "cuda_runtime.h"
#include "OptomechanicalVariables.h"
#include "constants.cuh"
#include "ProblemProperties.hpp"

class LightIntensity
{
private:
	//OptomechanicalVariables variables;
public:
	//__device__ __host__ LightIntensity(OptomechanicalVariables variables) : variables(variables) {}
	__device__ __host__ LightIntensity() {}

	static __device__ __host__ double compute_x_profile(double x, OptomechanicalVariables variables)
	{
		return exp(-pow(x - variables.location_x0_mode, 2) / (2 * pow(variables.sigma_optical_mode, 2)));
	}

	static __device__ __host__ double compute_intensity(double fluid_height, double x, OptomechanicalVariables variables)
	{
		double delta_f = variables.detuning + variables.G * fluid_height;

		return 0.25 * cuda::std::pow(variables.gamma, 2.0) * variables.max_intensity / ( cuda::std::pow(delta_f, 2.0) + cuda::std::pow(variables.gamma / 2, 2.0)) * compute_x_profile(x, variables);
	}

	

	static __device__ __host__ inline double get_current_intensity_drive_strength(OptomechanicalVariables variables, ProblemProperties properties)
	{
		return hbar_d / (properties.base_energy * properties.base_time) * variables.G; // G is in a.u. of frequency per length. So hbar gets converted adimensionalized this way.
	}
};

#endif // LIGHT_INTENSITY_H