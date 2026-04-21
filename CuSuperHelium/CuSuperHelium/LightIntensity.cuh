
#pragma once
#ifndef LIGHT_INTENSITY_H
#define LIGHT_INTENSITY_H

#include "cuda_runtime.h"
#include "OptomechanicalVariables.h"

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
		double delta_f = variables.detuning - variables.G * fluid_height;

		return variables.gamma * variables.gamma / 4.0 * variables.max_intensity / (delta_f * delta_f + (variables.gamma / 2) * (variables.gamma / 2)) * compute_x_profile(x, variables);
	}

	

	static __device__ __host__ inline double get_current_intensity_drive_strength(OptomechanicalVariables variables)
	{
		return  variables.G; // Placeholder for the actual calculation of the current intensity drive strength based on the optomechanical variables.
	}
};

#endif // LIGHT_INTENSITY_H