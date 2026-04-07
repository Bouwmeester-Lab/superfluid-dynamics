
#pragma once
#ifndef LIGHT_INTENSITY_H
#define LIGHT_INTENSITY_H

#include "cuda_runtime.h"
#include "OptomechanicalVariables.h"

class LightIntensity
{
private:
	OptomechanicalVariables variables;
public:
	__device__ __host__ LightIntensity(OptomechanicalVariables variables) : variables(variables) {}
	__device__ __host__ LightIntensity() {}

	__device__ __host__ double compute_intensity(double fluid_height) const
	{
		double delta_f = variables.detuning - variables.G * fluid_height;

		return variables.gamma * variables.gamma / 4.0 * variables.max_intensity / (delta_f * delta_f + (variables.gamma / 2) * (variables.gamma / 2));
	}

	__device__ __host__ inline double get_current_intensity_drive_strength() const
	{
		return  1.0; // Placeholder for the actual calculation of the current intensity drive strength based on the optomechanical variables.
	}
};

#endif // LIGHT_INTENSITY_H