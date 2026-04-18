#pragma once
#ifndef OPTOMECHANICAL_VARIABLES_H

struct OptomechanicalVariables
{
public:
	// initial detuning in the experiment
	double detuning = 0.0;
	// optical linewidth of the resonator
	double gamma = 1.0;

	// optomechanical coupling strength (Hz/m)
	double G = 1.0;

	// "delayed" strength of the optical effect on the superfluid.
	double Tau = 1.0;

	// max intensity of the optical field
	double max_intensity;

	double initial_time = 0.0;

	double location_x0_mode = 0.0;
	double sigma_optical_mode = 1.0;
	// 
	double Beta;
	const double hbar = 1.0545718e-34; // Planck's constant over 2pi

	

	__device__ __host__ double calculateDelayedIntensityTermStrength() {
		return 0.0;
	}
};
#endif // !OPTOMECHANICAL_VARIABLES_H
