#pragma once

#include "SimulationFunctions.cuh"

extern "C" {
	int dispertionTest256(double wavelength, double simulationTime, double rho, double kappa, double depth, int steps);
}

