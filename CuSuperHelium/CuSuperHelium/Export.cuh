#pragma once

#include "SimulationFunctions.cuh"
#include "ProblemProperties.hpp"
#include "constants.cuh"
#include "SimulationRunner.cuh"

extern "C" 
{
	__declspec(dllexport) int dispertionTest256(double wavelength, double simulationTime, double rho, double kappa, double depth, int steps);
	__declspec(dllexport) int calculateRHS256(const char* inputFile, const char* outputFile, double L, double rho, double kappa, double depth);
	ProblemProperties adimensionalizeProperties(ProblemProperties props, double L, double rhoHelium = 150);
}

