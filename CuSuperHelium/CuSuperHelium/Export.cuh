#pragma once

#include "SimulationFunctions.cuh"
#include "ProblemProperties.hpp"
#include "constants.cuh"
#include "SimulationRunner.cuh"
#include "ExportTypes.cuh"
#include "Derivatives.cuh"
#include "JacobianCalculator.cuh"

extern "C" 
{
	
	__declspec(dllexport) int dispertionTest256(double wavelength, double simulationTime, double rho, double kappa, double depth, int steps);
	__declspec(dllexport) int calculateRHS256FromFile(const char* inputFile, const char* outputFile, double L, double rho, double kappa, double depth);
	
	__declspec(dllexport) int calculateRHS256FromVectors(const double* x, const double* y, const double* phi, double* vx, double* vy, double* rhsPhi, double L, double rho, double kappa, double depth);

	__declspec(dllexport) int calculateRHS2048FromVectors(const double* x, const double* y, const double* phi, double* vx, double* vy, double* rhsPhi, double L, double rho, double kappa, double depth);

	/// <summary>
	/// Calculates vorticities and their derivatives from input vectors and physical parameters for a 256-point system.
	/// </summary>
	/// <param name="Z">Pointer to an array of doubles representing the Z vector.</param>
	/// <param name="phi">Pointer to an array of doubles representing the phi vector.</param>
	/// <param name="a">Pointer to an array of doubles for output or intermediate calculations.</param>
	/// <param name="Zp">Pointer to an array of doubles to store the first derivative of Z (can be nullptr)</param>
	/// <param name="Zpp">Pointer to an array of doubles to store the second derivative of Z (can be nullptr)</param>
	/// <param name="L">The length parameter of the system.</param>
	/// <param name="rho">The density parameter.</param>
	/// <param name="kappa">The vorticity parameter.</param>
	/// <param name="depth">The depth parameter.</param>
	/// <returns>Returns an integer status code indicating the success or failure of the calculation.</returns>
	__declspec(dllexport) int calculateVorticities256FromVectors(const c_double* Z, const c_double* phi, double* a, c_double* Zp, c_double* Zpp, double L, double rho, double kappa, double depth);
	__declspec(dllexport) int calculateDerivativeFFT256(const c_double* input, c_double* output);
	__declspec(dllexport) int calculateJacobian256(const c_double* Zphi, const double* jac);
	__declspec(dllexport) int calculateRHS256FromVectorsBatched(const double* x, const double* y, const double* phi, double* vx, double* vy, double* rhsPhi, double L, double rho, double kappa, double depth, int batchSize);

	__declspec(dllexport) int calculatePerturbedStates256(const double* x, const double* y, const double* phi, c_double* Zperturbed, double L, double rho, double kappa, double depth, double epsilon);
	ProblemProperties adimensionalizeProperties(ProblemProperties props, double L, double rhoHelium = 150);
}

