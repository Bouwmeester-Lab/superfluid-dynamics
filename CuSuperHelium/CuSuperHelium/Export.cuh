#pragma once

#include "SimulationFunctions.cuh"
#include "ProblemProperties.hpp"
#include "constants.cuh"
#include "SimulationRunner.cuh"
#include "ExportTypes.cuh"
#include "Derivatives.cuh"
#include "JacobianCalculator.cuh"
#include "RealBoundaryIntegralCalculator.cuh"
#include "GaussLegendre.cuh"
#include <chrono>

extern "C" 
{
	///// <summary>
	///// This function initializes the simulation spaces required for computations. In particular it makes sure that the 
	///// </summary>
	///// <returns></returns>
	//__declspec(dllexport) int initializeSimulationSpaces();

	__declspec(dllexport) int dispertionTest256(double wavelength, double simulationTime, double rho, double kappa, double depth, int steps);
	__declspec(dllexport) int calculateRHS256FromFile(const char* inputFile, const char* outputFile, double L, double rho, double kappa, double depth);
	
	__declspec(dllexport) int calculateRHS256FromVectors(const double* x, const double* y, const double* phi, double* vx, double* vy, double* rhsPhi, double L, double rho, double kappa, double depth);
	__declspec(dllexport) int calculateRHSFromVectors(const double* x, const double* y, const double* phi, double* vx, double* vy, double* rhsPhi, double L, double rho, double kappa, double depth, size_t N);

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
	__declspec(dllexport) int calculateJacobian(const double* state, double* jac, double L, double rho, double kappa, double depth, double epsilon, size_t N);
	__declspec(dllexport) int calculateRHS256FromVectorsBatched(const double* x, const double* y, const double* phi, double* vx, double* vy, double* rhsPhi, double L, double rho, double kappa, double depth, int batchSize);

	__declspec(dllexport) int calculatePerturbedStates256(const double* x, const double* y, const double* phi, c_double* Zperturbed, double L, double rho, double kappa, double depth, double epsilon);
	/// <summary>
	/// Runs a Gauss-Legendre integration simulation with the given initial state and simulation properties.
	/// Remember to free the allocated memory for statesOut and timesOut using integrateSimulationGL2_freeMemory!!
	/// </summary>
	/// <param name="initialState">Initial state provided by caller</param>
	/// <param name="statesOut">A double** for storing the states recorded if recording is enabled. If not, the code will return a single 3*N vector representing the final state.</param>
	/// <param name="statesCount">Number of states of size 3*N saved, if no recording enabled, this will be 1 (the final state).</param>
	/// <param name="timesOut">A dynamically created array to store all the times recorded.</param>
	/// <param name="timesCount">Number of times recorded. This is 0 if recording is disabled.</param>
	/// <param name="simProperties"></param>
	/// <param name="glCOptions"></param>
	/// <returns></returns>
	__declspec(dllexport) int integrateSimulationGL2(double* initialState, double** statesOut, size_t* statesCount, double** timesOut, size_t* timesCount, SimProperties* simProperties, GaussLegendreOptions* glCOptions, size_t N);
	__declspec(dllexport) int integrateSimulationGL2_freeMemory(double* statesOut, double* timesOut);

	ProblemProperties adimensionalizeProperties(ProblemProperties props, double L, double rhoHelium = 150);
}

