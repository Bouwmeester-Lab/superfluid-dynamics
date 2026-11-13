#pragma once
#ifndef EXPORT_API_H
#define EXPORT_API_H

#include "Export/ExportTypes.cuh"
#include "Optomechanics/OptomechanicalVariables.h"
#include "Physics/DimensionlessParameters.h"
#include "BoundaryIntegral/Axisymmetric/RadialVelocityCalculator.cuh"
#include <cstddef>

extern "C" 
{
	// Periodic-2D exports take the number of collocation points N (any N >= 2) as an argument.

	/// RHS of the state stored in an HDF5 file of N points; the result is written to another HDF5 file.
	__declspec(dllexport) int calculateRHSFromFile(const char* inputFile, const char* outputFile, SimProperties* simProperties, size_t N);

	__declspec(dllexport) int calculateRHSFromVectors(const double* x, const double* y, const double* phi, double* vx, double* vy, double* rhsPhi, SimProperties* simProperties, size_t N);
	/// RHS of batchSize independent interfaces of N points each, stored back to back in x, y and phi.
	__declspec(dllexport) int calculateRHSFromVectorsBatched(const double* x, const double* y, const double* phi, double* vx, double* vy, double* rhsPhi, SimProperties* simProperties, size_t N, size_t batchSize);

	/// <summary>
	/// Calculates vorticities and the derivatives of Z from input vectors and physical parameters for an N-point system.
	/// </summary>
	/// <param name="Z">Pointer to an array of doubles representing the Z vector.</param>
	/// <param name="phi">Pointer to an array of doubles representing the phi vector.</param>
	/// <param name="a">Pointer to an array of doubles for output or intermediate calculations.</param>
	/// <param name="Zp">Pointer to an array of doubles to store the first derivative of Z (can be nullptr)</param>
	/// <param name="Zpp">Pointer to an array of doubles to store the second derivative of Z (can be nullptr)</param>
	/// <param name="simProperties">Physical parameters (SI) and solver options.</param>
	/// <param name="N">Number of collocation points.</param>
	/// <returns>Returns an integer status code indicating the success or failure of the calculation.</returns>
	__declspec(dllexport) int calculateVorticitiesFromVectors(const c_double* Z, const c_double* phi, double* a, c_double* Zp, c_double* Zpp, SimProperties* simProperties, size_t N);
	/// <summary>
	/// Surface velocities (v_r, v_z) of an axisymmetric flow from the surface z(r) and potential phi(r) at N points,
	/// using the Bessel-series Green function with the default number of modes (1024). Any N >= 2.
	/// </summary>
	__declspec(dllexport) int calculateVelocitiesRadialSymmetry(const double* r, const double* z, const double* phi, double* vr, double* vz, SimProperties* simProperties, size_t N);
	/// <summary>
	/// Same as calculateVelocitiesRadialSymmetry, with the number of Bessel modes nModes chosen by the caller.
	/// </summary>
	__declspec(dllexport) int calculateVelocitiesRadialSymmetryModes(const double* r, const double* z, const double* phi, double* vr, double* vz, SimProperties* simProperties, size_t N, size_t nModes);

	/// Spectral derivative of N periodic complex samples (no 2 pi / N scaling).
	__declspec(dllexport) int calculateDerivativeFFT(const c_double* input, c_double* output, size_t N);
	__declspec(dllexport) int calculateJacobian(const double* state, double* jac, SimProperties* simProperties, double epsilon, size_t N);

	/// The 3N perturbed states (one per state variable, +epsilon) used by the Jacobian, 6 N^2 complex values.
	__declspec(dllexport) int calculatePerturbedStates(const double* x, const double* y, const double* phi, c_double* Zperturbed, SimProperties* simProperties, double epsilon, size_t N);
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


	__declspec(dllexport) int integrateOptomechanicalSimulationRK4(double* initialState, double** statesOut, size_t* statesCount, double** timesOut, size_t* timesCount, SimProperties* simProperties, RK4SolverOptions* rkOptions, COptomechanicalVariables* optomechanicalVariables, size_t N);
	__declspec(dllexport) int integrateOptomechanicalSimulationRK4_freeMemory(double* statesOut, double* timesOut);

	__declspec(dllexport) int integrateAugmentedOptomechanicalSimulationRK4(double* initialState, double** statesOut, size_t* statesCount, double** timesOut, size_t* timesCount, SimProperties* simProperties, RK4SolverOptions* rkOptions, COptomechanicalVariables* optomechanicalVariables, size_t N);
	__declspec(dllexport) int integrateAugmentedOptomechanicalSimulationRK4_freeMemory(double* statesOut, double* timesOut);

	__declspec(dllexport) int calculateRhsAugmentedOptomechanical(double* state, double* rhs, SimProperties* simProperties, COptomechanicalVariables* optomechanicalVariables, size_t N);

}

#endif
