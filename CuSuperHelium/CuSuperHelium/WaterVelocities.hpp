#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utilities.cuh"
#include "cufft.h"
#include "constants.cuh"
#define Coeff_Vel 1.0/(4.0 * PI_d) // Coefficient for the velocities in the water model 
/// <summary>
/// Forms the matrix and vector needed for obtaining the velocities in the water model.
/// </summary>
/// <param name="ZPhi"></param>
/// <param name="ZPhiPrime"></param>
/// <param name="Zpp"></param>
/// <param name="N"></param>
/// <param name="out1">V1 matrix</param>
/// <param name="out2">Diagonal entries of V2</param>
/// <param name="lower"></param>
void createVelocityMatrices(cufftDoubleComplex* ZPhi, cufftDoubleComplex* ZPhiPrime, cufftDoubleComplex* Zpp, int N, cufftDoubleComplex* out1, cufftDoubleComplex* out2, bool lower = true);
/// <summary>
/// Calculates the veloctities in the water model from the V1 and V2 matrices calculated by createVelocityMatrices.
/// </summary>
/// <param name="a">The vorticities strengths</param>
/// <param name="aprime">The derivative of a</param>
/// <param name="V1">An NxN matrix allowing for the calculation of the velocities.</param>
/// <param name="V2">An N vector representing the diagonal entries of V2 (purely diagonal matrix)</param>
/// <param name="N">Size of the system.</param>
///void calculateVelocities(double* a, double* aprime, cufftDoubleComplex* V1, cufftDoubleComplex* V2, int N);

void calculateDiagonalVectorMultiplication(cufftDoubleComplex* diag, cufftDoubleComplex* vec, cufftDoubleComplex* out, int N);
//void createVelocityMatrices(double* vorticities, double aprime, cufftDoubleComplex* ZPhi, cufftDoubleComplex* ZPhiPrime, cufftDoubleComplex* Zpp, double* u1, double* v1, double* u2, double* v2);

class VelocityCalculator {
private:
	cublasHandle_t handle; ///< cuBLAS handle for managing BLAS operations
public:
	VelocityCalculator();
	~VelocityCalculator();
	void calculateVelocities(cufftDoubleComplex* a, cufftDoubleComplex* aprime, cufftDoubleComplex* V1, cufftDoubleComplex* V2, int N, cufftDoubleComplex* velocities);
};