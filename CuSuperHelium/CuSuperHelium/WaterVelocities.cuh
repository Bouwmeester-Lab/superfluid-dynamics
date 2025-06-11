#pragma once
#ifndef WATERVELOCITIES_H
#define WATERVELOCITIES_H


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
__global__ void createVelocityMatrices(cufftDoubleComplex* ZPhi, cufftDoubleComplex* ZPhiPrime, cufftDoubleComplex* Zpp, int N, cufftDoubleComplex* out1, cufftDoubleComplex* out2, bool lower = true);
/// <summary>
/// Calculates the veloctities in the water model from the V1 and V2 matrices calculated by createVelocityMatrices.
/// </summary>
/// <param name="a">The vorticities strengths</param>
/// <param name="aprime">The derivative of a</param>
/// <param name="V1">An NxN matrix allowing for the calculation of the velocities.</param>
/// <param name="V2">An N vector representing the diagonal entries of V2 (purely diagonal matrix)</param>
/// <param name="N">Size of the system.</param>
///void calculateVelocities(double* a, double* aprime, cufftDoubleComplex* V1, cufftDoubleComplex* V2, int N);

__global__ void calculateDiagonalVectorMultiplication(cufftDoubleComplex* diag, cufftDoubleComplex* vec, cufftDoubleComplex* out, int N);


__global__ void createVelocityMatrices(cufftDoubleComplex* ZPhi, cufftDoubleComplex* ZPhiPrime, cufftDoubleComplex* Zpp, int N, cufftDoubleComplex* out1, cufftDoubleComplex* out2, bool lower)
{
	int k = blockIdx.y * blockDim.y + threadIdx.y; // row
	int j = blockIdx.x * blockDim.x + threadIdx.x; // col

	if (k < N && j < N) {
		int indx = k + j * N; // column major index
		if (k == j)
		{
			// we are in the diagonal:
			cuDoubleComplex val = cuCdiv(make_cuDoubleComplex(1.0, 0), ZPhiPrime[k]);
			out1[indx] = cuCmul(make_cuDoubleComplex(0, Coeff_Vel), cuCdiv(Zpp[k], cuCmul(ZPhi[k], ZPhi[k]))); // no normalization by N since it would be cancelled out by the division.
			if (lower)
			{
				out1[indx] = cuCadd(out1[indx], cMulScalar(0.5 * N, val)); // the *N is because the derivatives obtained by the FFT derivatives are not normalized by N so this will apply the normalization.
			}
			else
			{
				out1[indx] = cuCsub(out1[indx], cMulScalar(0.5 * N, val));
			}
			out2[k] = cMulScalar(-2.0, val);
		}
		else
		{
			out1[indx] = cuCmul(make_cuDoubleComplex(0, Coeff_Vel), cotangent_complex(cMulScalar(0.5, cuCsub(ZPhi[k], ZPhi[j])))); // no /N here since Z is not obtained by an FFT.
		}
	}
}

__global__ void calculateDiagonalVectorMultiplication(cufftDoubleComplex* diag, cufftDoubleComplex* vec, cufftDoubleComplex* out, int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		out[i] = cuCmul(diag[i], vec[i]);
	}
}


//void createVelocityMatrices(double* vorticities, double aprime, cufftDoubleComplex* ZPhi, cufftDoubleComplex* ZPhiPrime, cufftDoubleComplex* Zpp, double* u1, double* v1, double* u2, double* v2);

template<int N>
class VelocityCalculator {
private:
	cublasHandle_t handle; ///< cuBLAS handle for managing BLAS operations
	const int threads = 256;
	const int blocks = (N + threads - 1) / threads;
	
	const dim3 matrix_threads;// (16, 16);     // 256 threads per block in 2D
	const dim3 matrix_blocks; // ((N + 15) / 16, (N + 15) / 16);

public:
	VelocityCalculator();
	~VelocityCalculator();
	void calculateVelocities(cufftDoubleComplex* ZPhi,
		cufftDoubleComplex* ZPhiPrime,
		cufftDoubleComplex* Zpp,
		cufftDoubleComplex* a,
		cufftDoubleComplex* aprime,
		cufftDoubleComplex* V1,
		cufftDoubleComplex* V2,
		cufftDoubleComplex* velocities,
		bool lower = true);
};

template<int N>
VelocityCalculator<N>::VelocityCalculator() : matrix_threads(16, 16), matrix_blocks((N + 15) / 16, (N + 15) / 16)
{
	cublasCreate(&handle);
}

template<int N>
VelocityCalculator<N>::~VelocityCalculator()
{
	if (handle) {
		cublasDestroy(handle);
	}
}

template<int N>
void VelocityCalculator<N>::calculateVelocities(cufftDoubleComplex* ZPhi,
	cufftDoubleComplex* ZPhiPrime,
	cufftDoubleComplex* Zpp,
	cufftDoubleComplex* a, 
	cufftDoubleComplex* aprime,
	cufftDoubleComplex* V1,
	cufftDoubleComplex* V2,
	cufftDoubleComplex* velocities,
	bool lower)
{
	// create the V1 matrix and V2 diagonal vector
	createVelocityMatrices<<<matrix_blocks, matrix_threads>>>(ZPhi, ZPhiPrime, Zpp, N, V1, V2, lower);

	// calculate v2*aprime
	calculateDiagonalVectorMultiplication << <blocks, threads >> > (V2, aprime, velocities, N);


	// calculate V1 * a + v2*aprime
	const cufftDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
	const cufftDoubleComplex beta = make_cuDoubleComplex(1.0, 0.0);
	cublasZgemv(handle, CUBLAS_OP_N, N, N, &alpha, V1, N, a, 1, &beta, velocities, 1);
	// calculate the conjugate of the velocities, since this df/dz = u - i v
	conjugate_vector<<<blocks, threads >>>(velocities, velocities, N);
}
#endif // !WATERVELOCITIES_H