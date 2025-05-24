#pragma once
#ifndef Derivatives_H

#include <cufft.h>
#include <thrust/complex.h>
#include "array"
#include <stdexcept>
#include "utilities.cuh"
#include <iostream>

template <int N, int batchSize>
class FftDerivative 
{
	cufftHandle plan;
	/// <summary>
	/// Stores the coefficients in an array
	/// </summary>
	cufftDoubleComplex* coeffs;
	/// <summary>
	/// The values to multiply the coefficients with to obtain the derivative.
	/// </summary>
	cufftDoubleComplex* derivativeCoeffs;
public:
	/// <summary>
	/// Creates the FFT execution plan required for the computation.
	/// </summary>
	cudaError_t initialize(bool filterIndx = false);
	/// <summary>
	/// Derivates in with FFT and writes the result in out. Make sure to have run setDevice at least once before using.
	/// </summary>
	/// <param name="in">Must be of size N*batchSize</param>
	/// <param name="out">Must be of size N*batchSize</param>
	void exec(cufftDoubleComplex* in, cufftDoubleComplex* out);
	FftDerivative() {};
	~FftDerivative();
};

/// <summary>
/// Calculates the derivative of Z assuming it contains a linear part like 2*pi*j/N and the Phi derivative in a single batch
/// </summary>
/// <typeparam name="N"></typeparam>
/// <typeparam name="batchSize"></typeparam>
template <int N>
class ZPhiDerivative 
{
private:
	FftDerivative<N, 2> fftDerivative;
	/// <summary>
	/// Represents an array containing the linear part of Z: 2*pi/N * j
	/// </summary>
	double* devLinearPartZ;
	/// <summary>
	/// Represents an array containing the linear part of Phi. -(1+rho)*ppi*U/N * j
	/// </summary>
	double* devLinearPartPhi;
public:
	ZPhiDerivative();
};

double filterIndexTanh(int m, int N);

template <int N, int batchSize>
cudaError_t FftDerivative<N, batchSize>::initialize(bool filterIndx)
{
	cudaError_t cudaStatus;

	std::array<cufftDoubleComplex, N * batchSize> der;
	for (int i = 0; i < batchSize; i++)
	{
		for (int j = 0; j < N; j++) {
			der[batchSize * i + j].x = 0;
			if (j < N / 2)
			{
				der[batchSize * i + j].y = j;
			}
			else
			{
				der[batchSize * i + j].y = j - N;
			}
			if (filterIndx)
			{
				der[batchSize * i + j].y = filterIndexTanh(abs(der[batchSize * i + j].y), N);
			}
			std::cout << der[batchSize * i + j].y << std::endl;
		}
	}




	cudaStatus = cudaMalloc(&derivativeCoeffs, sizeof(cufftDoubleComplex) * N * batchSize);
	if (cudaStatus != cudaSuccess) {
		//fprintf(stderr, "cudaMalloc failed!");
		return cudaStatus;
	}
	cudaStatus = cudaMalloc(&coeffs, N * batchSize * sizeof(cufftDoubleComplex)); // allocates the array with all the coefficients of the FFT
	if (cudaStatus != cudaSuccess) {
		//fprintf(stderr, "cudaMalloc failed!");
		return cudaStatus;
	}
	// copy the coefficients to GPU
	cudaStatus = cudaMemcpy(derivativeCoeffs, der.data(), sizeof(cufftDoubleComplex) * N * batchSize, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		//fprintf(stderr, "cudaMalloc failed!");
		return cudaStatus;
	}
	cufftPlan1d(&plan, N, CUFFT_Z2Z, batchSize);

	return cudaStatus;
}

template<int N, int batchSize>
void FftDerivative<N, batchSize>::exec(cufftDoubleComplex* in, cufftDoubleComplex* out)
{
	if (coeffs == nullptr)
	{
		throw std::runtime_error("The FFT class wasn't initialized!");
	}

	auto result = cufftExecZ2Z(plan, in, coeffs, CUFFT_FORWARD);

	if (result != CUFFT_SUCCESS) {
		printf("failed");
	}

	const int threads = 256;
	const int blocks = (N + threads - 1) / threads;
	
	complex_pointwise_mul << <blocks, threads >> > (coeffs, derivativeCoeffs, coeffs, N); // multiplies the coefficients by the derivative
	
	cufftExecZ2Z(plan, coeffs, out, CUFFT_INVERSE); // doesn't normalize by 1/N https://stackoverflow.com/questions/14441142/scaling-in-inverse-fft-by-cufft
}

template<int N, int batchSize>
FftDerivative<N, batchSize>::~FftDerivative()
{
	cufftDestroy(plan);
	// delete all pointers
	cudaFree(derivativeCoeffs);
	cudaFree(coeffs);
	//cudaFree()
}

template<int N>
inline ZPhiDerivative<N>::ZPhiDerivative()
{
	fftDerivative.initialize();

}

#endif // !Derivatives_H


