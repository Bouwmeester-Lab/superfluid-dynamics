#pragma once
#ifndef DERIVATIVES_H
#define DERIVATIVES_H

#include "ProblemProperties.hpp"

#include "device_launch_parameters.h"
//#include <thrust/complex.h>
#include "constants.cuh"
#include <cufft.h>
#include "array"
#include <stdexcept>
#include "utilities.cuh"
#include <iostream>
#include "constants.cuh"


__device__ double filterIndexTanh(int m, int N);

__device__ double filterIndexTanh(int m, int N)
{
	return 0.5 * (1 - tanh(40 * (static_cast<double>(m) / N - 0.25)));
}

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
	FftDerivative<N, 1> singleDerivative;
	/// <summary>
	/// Represents an array containing the linear part of Z: 2*pi/N * j and the linear part of Phi. -(1+rho)*pi*U/N * j
	/// </summary>
	double* devLinearPartZPhi;
	/// <summary>
	/// Holds the periodic Zphi.
	/// </summary>
	cufftDoubleComplex* devPeriodicZPhi;
	ProblemProperties& properties;
public:
	ZPhiDerivative(ProblemProperties& properties);
	~ZPhiDerivative();
	/// <summary>
	/// Calculates the derivative of Z and Phi.
	/// </summary>
	/// <param name="ZPhi">A single dev array containg Z and Phi back to back.</param>
	/// <param name="ZPhiPrime">A single dev array used as output to contain the derivative of Z and Phi back to back</param>
	/// <param name="Zpp">A single dev array of size N used as the output for the double derivative of Z</param>
	void exec(cufftDoubleComplex* ZPhi, cufftDoubleComplex* ZPhiPrime, cufftDoubleComplex* Zpp);
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
			// std::cout << der[batchSize * i + j].y << std::endl;
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
	//printf("Initialized cufft");
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
		printf("failed fft forward");
	}

	const int threads = 256;
	const int blocks = (N + threads - 1) / threads;
	
	complex_pointwise_mul<<<blocks, threads>>>(coeffs, derivativeCoeffs, coeffs, N); // multiplies the coefficients by the derivative
	
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
inline ZPhiDerivative<N>::ZPhiDerivative(ProblemProperties& properties) : properties(properties)
{
	fftDerivative.initialize();
	singleDerivative.initialize();

	std::array<double, 2 * N> ZPhiLinear;
	for (int j = 0; j < N; j++) 
	{
		ZPhiLinear[j] = 2 * PI_d * (double)j / N;
		ZPhiLinear[N + j] = -(1 + properties.rho) * PI_d * properties.U / N * (double)j;
	}
	// copy the array to the device
	cudaMalloc(&devLinearPartZPhi, 2 * N * sizeof(cufftDoubleComplex));
	cudaMemcpy(devLinearPartZPhi, ZPhiLinear.data(), 2 * N * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);

	cudaMalloc(&devPeriodicZPhi, 2 * N * sizeof(cufftDoubleComplex));
}

template<int N>
inline ZPhiDerivative<N>::~ZPhiDerivative()
{
	cudaFree(devLinearPartZPhi);
	cudaFree(devPeriodicZPhi);
}

template<int N>
inline void ZPhiDerivative<N>::exec(cufftDoubleComplex* ZPhi, cufftDoubleComplex* ZPhiPrime, cufftDoubleComplex* Zpp)
{
	const int threads = 256;
	const int blocks = (N + threads - 1) / threads;

	vector_subtract_complex_real<<< blocks, threads>>>(ZPhi, devLinearPartZPhi, devPeriodicZPhi, 2 * N); 
	fftDerivative.exec(devPeriodicZPhi, ZPhiPrime);
	//cudaDeviceSynchronize();
	// calculate the double derivative of Z.
	singleDerivative.exec(ZPhiPrime, Zpp);
	//cudaDeviceSynchronize();
	// add the linear part back
	vector_scalar_add_complex_real << <blocks, threads >> > (ZPhiPrime, 2.0 * PI_d / (double)N, ZPhiPrime, N, 0);
	vector_scalar_add_complex_real << <blocks, threads >> > (ZPhiPrime, -(1 + properties.rho) * PI_d * properties.U / N, ZPhiPrime, N, N);

}



#endif // !DERIVATIVES_H