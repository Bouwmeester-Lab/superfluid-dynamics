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
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;
#if DEBUG_FFT
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;
#endif // DEBUG_FFT

__device__ double filterIndexTanh(int m, int N);

__device__ double filterIndexTanh(int m, int N)
{
	return 0.5 * (1 - tanh(40 * (static_cast<double>(m) / N - 0.25)));
}

template <typename T, int N>
class FftDerivativeBase
{
	protected:
	/// <summary>
	/// The FFT plan used for the computation.
	/// </summary>
	cufftHandle plan;
public:
	virtual cudaError_t initialize(bool filterIndx = false) = 0;
	void exec(T* in, T* out) = 0;
};

template <int N>
class ComplexFftDerivative : public FftDerivativeBase<cufftDoubleComplex, N>
{

};

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
	cufftDoubleComplex* filterCoeffs;
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
	void exec(cufftDoubleComplex* in, cufftDoubleComplex* out, const bool doubleDev = false, double scaling = 1.0, bool filter = false);
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
	FftDerivative<N, 1> fftDerivative;
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

	std::vector<cufftDoubleComplex> der(N * batchSize);
	std::vector<double> derReal(N * batchSize);
	for (int i = 0; i < batchSize; i++)
	{
		for (int j = 0; j < N; j++) {
			/*der[batchSize * i + j].x = 0;
			if (j < N / 2)
			{
				der[batchSize * i + j].y = j;
			}
			else
			{
				der[batchSize * i + j].y = j - N;
			}*/
			
			der[batchSize * i + j].x = filterTanh(abs(j - N/2), N, 0.1*PI_d, 0.4);
			der[batchSize * i + j].y = 0; // the filter coefficients are only real, so the imaginary part is 0

			if (der[batchSize * i + j].x < 1e-11)
			{
				der[batchSize * i + j].x = 0.0; // set the small values to 0 exactly
			}


			derReal[batchSize * i + j] = der[batchSize * i + j].x;

			
			// std::cout << der[batchSize * i + j].y << std::endl;
		}
	}
	/*plt::figure();
	plt::plot(derReal, { {"label", "filter coefficients"} });*/



	cudaStatus = cudaMalloc(&filterCoeffs, sizeof(cufftDoubleComplex) * N * batchSize);
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
	cudaStatus = cudaMemcpy(filterCoeffs, der.data(), sizeof(cufftDoubleComplex) * N * batchSize, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		//fprintf(stderr, "cudaMalloc failed!");
		return cudaStatus;
	}
	cufftPlan1d(&plan, N, CUFFT_Z2Z, batchSize);
	//printf("Initialized cufft");
	return cudaStatus;
}

template<int N, int batchSize>
void FftDerivative<N, batchSize>::exec(cufftDoubleComplex* in, cufftDoubleComplex* out, const bool doubleDev, double scaling, bool filter)
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
	
	
	
#ifdef DEBUG_FFT
	std::array<cufftDoubleComplex, N* batchSize> coeffsHost;
	cudaMemcpy(coeffsHost.data(), coeffs, N * batchSize * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
	printf("Coefficients after FFT:\n");
	for (int i = 0; i < N * batchSize; i++) 
	{
		printf("%.10e, %.10e \n", coeffsHost[i].x, coeffsHost[i].y);
	}
#endif // DEBUG_FFT
	//cudaDeviceSynchronize();
	if (doubleDev) 
	{
		
		second_derivative_fft<< <blocks, threads>> >(coeffs, coeffs, N);
	}
	else 
	{
		first_derivative_multiplication << <blocks, threads >> > (coeffs, coeffs, N);
	}
	if (filter) 
	{
		multiply_element_wise<< <blocks, threads >> > (coeffs, filterCoeffs, coeffs, N * batchSize); // multiply the coefficients with the filter coefficients
	}
#ifdef  DEBUG_FFT



	cudaMemcpy(coeffsHost.data(), coeffs, N * batchSize * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
	printf("Coefficients after multiplication k:\n\n");
	for (int i = 0; i < N * batchSize; i++)
	{
		printf("%.10e, %.10e \n", coeffsHost[i].x, coeffsHost[i].y);
	}
	cudaDeviceSynchronize();
#endif //  DEBUG_FFT
	cufftExecZ2Z(plan, coeffs, out, CUFFT_INVERSE); // doesn't normalize by 1/N https://stackoverflow.com/questions/14441142/scaling-in-inverse-fft-by-cufft
#ifdef DEBUG_FFT
	cudaMemcpy(coeffsHost.data(), out, N * batchSize * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
	printf("Final result after inverse FFT:\n\n");
	for (int i = 0; i < N * batchSize; i++)
	{
		printf("%.10e, %.10e \n", coeffsHost[i].x, coeffsHost[i].y);
	}
	std::cin.get();
#endif // DEBUG_FFT
	if(scaling != 1.0) 
	{
		vector_mutiply_scalar << <blocks, threads >> > (out, scaling, out, N * batchSize, 0); // multiply by the scaling factor
	}
}

template<int N, int batchSize>
FftDerivative<N, batchSize>::~FftDerivative()
{
	cufftDestroy(plan);
	// delete all pointers
	cudaFree(filterCoeffs);
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
		ZPhiLinear[N + j] = -(1 + properties.rho) * PI_d * properties.U / N * (double)j; // Phi linear part
	}
	// copy the array to the device
	cudaMalloc(&devLinearPartZPhi, 2 * N * sizeof(double));
	cudaMemcpy(devLinearPartZPhi, ZPhiLinear.data(), 2 * N * sizeof(double), cudaMemcpyHostToDevice);

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
	const int blocks2N = (2*N + threads - 1) / threads;
	const int blocks = (N + threads - 1) / threads;

	vector_subtract_complex_real<<< blocks2N, threads>>>(ZPhi, devLinearPartZPhi, devPeriodicZPhi, 2 * N);

	fftDerivative.exec(devPeriodicZPhi, Zpp, true, 4.0*PI_d * PI_d / (N*N));

	fftDerivative.exec(devPeriodicZPhi, ZPhiPrime, false, 2.0 * PI_d / N); // 2.0 * PI_d / N
	fftDerivative.exec(devPeriodicZPhi + N, ZPhiPrime + N, false, 2.0 * PI_d / N); // calculates the derivative of Z and Phi

	

	//cudaDeviceSynchronize();
	// calculate the double derivative of Z.
	
#ifdef DEBUG_FFT
	cudaDeviceSynchronize();
	std::array<cufftDoubleComplex, 2 * N> ZPhiPrimeHost;
	std::array<cufftDoubleComplex, 2 * N> ZppHost;
	std::array<cufftDoubleComplex, 2*N> ZPhiHost;
	std::vector<double> xp(N, 0);
	std::vector<double> yp(N, 0);

	std::vector<double> xpp(N, 0);
	std::vector<double> ypp(N, 0);

	std::vector<double> x(N, 0);
	std::vector<double> y(N, 0);
	cudaMemcpy(ZPhiPrimeHost.data(), ZPhiPrime, 2 * N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
	cudaMemcpy(ZppHost.data(), Zpp, N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
	cudaMemcpy(ZPhiHost.data(), ZPhi, 2 * N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);

	printf("Zphi prime\n");
	for (int i = 0; i < 2 * N; i++)
	{
		if (i < N)
		{
			x[i] = ZPhiHost[i].x;
			y[i] = ZPhiHost[i].y;

			xp[i] = ZPhiPrimeHost[i].x;
			yp[i] = ZPhiPrimeHost[i].y;

			xpp[i] = ZppHost[i].x;
			ypp[i] = ZppHost[i].y;
			printf("%f + i %f\n", ZppHost[i].x, ZppHost[i].y);
		}
		
	}
	plt::figure();
	/*plt::plot(x, { {"label", "x"} });
	plt::plot(y, { {"label", "y"} });*/
	plt::plot(xp, { {"label", "xp"} });
	plt::plot(yp, { {"label", "yp"} });

	plt::plot(xpp, { {"label", "xpp"} });
	plt::plot(ypp, { {"label", "ypp"} });
	plt::legend();
	plt::show();
	std::cin.get();
#endif
	//cudaDeviceSynchronize();
	// add the linear part back, I need to add 1 and multiply everything by 2*pi/N.
	vector_scalar_add_complex_real << <blocks, threads >> > (ZPhiPrime, 2.0* PI_d/N, ZPhiPrime, N, 0); //

	//vector_mutiply_scalar << < blocks2N, threads >> > (ZPhiPrime,  2.0 * PI_d / (double)(N), ZPhiPrime, 2*N, 0); // multiply by 2*pi/N to account for the 2*pi/N term from the dj'/dj where j' = 2*pi/N * j on each derivative
	//
	//vector_mutiply_scalar << < blocks, threads >> > (Zpp, 4.0 * PI_d * PI_d / (N * N), Zpp, N, 0); // multiply by 2*pi/N to account for the 2*pi/N term from the dj'/dj where j' = 2*pi/N * j on each derivative

	if (properties.U != 0) 
	{
		vector_scalar_add_complex_real << <blocks, threads >> > (ZPhiPrime, -(1 + properties.rho) * PI_d * properties.U / N, ZPhiPrime, N, N); // add the linear part of Phi
	}
		
	

}



#endif // !DERIVATIVES_H