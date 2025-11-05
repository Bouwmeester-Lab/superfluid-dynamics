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
class FftDerivative final
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
	void exec(const std_complex* in, std_complex* out, const bool doubleDev = false, double scaling = 1.0, bool filter = false);
	FftDerivative() {};
	~FftDerivative();
};

/// <summary>
/// Calculates the derivative of Z assuming it contains a linear part like 2*pi*j/N and the Phi derivative in a single batch
/// </summary>
/// <typeparam name="N"></typeparam>
/// <typeparam name="batchSize">Number of batches</typeparam>
template <int N, size_t batchSize>
class ZPhiDerivative final
{
private:
	FftDerivative<N, batchSize> fftDerivative;
	FftDerivative<N, batchSize> singleDerivative;
	/// <summary>
	/// Represents an array containing the linear part of Z: 2*pi/N * j and the linear part of Phi. -(1+rho)*pi*U/N * j
	/// </summary>
	double* devLinearPartZ;
	double* devLinearPartPhi;
	/// <summary>
	/// Holds the periodic Zphi.
	/// </summary>
	std_complex* devPeriodicZ;
	std_complex* devPeriodicPhi;

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
	void exec(const std_complex* Z, const std_complex* Phi, std_complex* ZPrime, std_complex* PhiPrime, std_complex* Zpp);
};

double filterIndexTanh(int m, int N);

template <int N, int batchSize>
cudaError_t FftDerivative<N, batchSize>::initialize(bool filterIndx)
{
	cudaError_t cudaStatus;

	std::vector<cufftDoubleComplex> der(N * batchSize);
	// TODO: fix filtering for batchSize > 512. Currently no filtering is used.
	//std::vector<double> derReal(N * batchSize);
	//for (int i = 0; i < batchSize; i++)
	//{
	//	for (int j = 0; j < N; j++) {
	//		/*der[batchSize * i + j].x = 0;
	//		if (j < N / 2)
	//		{
	//			der[batchSize * i + j].y = j;
	//		}
	//		else
	//		{
	//			der[batchSize * i + j].y = j - N;
	//		}*/
	//		
	//		der[batchSize * i + j].x = filterTanh(abs(j - N/2), N, 0.01*PI_d, 0.2);
	//		der[batchSize * i + j].y = 0; // the filter coefficients are only real, so the imaginary part is 0

	//		if (der[batchSize * i + j].x < 1e-11)
	//		{
	//			der[batchSize * i + j].x = 0.0; // set the small values to 0 exactly
	//		}


	//		derReal[batchSize * i + j] = der[batchSize * i + j].x;

	//		
	//		// std::cout << der[batchSize * i + j].y << std::endl;
	//	}
	//}
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
	//cudaStatus = cudaMemcpy(filterCoeffs, der.data(), sizeof(cufftDoubleComplex) * N * batchSize, cudaMemcpyHostToDevice);
	//if (cudaStatus != cudaSuccess) {
	//	//fprintf(stderr, "cudaMalloc failed!");
	//	return cudaStatus;
	//}
	if(batchSize == 1)
		cufftPlan1d(&plan, N, CUFFT_Z2Z, batchSize);
	else {
		int n[1] = { N };
		int istride = 1, ostride = 1;
		int idist = static_cast<int>(N);
		int odist = static_cast<int>(N);
		int inembed[1] = { static_cast<int>(N) };
		int onembed[1] = { static_cast<int>(N) };
		//int batch = static_cast<int>(2 * M);

		cufftPlanMany(&plan, 1, n, inembed, istride, idist,
			onembed, ostride, odist,
			CUFFT_Z2Z, batchSize);
	}
	//printf("Initialized cufft");
	return cudaStatus;
}

template<int N, int batchSize>
void FftDerivative<N, batchSize>::exec(const std_complex* in, std_complex* out, const bool doubleDev, double scaling, bool filter)
{
	if (coeffs == nullptr)
	{
		throw std::runtime_error("The FFT class wasn't initialized!");
	}

	auto result = cufftExecZ2Z(plan, reinterpret_cast<cufftDoubleComplex*>(const_cast<std_complex*>(in)), coeffs, CUFFT_FORWARD);

	if (result != CUFFT_SUCCESS) {
		printf("failed fft forward");
	}

	const int threads = 256;
	const int blocks = (batchSize * N + threads - 1) / threads;
	
	
	
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
		
		second_derivative_fft<< <blocks, threads>> >(coeffs, coeffs, N, batchSize);
	}
	else 
	{
		first_derivative_multiplication << <blocks, threads >> > (coeffs, coeffs, N, batchSize);
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
	cufftExecZ2Z(plan, coeffs, reinterpret_cast<cufftDoubleComplex*>(out), CUFFT_INVERSE); // doesn't normalize by 1/N https://stackoverflow.com/questions/14441142/scaling-in-inverse-fft-by-cufft
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
	if(plan) cufftDestroy(plan);
	// delete all pointers
	if(filterCoeffs) cudaFree(filterCoeffs);
	if(coeffs) cudaFree(coeffs);
	//cudaFree()

	auto error = cudaGetLastError();
	if (error != cudaSuccess) {
		std::cerr << "CUDA error in FftDerivative destructor: " << cudaGetErrorString(error) << std::endl;
	}
}

template<int N, size_t batchSize>
inline ZPhiDerivative<N, batchSize>::ZPhiDerivative(ProblemProperties& properties) : properties(properties)
{
	//std::cout << "Initializing ZPhiDerivative\n";
	fftDerivative.initialize();
	singleDerivative.initialize();

	std::array<double, N> ZLinear;
	std::array<double, N> PhiLinear;
	for (int j = 0; j < N; j++) 
	{
		ZLinear[j] = 2 * PI_d * (double)j / N;
		PhiLinear[j] = -(1 + properties.rho) * PI_d * properties.U / N * (double)j; // Phi linear part
	}
	// copy the array to the device
	checkCuda(cudaMalloc(&devLinearPartZ, N * sizeof(double)));
	checkCuda(cudaMalloc(&devLinearPartPhi, N * sizeof(double)));

	checkCuda(cudaMemcpy(devLinearPartZ, ZLinear.data(), N * sizeof(double), cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(devLinearPartPhi, PhiLinear.data(), N * sizeof(double), cudaMemcpyHostToDevice));

	checkCuda(cudaMalloc(&devPeriodicZ, batchSize * N * sizeof(std_complex)));
	checkCuda(cudaMalloc(&devPeriodicPhi, batchSize * N * sizeof(std_complex)));

	//std::cout << "Initialized ZPhiDerivative\n";
}

template<int N, size_t batchSize>
inline ZPhiDerivative<N, batchSize>::~ZPhiDerivative()
{
	cudaFree(devLinearPartZ);
	cudaFree(devLinearPartPhi);
	cudaFree(devPeriodicZ);
	cudaFree(devPeriodicPhi);
}

template<int N, size_t batchSize>
inline void ZPhiDerivative<N, batchSize>::exec(const std_complex* Z, const std_complex* Phi, std_complex* ZPrime, std_complex* PhiPrime, std_complex* Zpp)
{
	const int threads = 256;
	const int blocks2N = (2*N * batchSize + threads - 1) / threads;
	const int blocks = (batchSize * N + threads - 1) / threads;

	batched_vector_subtract_singletime_complex_real <<< blocks, threads>>>(Z, devLinearPartZ, devPeriodicZ, N, batchSize);
	batched_vector_subtract_singletime_complex_real << < blocks, threads >> > (Phi, devLinearPartPhi, devPeriodicPhi, N, batchSize); // subtract the linear part of Z and Phi
	//cudaDeviceSynchronize();

	fftDerivative.exec(devPeriodicZ, Zpp, true, 4.0 * PI_d * PI_d / (N*N)); // TODO: check the scaling here, not sure about dividing by N again here since the inverse FFT already does that in the coefficients

	fftDerivative.exec(devPeriodicZ, ZPrime, false, 2.0 * PI_d /N); // 2.0 * PI_d / N
	fftDerivative.exec(devPeriodicPhi, PhiPrime, false, 2.0 * PI_d / N); // calculates the derivative of Z and Phi 2.0 * PI_d / N

	

	//cudaDeviceSynchronize();
	// calculate the double derivative of Z.
//#define DEBUG_FFT
#ifdef DEBUG_FFT
	cudaDeviceSynchronize();
	std::array<cufftDoubleComplex,  1* N> periodicPhiHost;
	std::array<cufftDoubleComplex, 1 * N> PhiHost;
	std::array<cufftDoubleComplex, 1 * N> phiPrimeHost;
	//std::array<cufftDoubleComplex, 2*N> ZPhiHost;
	//std::vector<double> xp(N, 0);
	//std::vector<double> yp(N, 0);

	//std::vector<double> xpp(N, 0);
	//std::vector<double> ypp(N, 0);

	std::vector<double> phiPeriodic(N, 0);
	std::vector<double> phiOriginal(N, 0);
	std::vector<double> phiPrime(N, 0);
	cudaMemcpy(periodicPhiHost.data(), devPeriodicPhi, N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
	cudaMemcpy(PhiHost.data(), Phi, N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
	cudaMemcpy(phiPrimeHost.data(), PhiPrime, N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);

	//printf("Zphi prime\n");
	for (int i = 0; i < N; i++)
	{
		
		phiPeriodic[i] = periodicPhiHost[i].x;
		phiOriginal[i] = PhiHost[i].x;
		phiPrime[i] = phiPrimeHost[i].x;
	}
	plt::figure();
	/*plt::plot(x, { {"label", "x"} });
	plt::plot(y, { {"label", "y"} });*/
	//plt::plot(phiPeriodic, { {"label", "phi periodic"} });
	//plt::plot(phiOriginal, { {"label", "phi"} });
	plt::plot(phiPrime, { {"label", "phi prime'"} });
	//plt::plot(yp, { {"label", "yp"} });

	//plt::plot(xpp, { {"label", "xpp"} });
	//plt::plot(ypp, { {"label", "ypp"} });
	plt::legend();
	plt::show();
	std::cin.get();
#endif
	//cudaDeviceSynchronize();
	// add the linear part back
	vector_scalar_add_complex_real << <blocks, threads >> > (ZPrime, 2.0* PI_d/N, ZPrime, N * batchSize, 0);

	//vector_mutiply_scalar << < blocks2N, threads >> > (ZPhiPrime,  2.0 * PI_d / (double)(N), ZPhiPrime, 2*N, 0); // multiply by 2*pi/N to account for the 2*pi/N term from the dj'/dj where j' = 2*pi/N * j on each derivative
	//
	//vector_mutiply_scalar << < blocks, threads >> > (Zpp, 4.0 * PI_d * PI_d / (N * N), Zpp, N, 0); // multiply by 2*pi/N to account for the 2*pi/N term from the dj'/dj where j' = 2*pi/N * j on each derivative

	if (properties.U != 0) 
	{
		vector_scalar_add_complex_real << <blocks, threads >> > (PhiPrime, -(1 + properties.rho) * PI_d * properties.U / N, PhiPrime, N * batchSize, 0); // add the linear part of Phi
	}
}



#endif // !DERIVATIVES_H