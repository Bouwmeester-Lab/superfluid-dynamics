#pragma once
#ifndef DERIVATIVES_H
#define DERIVATIVES_H


#include "device_launch_parameters.h"
//#include <thrust/complex.h>
#include "Core/constants.cuh"
#include <cufft.h>
#include <array>
#include <vector>
#include <stdexcept>
#include "Core/CudaChecks.cuh"
#include "Core/VectorKernels.cuh"
#include "Math/FftKernels.cuh"
#include <iostream>
#include "Core/constants.cuh"

__device__ __host__ inline double filterIndexTanh(int m, int N);

__device__ __host__ inline double filterIndexTanh(int m, int N)
{
	return 0.5 * (1 - tanh(40 * (static_cast<double>(m) / N - 0.25)));
}

class FftDerivative final
{
	const int N;         ///< Points per batch
	const int batchSize; ///< Number of independent transforms
	cufftHandle plan = 0;
	/// <summary>
	/// Stores the coefficients in an array
	/// </summary>
	cufftDoubleComplex* coeffs = nullptr;
	/// <summary>
	/// The values to multiply the coefficients with to obtain the derivative.
	/// </summary>
	cufftDoubleComplex* filterCoeffs = nullptr;
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
	FftDerivative(int n, int batchSize = 1) : N(n), batchSize(batchSize)
	{
		if (n < 2 || batchSize < 1)
		{
			throw std::invalid_argument("FftDerivative needs at least 2 points and 1 batch");
		}
	}
	FftDerivative(const FftDerivative&) = delete;
	FftDerivative& operator=(const FftDerivative&) = delete;
	~FftDerivative();
};

/// <summary>
/// Calculates the derivative of Z assuming it contains a linear part like 2*pi*j/N and the Phi derivative in a single batch
/// </summary>
/// <typeparam name="N"></typeparam>
/// <typeparam name="batchSize">Number of batches</typeparam>
class ZPhiDerivative final
{
private:
	const int N;
	const size_t batchSize;
	FftDerivative fftDerivative;
	FftDerivative singleDerivative;
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

	double densityRatio;   // rho of the problem (enters the linear part of phi)
	double backgroundFlow; // U of the problem
public:
	ZPhiDerivative(int n, size_t batchSize, double densityRatio, double backgroundFlow);
	ZPhiDerivative(const ZPhiDerivative&) = delete;
	ZPhiDerivative& operator=(const ZPhiDerivative&) = delete;
	~ZPhiDerivative();
	/// <summary>
	/// Calculates the derivative of Z and Phi.
	/// </summary>
	/// <param name="ZPhi">A single dev array containg Z and Phi back to back.</param>
	/// <param name="ZPhiPrime">A single dev array used as output to contain the derivative of Z and Phi back to back</param>
	/// <param name="Zpp">A single dev array of size N used as the output for the double derivative of Z</param>
	void exec(const std_complex* Z, const std_complex* Phi, std_complex* ZPrime, std_complex* PhiPrime, std_complex* Zpp);
};


inline cudaError_t FftDerivative::initialize(bool filterIndx)
{
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



	CUSH_CHECK(cudaMalloc(&filterCoeffs, sizeof(cufftDoubleComplex) * N * batchSize));
	CUSH_CHECK(cudaMalloc(&coeffs, N * batchSize * sizeof(cufftDoubleComplex))); // allocates the array with all the coefficients of the FFT
	// copy the coefficients to GPU
	//cudaStatus = cudaMemcpy(filterCoeffs, der.data(), sizeof(cufftDoubleComplex) * N * batchSize, cudaMemcpyHostToDevice);
	//if (cudaStatus != cudaSuccess) {
	//	//fprintf(stderr, "cudaMalloc failed!");
	//	return cudaStatus;
	//}
	if(batchSize == 1)
		CUSH_CHECK(cufftPlan1d(&plan, N, CUFFT_Z2Z, batchSize));
	else {
		int n[1] = { N };
		int istride = 1, ostride = 1;
		int idist = static_cast<int>(N);
		int odist = static_cast<int>(N);
		int inembed[1] = { static_cast<int>(N) };
		int onembed[1] = { static_cast<int>(N) };
		//int batch = static_cast<int>(2 * M);

		CUSH_CHECK(cufftPlanMany(&plan, 1, n, inembed, istride, idist,
			onembed, ostride, odist,
			CUFFT_Z2Z, batchSize));
	}
	//printf("Initialized cufft");
	return cudaSuccess;
}

inline void FftDerivative::exec(const std_complex* in, std_complex* out, const bool doubleDev, double scaling, bool filter)
{
	if (coeffs == nullptr)
	{
		throw std::runtime_error("The FFT class wasn't initialized!");
	}

	CUSH_CHECK(cufftExecZ2Z(plan, reinterpret_cast<cufftDoubleComplex*>(const_cast<std_complex*>(in)), coeffs, CUFFT_FORWARD));

	const int threads = 256;
	const int blocks = (batchSize * N + threads - 1) / threads;
	
	
	
#ifdef DEBUG_FFT
	std::vector<cufftDoubleComplex> coeffsHost(N * batchSize);
	CUSH_CHECK(cudaMemcpy(coeffsHost.data(), coeffs, N * batchSize * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost));
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
		CUSH_CHECK_LAUNCH();
	}
	else 
	{
		first_derivative_multiplication << <blocks, threads >> > (coeffs, coeffs, N, batchSize);
		CUSH_CHECK_LAUNCH();
	}
	if (filter) 
	{
		multiply_element_wise<< <blocks, threads >> > (coeffs, filterCoeffs, coeffs, N * batchSize);
		CUSH_CHECK_LAUNCH(); // multiply the coefficients with the filter coefficients
	}
#ifdef  DEBUG_FFT



	CUSH_CHECK(cudaMemcpy(coeffsHost.data(), coeffs, N * batchSize * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost));
	printf("Coefficients after multiplication k:\n\n");
	for (int i = 0; i < N * batchSize; i++)
	{
		printf("%.10e, %.10e \n", coeffsHost[i].x, coeffsHost[i].y);
	}
	CUSH_CHECK(cudaDeviceSynchronize());
#endif //  DEBUG_FFT
	CUSH_CHECK(cufftExecZ2Z(plan, coeffs, reinterpret_cast<cufftDoubleComplex*>(out), CUFFT_INVERSE)); // doesn't normalize by 1/N https://stackoverflow.com/questions/14441142/scaling-in-inverse-fft-by-cufft
#ifdef DEBUG_FFT
	CUSH_CHECK(cudaMemcpy(coeffsHost.data(), out, N * batchSize * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost));
	printf("Final result after inverse FFT:\n\n");
	for (int i = 0; i < N * batchSize; i++)
	{
		printf("%.10e, %.10e \n", coeffsHost[i].x, coeffsHost[i].y);
	}
	std::cin.get();
#endif // DEBUG_FFT
	if(scaling != 1.0) 
	{
		vector_mutiply_scalar << <blocks, threads >> > (out, scaling, out, N * batchSize, 0);
		CUSH_CHECK_LAUNCH(); // multiply by the scaling factor
	}
}

inline FftDerivative::~FftDerivative()
{
	if (plan) CUSH_CHECK_NOTHROW(cufftDestroy(plan));
	// delete all pointers
	if (filterCoeffs) CUSH_CHECK_NOTHROW(cudaFree(filterCoeffs));
	if (coeffs) CUSH_CHECK_NOTHROW(cudaFree(coeffs));
	//cudaFree()

	CUSH_CHECK_NOTHROW(cudaGetLastError());
}

inline ZPhiDerivative::ZPhiDerivative(int n, size_t batchSize, double densityRatio, double backgroundFlow) : N(n), batchSize(batchSize),
	fftDerivative(n, static_cast<int>(batchSize)), singleDerivative(n, static_cast<int>(batchSize)), densityRatio(densityRatio), backgroundFlow(backgroundFlow)
{
	//std::cout << "Initializing ZPhiDerivative\n";
	fftDerivative.initialize();
	singleDerivative.initialize();

	std::vector<double> ZLinear(N);
	std::vector<double> PhiLinear(N);
	for (int j = 0; j < N; j++) 
	{
		ZLinear[j] = 2 * PI_d * (double)j / N;
		PhiLinear[j] = -(1 + densityRatio) * PI_d * backgroundFlow / N * (double)j; // Phi linear part
	}
	// copy the array to the device
	CUSH_CHECK(cudaMalloc(&devLinearPartZ, N * sizeof(double)));
	CUSH_CHECK(cudaMalloc(&devLinearPartPhi, N * sizeof(double)));

	CUSH_CHECK(cudaMemcpy(devLinearPartZ, ZLinear.data(), N * sizeof(double), cudaMemcpyHostToDevice));
	CUSH_CHECK(cudaMemcpy(devLinearPartPhi, PhiLinear.data(), N * sizeof(double), cudaMemcpyHostToDevice));

	CUSH_CHECK(cudaMalloc(&devPeriodicZ, batchSize * N * sizeof(std_complex)));
	CUSH_CHECK(cudaMalloc(&devPeriodicPhi, batchSize * N * sizeof(std_complex)));

	//std::cout << "Initialized ZPhiDerivative\n";
}

inline ZPhiDerivative::~ZPhiDerivative()
{
	CUSH_CHECK_NOTHROW(cudaFree(devLinearPartZ));
	CUSH_CHECK_NOTHROW(cudaFree(devLinearPartPhi));
	CUSH_CHECK_NOTHROW(cudaFree(devPeriodicZ));
	CUSH_CHECK_NOTHROW(cudaFree(devPeriodicPhi));
}

inline void ZPhiDerivative::exec(const std_complex* Z, const std_complex* Phi, std_complex* ZPrime, std_complex* PhiPrime, std_complex* Zpp)
{
	const int threads = 256;
	const int blocks2N = (2*N * batchSize + threads - 1) / threads;
	const int blocks = (batchSize * N + threads - 1) / threads;

	batched_vector_subtract_singletime_complex_real <<< blocks, threads>>>(Z, devLinearPartZ, devPeriodicZ, N, batchSize);
		CUSH_CHECK_LAUNCH();
	batched_vector_subtract_singletime_complex_real << < blocks, threads >> > (Phi, devLinearPartPhi, devPeriodicPhi, N, batchSize);
	CUSH_CHECK_LAUNCH(); // subtract the linear part of Z and Phi
	//cudaDeviceSynchronize();

	fftDerivative.exec(devPeriodicZ, Zpp, true, 4.0 * PI_d * PI_d / (N*N)); // TODO: check the scaling here, not sure about dividing by N again here since the inverse FFT already does that in the coefficients

	fftDerivative.exec(devPeriodicZ, ZPrime, false, 2.0 * PI_d /N); // 2.0 * PI_d / N
	fftDerivative.exec(devPeriodicPhi, PhiPrime, false, 2.0 * PI_d / N); // calculates the derivative of Z and Phi 2.0 * PI_d / N

	

	//cudaDeviceSynchronize();
	// calculate the double derivative of Z.
	//cudaDeviceSynchronize();
	// add the linear part back
	vector_scalar_add_complex_real << <blocks, threads >> > (ZPrime, 2.0* PI_d/N, ZPrime, N * batchSize, 0);
	CUSH_CHECK_LAUNCH();

	//vector_mutiply_scalar << < blocks2N, threads >> > (ZPhiPrime,  2.0 * PI_d / (double)(N), ZPhiPrime, 2*N, 0); // multiply by 2*pi/N to account for the 2*pi/N term from the dj'/dj where j' = 2*pi/N * j on each derivative
	//
	//vector_mutiply_scalar << < blocks, threads >> > (Zpp, 4.0 * PI_d * PI_d / (N * N), Zpp, N, 0); // multiply by 2*pi/N to account for the 2*pi/N term from the dj'/dj where j' = 2*pi/N * j on each derivative

	if (backgroundFlow != 0) 
	{
		vector_scalar_add_complex_real << <blocks, threads >> > (PhiPrime, -(1 + densityRatio) * PI_d * backgroundFlow / N, PhiPrime, N * batchSize, 0);
		CUSH_CHECK_LAUNCH(); // add the linear part of Phi
	}
}



#endif // !DERIVATIVES_H
