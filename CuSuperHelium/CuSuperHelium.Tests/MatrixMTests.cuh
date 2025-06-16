#pragma once

#include "createM.cuh"
#include "gtest/gtest.h"
#include <complex>
#include <array>
#include "Derivatives.cuh"
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

using namespace std::complex_literals;


double X(double j, double h, double omega, double t) {
	return j - h * std::exp(1i * (j - omega * t)).imag();
}

double Xprime(double j, double h, double omega, double t) {
	return 1 - h * std::cos(j - omega * t);
}

double Xpp(double j, double h, double omega, double t) {
	return h * std::sin(j - omega * t);
}

double Y(double j, double h, double omega, double t) {
	return h * std::exp(1i * (j - omega * t)).real();
}

double Yprime(double j, double h, double omega, double t) {
	return - h * std::sin(j - omega * t);
}

double Ypp(double j, double h, double omega, double t) {
	return -h * std::cos(j - omega * t);
}

double Phi(double j, double h, double omega, double t, double rho) {
	return h * ((1 + rho) * omega * std::exp(1i * (j - omega * t))).imag();
}

double PhiPrime(double j, double h, double omega, double t, double rho) {
	return h * (1.0 + rho) * omega * std::cos(j - omega * t); // without 2 * pi/N scaling factor due to sampling like 2*pi/N * k
}

TEST(Kernels, MMatrixKernel) 
{
	const int N = 4;
	double* devM; // device pointer for the matrix M
	cufftDoubleComplex* devZPhi; // device pointer for ZPhi
	cufftDoubleComplex* devZPhiPrime; // device pointer for ZPhiPrime
	cufftDoubleComplex* devZpp; // device pointer for Zpp

	cudaMalloc(&devM, N * N * sizeof(double));
	cudaMalloc(&devZPhi, N * sizeof(cufftDoubleComplex));
	cudaMalloc(&devZPhiPrime, N * sizeof(cufftDoubleComplex));
	cudaMalloc(&devZpp, N * sizeof(cufftDoubleComplex));

	dim3 matrix_threads(16, 16), matrix_blocks((N + 15) / 16, (N + 15) / 16);

	std::array<std::complex<double>, N> Z = { 0.84147098 + 0.54030231i, 1. + 1.i,
		1.15852902 + 0.54030231i, 2.09070257 - 0.41614684i };



	//createMKernel()
}

/// <summary>
/// Tests the accuracy of the first and second derivative of the functions X and Y using ZPhiDerivative class.
/// </summary>
/// <param name=""></param>
/// <param name=""></param>
TEST(Kernels, ZPhiDerivatives) 
{
	const int N = 64;
	double j = 0;
	double x = 0;
	double y = 0;
	double t = 0.1;
	double h = 0.5;
	double omega = 10;

	ProblemProperties properties;
	properties.rho = 0.0;
	properties.kappa = 0.0;
	properties.U = 0.0;

	std::complex<double> z;
	std::array<cufftDoubleComplex, 2*N> Z;


	std::vector<double> XprimeValues(N, 0);
	std::vector<double> YprimeValues(N, 0);
	std::vector<double> XPrimeCalculatedValues(N, 0);
	std::vector<double> YPrimeCalculatedValues(N, 0);

	std::vector<double> XppValues(N, 0);
	std::vector<double> YppValues(N, 0);

	std::vector<double> XppCalculatedValues(N, 0);
	std::vector<double> YppCalculatedValues(N, 0);


	std::vector<double> PhiPrimeValues(N, 0);
	std::vector<double> PhiPrimeCalculatedValues(N, 0);

	for (int i = 0; i < N; i++) 
	{
		j = 2 * PI_d * i / (double)N;
		Z[i].x = X(j, h, omega, t);
		Z[i].y = Y(j, h, omega, t);

		Z[i + N].x = Phi(j, h, omega, t, properties.rho);
		Z[i + N].y = 0; // Imaginary part is zero since Phi is real.

		XprimeValues[i] = Xprime(j, h, omega, t) * 2.0*PI_d / N;
		YprimeValues[i] = Yprime(j, h, omega, t) * 2.0 * PI_d / N;
		XppValues[i] = Xpp(j, h, omega, t) * 4.0 * PI_d * PI_d / (N * N);
		YppValues[i] = Ypp(j, h, omega, t) * 4.0 * PI_d * PI_d/ (N * N);
		PhiPrimeValues[i] = PhiPrime(j, h, omega, t, properties.rho) * 2.0 * PI_d / N;
	}
	

	ZPhiDerivative<N> zPhiDerivative(properties);
	
	cufftDoubleComplex* devZ;
	cufftDoubleComplex* devZPhiPrime;
	cufftDoubleComplex* devZpp;

	cudaMalloc(&devZ, 2 * N * sizeof(cufftDoubleComplex));
	cudaMalloc(&devZPhiPrime, 2 * N * sizeof(cufftDoubleComplex));
	cudaMalloc(&devZpp, N * sizeof(cufftDoubleComplex));

	cudaMemcpy(devZ, Z.data(), 2 * N * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);

	zPhiDerivative.exec(devZ, devZPhiPrime, devZpp);

	cudaDeviceSynchronize();
	// offload the results to host
	std::array<cufftDoubleComplex, 2*N> ZPhiPrime;
	std::array<cufftDoubleComplex, N> Zpp;
	
	cudaMemcpy(ZPhiPrime.data(), devZPhiPrime, 2 * N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
	cudaMemcpy(Zpp.data(), devZpp, N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
	plt::figure();
	plt::plot(XprimeValues, { {"label", "Expected X Derivative"} });
	

	for (int i = 0; i < N; i++)
	{
		XPrimeCalculatedValues[i] = ZPhiPrime[i].x;
		YPrimeCalculatedValues[i] = ZPhiPrime[i].y;

		XppCalculatedValues[i] = Zpp[i].x;
		YppCalculatedValues[i] = Zpp[i].y;

		PhiPrimeCalculatedValues[i] = ZPhiPrime[i + N].x;

		EXPECT_NEAR(ZPhiPrime[i].x, XprimeValues[i], 1e-14) << "Xprime mismatch at index " << i;
		EXPECT_NEAR(ZPhiPrime[i].y, YprimeValues[i], 1e-14) << "Yprime mismatch at index " << i;

		EXPECT_NEAR(Zpp[i].x, XppValues[i], 1e-14) << "Xpp mismatch at index " << i;
		EXPECT_NEAR(Zpp[i].y, YppValues[i], 1e-14) << "Ypp mismatch at index " << i;

		EXPECT_NEAR(ZPhiPrime[i + N].x, PhiPrimeValues[i], 1e-14) << "PhiPrime mismatch at index " << i;
	}

	plt::plot(XPrimeCalculatedValues, { {"label", "Calculated X Prime"} });

	plt::plot(YprimeValues, { {"label", "Expected Y Derivative"} });
	plt::plot(YPrimeCalculatedValues, { {"label", "Calculated Y Prime"} });
	plt::xlabel("Index");
	plt::ylabel("Value");
	plt::legend();
	plt::figure();

	plt::plot(XppValues, { {"label", "Expected X Second Derivative"} });
	plt::plot(XppCalculatedValues, { {"label", "Calculated X Second Derivative"} });

	plt::plot(YppValues, { {"label", "Expected Y Second Derivative"} });
	plt::plot(YppCalculatedValues, { {"label", "Calculated Y Second Derivative"} });
	plt::legend();
	plt::xlabel("Index");
	plt::ylabel("Value");

	plt::figure();
	plt::plot(PhiPrimeValues, { {"label", "Expected Phi Prime"} });
	plt::plot(PhiPrimeCalculatedValues, { {"label", "Calculated Phi Prime"} });

	plt::legend();
	//plt::show(false);
}