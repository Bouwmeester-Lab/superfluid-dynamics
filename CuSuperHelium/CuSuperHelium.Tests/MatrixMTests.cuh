#pragma once

#include "createM.cuh"
#include "gtest/gtest.h"
#include <complex>
#include <array>
#include "Derivatives.cuh"
#include "WaterVelocities.cuh"
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

void prepareZPhi(cufftDoubleComplex* ZPhi, cufftDoubleComplex* ZPhiPrime, cufftDoubleComplex* Zpp, double h, double omega, double t, double rho, int N)
{
	double j = 0;
	for (int i = 0; i < N; i++)
	{
		j = 2 * PI_d * i / (double)N;
		ZPhi[i].x = X(j, h, omega, t);
		ZPhi[i].y = Y(j, h, omega, t);

		ZPhi[i + N].x = Phi(j, h, omega, t, rho);
		ZPhi[i + N].y = 0; // Imaginary part is zero since Phi is real.

		ZPhiPrime[i].x = Xprime(j, h, omega, t) * 2.0 * PI_d / N;
		ZPhiPrime[i].y = Yprime(j, h, omega, t) * 2.0 * PI_d / N;

		Zpp[i].x = Xpp(j, h, omega, t) * 4.0 * PI_d * PI_d / (N * N);
		Zpp[i].y = Ypp(j, h, omega, t) * 4.0 * PI_d * PI_d / (N * N);

		ZPhiPrime[i + N].x = PhiPrime(j, h, omega, t, rho) * 2.0 * PI_d / N;
		ZPhiPrime[i + N].y = 0; // Imaginary part is zero since PhiPrime is real.
	}
}

void createMMatrix(double* M, double h, double omega, double rho, double t, int N) 
{
	double k1, k2;
	std::complex<double> zpp, zp, zk1, zk2, z_sub;
	for (int i = 0; i < N; i++)
	{
		k1 = 2.0 * PI_d * i / (double)N;
		for (int j = 0; j < N; j++)
		{
			k2 = 2.0 * PI_d * j / (double)N;
			if (i == j)
			{
				zpp = 4.0 * PI_d * PI_d / ( N * N) * (Xpp(k2, h, omega, t) + 1i * Ypp(k2, h, omega, t));
				zp = 2.0*PI_d / N * ( Xprime(k2, h, omega, t) + 1i * Yprime(k2, h, omega, t));
				M[i + j * N] = 0.5 * (1 + rho) + 0.25 * (1 - rho) / PI_d * std::imag(zpp/zp); // imaginary part
			}
			else
			{
				zk1 = X(k1, h, omega, t) + 1i * Y(k1, h, omega, t);
				zk2 = X(k2, h, omega, t) + 1i * Y(k2, h, omega, t);
				z_sub = 0.5 * (zk1 - zk2);
				zp = 2.0 * PI_d / N * ( Xprime(k1, h, omega, t) + 1i * Yprime(k1, h, omega, t));
				M[i + j * N] = 0.25 * (1.0 - rho) / PI_d * std::imag(zp * std::cos(z_sub) / std::sin(z_sub));
			}
		}
	}
}

void createVelocityMatrix(std::complex<double>* V1, std::complex<double>* V2, double h, double omega, double t, int N, bool lower = true) 
{
	double k1, k2;
	std::complex<double> zpp, zp, zk1, zk2, z_sub;
	for (int i = 0; i < N; i++)
	{
		k1 = 2.0 * PI_d * i / (double)N;
		for (int j = 0; j < N; j++)
		{
			k2 = 2.0 * PI_d * j / (double)N;
			if (i == j)
			{
				zpp = 4.0 * PI_d * PI_d / (N * N) * (Xpp(k2, h, omega, t) + 1i * Ypp(k2, h, omega, t));
				zp = 2.0 * PI_d / N * (Xprime(k2, h, omega, t) + 1i * Yprime(k2, h, omega, t));
				V1[i + j * N] = 0.25i/PI_d * zpp / std::pow(zp, 2.0);//std::complex<double>(0.5 * (1 + rho), 0.25 * (1 - rho) / PI_d * std::imag(zpp / zp)); // imaginary part
				if (lower) {
					V1[i + j * N] += 0.5 / zp;
				}
				else {
					V1[i + j * N] -= 0.5 / zp;
				}

				V2[i] = -0.25i / PI_d * 2.0 / zp;
			}
			else
			{
				zk1 = X(k1, h, omega, t) + 1i * Y(k1, h, omega, t);
				zk2 = X(k2, h, omega, t) + 1i * Y(k2, h, omega, t);
				z_sub = 0.5 * (zk1 - zk2);
				//zp = 2.0 * PI_d / N * (Xprime(k1, h, omega, t) + 1i * Yprime(k1, h, omega, t));
				V1[i + j * N] = 0.25i/PI_d * std::cos(z_sub) / std::sin(z_sub);
			}
		}
	}
}

TEST(Kernels, Cotangent) 
{
	const int N = 64;


}

TEST(Kernels, MMatrixKernel) 
{
	const int N = 512;
	double t = 0.1;
	double h = 0.5;
	double omega = 10;
	double rho = 0.0;
	double* devM; // device pointer for the matrix M
	cufftDoubleComplex* devZPhi; // device pointer for ZPhi
	cufftDoubleComplex* devZPhiPrime; // device pointer for ZPhiPrime
	cufftDoubleComplex* devZpp; // device pointer for Zpp

	cudaMalloc(&devM, N * N * sizeof(double));
	cudaMalloc(&devZPhi, 2 * N * sizeof(cufftDoubleComplex));
	cudaMalloc(&devZPhiPrime, 2 * N * sizeof(cufftDoubleComplex));
	cudaMalloc(&devZpp, N * sizeof(cufftDoubleComplex));

	dim3 matrix_threads(16, 16), matrix_blocks((N + 15) / 16, (N + 15) / 16);

	std::array<cufftDoubleComplex, 2 * N> ZPhi;
	std::array<cufftDoubleComplex, N> Zpp;
	std::array<cufftDoubleComplex, 2 * N> ZPhiPrime;
	std::vector<double> MMatrix(N * N);
	std::vector<double> MMatrixCalculated(N * N);

	prepareZPhi(ZPhi.data(), ZPhiPrime.data(), Zpp.data(), h, omega, t, rho, N);
	// Copy the data to device
	cudaMemcpy(devZPhi, ZPhi.data(), 2 * N * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
	cudaMemcpy(devZPhiPrime, ZPhiPrime.data(), 2 * N * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
	cudaMemcpy(devZpp, Zpp.data(), N * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
	
	// for comparison to the kernel output
	createMMatrix(MMatrix.data(), h, omega, rho, t, N);
	cudaDeviceSynchronize();
	createMKernel<<<matrix_blocks, matrix_threads>>>(devM, devZPhi, devZPhiPrime, devZpp, 0.0, N);
	cudaDeviceSynchronize();
	cudaMemcpy(MMatrixCalculated.data(), devM, N * N * sizeof(double), cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			EXPECT_NEAR(MMatrixCalculated[i + j * N], MMatrix[i + j * N], 1e-14) << "Mismatch at (" << i << ", " << j << ")";
		}
	}
}

TEST(Kernels, Velocities) {
	const int N = 512;
	double t = 0.1;
	double h = 0.5;
	double omega = 10;
	double rho = 0.0;

	std::array<cufftDoubleComplex, 2 * N> ZPhi;
	std::array<cufftDoubleComplex, N> Zpp;
	std::array<cufftDoubleComplex, 2 * N> ZPhiPrime;
	prepareZPhi(ZPhi.data(), ZPhiPrime.data(), Zpp.data(), h, omega, t, rho, N);

	cufftDoubleComplex* devZPhi; // device pointer for ZPhi
	cufftDoubleComplex* devZPhiPrime; // device pointer for ZPhiPrime
	cufftDoubleComplex* devZpp; // device pointer for Zpp
	cufftDoubleComplex* devV1; // device pointer for V1
	cufftDoubleComplex* devV2; // device pointer for V2


	cudaMalloc(&devV1, N * N * sizeof(cufftDoubleComplex));
	cudaMalloc(&devV2, N * sizeof(cufftDoubleComplex)); // V2 is a diagonal matrix, so we only need N elements

	cudaMalloc(&devZPhi, 2 * N * sizeof(cufftDoubleComplex));
	cudaMalloc(&devZPhiPrime, 2 * N * sizeof(cufftDoubleComplex));
	cudaMalloc(&devZpp, N * sizeof(cufftDoubleComplex));
	//
	cudaMemcpy(devZPhi, ZPhi.data(), 2 * N * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
	cudaMemcpy(devZPhiPrime, ZPhiPrime.data(), 2 * N * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
	cudaMemcpy(devZpp, Zpp.data(), N * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
	//
	std::vector<std::complex<double>> V1(N*N);
	std::array<std::complex<double>, N> V2;

	createVelocityMatrix(V1.data(), V2.data(), h, omega, t, N);

	dim3 matrix_threads(16, 16), matrix_blocks((N + 15) / 16, (N + 15) / 16);
	createVelocityMatrices<<<matrix_blocks, matrix_threads >>>(devZPhi, devZPhiPrime, devZpp, N, devV1, devV2, true);

	cudaDeviceSynchronize();

	std::vector<cufftDoubleComplex> V1Calculated(N*N);
	std::array<cufftDoubleComplex, N> V2Calculated;

	cudaMemcpy(V1Calculated.data(), devV1, N * N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
	cudaMemcpy(V2Calculated.data(), devV2, N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
	
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) 
		{
			int indx = i + j * N;
			EXPECT_NEAR(V1Calculated[indx].x, V1[indx].real(), 1e-12) << "V1 real part mismatch at (" << i << ", " << j << ")";
			EXPECT_NEAR(V1Calculated[indx].y, V1[indx].imag(), 1e-12) << "V1 imag part mismatch at (" << i << ", " << j << ")";

			if (i == j) {
				EXPECT_NEAR(V2Calculated[i].x, V2[i].real(), 1e-12) << "V2 real part mismatch at index " << i;
				EXPECT_NEAR(V2Calculated[i].y, V2[i].imag(), 1e-12) << "V2 imag part mismatch at index " << i;
			}
		}
	}

}



/// <summary>
/// Tests the accuracy of the first and second derivative of the functions X and Y using ZPhiDerivative class.
/// </summary>
/// <param name=""></param>
/// <param name=""></param>
TEST(Kernels, ZPhiDerivatives) 
{
	const int N = 512;
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

/// <summary>
/// Tests the accuracy of the first and second derivative of the functions X and Y using ZPhiDerivative class.
/// </summary>
/// <param name=""></param>
/// <param name=""></param>
TEST(Kernels, ZPhiDerivativesTranslation)
{
	const int N = 32;
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
	std::array<cufftDoubleComplex, 2 * N> Z;


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
		Z[i].x = X(j, h, omega, t) + 2;
		Z[i].y = Y(j, h, omega, t);

		Z[i + N].x = Phi(j, h, omega, t, properties.rho);
		Z[i + N].y = 0; // Imaginary part is zero since Phi is real.

		XprimeValues[i] = Xprime(j, h, omega, t) * 2.0 * PI_d / N;
		YprimeValues[i] = Yprime(j, h, omega, t) * 2.0 * PI_d / N;
		XppValues[i] = Xpp(j, h, omega, t) * 4.0 * PI_d * PI_d / (N * N);
		YppValues[i] = Ypp(j, h, omega, t) * 4.0 * PI_d * PI_d / (N * N);
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
	std::array<cufftDoubleComplex, 2 * N> ZPhiPrime;
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