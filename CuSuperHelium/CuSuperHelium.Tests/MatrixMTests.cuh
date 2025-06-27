#pragma once

#include "createM.cuh"
#include "gtest/gtest.h"
#include <complex>
#include <array>
#include "Derivatives.cuh"
#include "WaterVelocities.cuh"
#include "matplotlibcpp.h"
#include "utilities.cuh"
#include "Jacobian.cuh"
#include "WaterBoundaryIntegralCalculator.cuh"

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

double TheoreticalPhiPrime(double j, double h, double omega, double t, double rho) {
	return h * (1.0 + rho) * omega * std::cos(j - omega * t); // without 2 * pi/N scaling factor due to sampling like 2*pi/N * k
}

void prepareZPhi(std_complex* Z, std_complex* _Phi, std_complex* Zp, std_complex* Zpp, std_complex* _PhiPrime, double h, double omega, double t, double rho, int N)
{
	double j = 0;
	for (int i = 0; i < N; i++)
	{
		j = 2 * PI_d * i / (double)N;
		Z[i] = std_complex(X(j, h, omega, t), Y(j, h, omega, t));

		_Phi[i] = Phi(j, h, omega, t, rho);

		Zp[i] =  std_complex(Xprime(j, h, omega, t) * 2.0 * PI_d / N, Yprime(j, h, omega, t) * 2.0 * PI_d / N);
		Zpp[i] = std_complex(Xpp(j, h, omega, t) * 4.0 * PI_d * PI_d / (N * N), Ypp(j, h, omega, t) * 4.0 * PI_d * PI_d / (N * N));

		_PhiPrime[i] = TheoreticalPhiPrime(j, h, omega, t, rho) * 2.0 * PI_d / N;
		
	}
}

double theoreticalMMatrix(int k, int j, double h) 
{
	if (k == 0 && j == 0) 
	{
		return 0.5 - 1.0 / (4.0) * h /  (1.0 - h) ;
	}
	if( k == 1 && j == 1) 
	{
		return 0.5 + 1.0 / (4.0) * h / (1.0 + h);
	}
	if (k == 1 && j == 0) 
	{
		return (h + 1) / 4.0 * std::sinh(2.0 * h) / (std::cosh(2.0 * h) + 1.0);
	}
	if (k == 0 && j == 1) 
	{
		return (h - 1) / 4.0 * std::sinh(2.0 * h) / (std::cosh(2.0 * h) + 1.0);
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
				M[i + j * N] = 0.25 * (1.0 - rho) / PI_d * std::imag(zp / std::tan(z_sub));
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
				V1[i + j * N] = -0.25i/PI_d * zpp / std::pow(zp, 2.0);//std::complex<double>(0.5 * (1 + rho), 0.25 * (1 - rho) / PI_d * std::imag(zpp / zp)); // imaginary part
				if (lower) {
					V1[i + j * N] += 0.5 / zp;
				}
				else {
					V1[i + j * N] -= 0.5 / zp;
				}

				V2[i] = 0.5i / (PI_d * zp);
			}
			else
			{
				zk1 = X(k1, h, omega, t) + 1i * Y(k1, h, omega, t);
				zk2 = X(k2, h, omega, t) + 1i * Y(k2, h, omega, t);
				z_sub = 0.5 * (zk1 - zk2);
				//zp = 2.0 * PI_d / N * (Xprime(k1, h, omega, t) + 1i * Yprime(k1, h, omega, t));
				V1[i + j * N] = -0.25i/PI_d / std::tan(z_sub);
			}
		}
	}
}

TEST(Kernels, JacobianCalculation) 
{
	const int N = 4;

	ProblemProperties properties;
	properties.rho = 0.0;
	properties.kappa = 0.0;
	properties.U = 0.0;

	WaterBoundaryIntegralCalculator<N> calculator(properties);

	Jacobian<std_complex, 2 * N> jacobian(calculator);

	std::array<std_complex, 2 * N> Z;
	std_complex* devZ;

	for(int i = 0; i < N; i++) 
	{
		double j = 2 * PI_d * i / (double)N;
		Z[i] = std_complex(X(j, 0, 10, 0.0), 0.0);
		Z[i + N] = 0.0;
	}

	//cudaMalloc(&devZ, 2 * N * sizeof(std_complex));
	//cudaMemcpy(devZ, Z.data(), 2 * N * sizeof(std_complex), cudaMemcpyHostToDevice);

	calculator.initialize_device(Z.data(), Z.data() + N);

	jacobian.calculateJacobian(calculator.getY0());

	cudaDeviceSynchronize();
	// Retrieve the Jacobian matrix from the device
	std::vector<std_complex> jacobianMatrix(4 * N * N);
	cudaMemcpy(jacobianMatrix.data(), jacobian.jacobianMatrix, 4 * N * N * sizeof(std_complex), cudaMemcpyDeviceToHost);



}

TEST(Kernels, TwoByTwoMMatrix) 
{
	const int N = 2;

	double t = 0.0;
	double h = 0.5;
	double omega = 10;
	double rho = 0.0;
	double* devM; // device pointer for the matrix M

	std_complex* devZ; // device pointer for ZPhi
	std_complex* devZp; // device pointer for ZPhiPrime
	std_complex* devZpp; // device pointer for Zpp

	std_complex* devPhi;
	std_complex* devPhiPrime;

	cudaMalloc(&devM, N * N * sizeof(double));
	cudaMalloc(&devZ, N * sizeof(std_complex));
	cudaMalloc(&devZp, N * sizeof(std_complex));
	cudaMalloc(&devZpp, N * sizeof(std_complex));

	cudaMalloc(&devPhi, N * sizeof(std_complex));
	cudaMalloc(&devPhiPrime, N * sizeof(std_complex));

	dim3 matrix_threads(16, 16), matrix_blocks((N + 15) / 16, (N + 15) / 16);

	std::array<std_complex, N> Z;
	std::array<std_complex, N> Zp;
	std::array<std_complex, N> Zpp;
	std::array<std_complex, N> ArrPhi;
	std::array<std_complex, N> PhiPrime;
	std::vector<double> MMatrix(N * N);
	std::vector<double> MMatrixCalculated(N * N);

	prepareZPhi(Z.data(), ArrPhi.data(), Zp.data(), Zpp.data(), PhiPrime.data(), h, omega, t, rho, N);

	cudaMemcpy(devZ, Z.data(), N * sizeof(std_complex), cudaMemcpyHostToDevice);
	cudaMemcpy(devZp, Zp.data(), N * sizeof(std_complex), cudaMemcpyHostToDevice);
	cudaMemcpy(devZpp, Zpp.data(), N * sizeof(std_complex), cudaMemcpyHostToDevice);

	std::vector<double> MMatrixExpected(N * N);

	for(int k = 0; k < N; k++)
	{
		for(int j = 0; j < N; j++)
		{
			MMatrixExpected[k + j * N] = theoreticalMMatrix(k, j, h);
		}
	}

	createMKernel << <matrix_blocks, matrix_threads >> > (devM, devZ, devZp, devZpp, 0.0, N);

	cudaDeviceSynchronize();
	cudaMemcpy(MMatrixCalculated.data(), devM, N * N * sizeof(double), cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			EXPECT_NEAR(MMatrixCalculated[i + j * N], MMatrixExpected[i + j * N], 1e-14) << "Mismatch at (" << i << ", " << j << ")";
		}
	}
}

TEST(Kernels, MMatrixKernel) 
{
	const int N = 1024;
	double t = 0.1;
	double h = 0.5;
	double omega = 10;
	double rho = 0.0;
	double* devM; // device pointer for the matrix M

	std_complex* devZ; // device pointer for ZPhi
	std_complex* devZp; // device pointer for ZPhiPrime
	std_complex* devZpp; // device pointer for Zpp

	std_complex* devPhi;
	std_complex* devPhiPrime;

	cudaMalloc(&devM, N * N * sizeof(double));
	cudaMalloc(&devZ, N * sizeof(std_complex));
	cudaMalloc(&devZp, N * sizeof(std_complex));
	cudaMalloc(&devZpp, N * sizeof(std_complex));

	cudaMalloc(&devPhi, N * sizeof(std_complex));
	cudaMalloc(&devPhiPrime, N * sizeof(std_complex));

	dim3 matrix_threads(16, 16), matrix_blocks((N + 15) / 16, (N + 15) / 16);

	std::array<std_complex, N> Z;
	std::array<std_complex, N> Zp;
	std::array<std_complex, N> Zpp;
	std::array<std_complex, N> ArrPhi;
	std::array<std_complex, N> PhiPrime;
	std::vector<double> MMatrix(N * N);
	std::vector<double> MMatrixCalculated(N * N);

	prepareZPhi(Z.data(), ArrPhi.data(), Zp.data(), Zpp.data(), PhiPrime.data(), h, omega, t, rho, N);
	// Copy the data to device
	cudaMemcpy(devZ, Z.data(), N * sizeof(std_complex), cudaMemcpyHostToDevice);
	cudaMemcpy(devZp, Zp.data(), N * sizeof(std_complex), cudaMemcpyHostToDevice);
	cudaMemcpy(devZpp, Zpp.data(), N * sizeof(std_complex), cudaMemcpyHostToDevice);

	cudaMemcpy(devPhi, ArrPhi.data(), N * sizeof(std_complex), cudaMemcpyHostToDevice);
	cudaMemcpy(devPhiPrime, PhiPrime.data(), N * sizeof(std_complex), cudaMemcpyHostToDevice);
	
	// for comparison to the kernel output
	createMMatrix(MMatrix.data(), h, omega, rho, t, N);
	cudaDeviceSynchronize();
	createMKernel<<<matrix_blocks, matrix_threads>>>(devM, devZ, devZp, devZpp, 0.0, N);
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
	const int N = 64;
	double t = 0.1;
	double h = 0.5;
	double omega = 10;
	double rho = 0.0;

	std_complex* devZ; // device pointer for ZPhi
	std_complex* devZp; // device pointer for ZPhiPrime
	std_complex* devZpp; // device pointer for Zpp

	std_complex* devPhi;
	std_complex* devPhiPrime;

	cudaMalloc(&devZ, N * sizeof(std_complex));
	cudaMalloc(&devZp, N * sizeof(std_complex));
	cudaMalloc(&devZpp, N * sizeof(std_complex));

	cudaMalloc(&devPhi, N * sizeof(std_complex));
	cudaMalloc(&devPhiPrime, N * sizeof(std_complex));

	dim3 matrix_threads(16, 16), matrix_blocks((N + 15) / 16, (N + 15) / 16);

	std::array<std_complex, N> Z;
	std::array<std_complex, N> Zp;
	std::array<std_complex, N> Zpp;
	std::array<std_complex, N> ArrPhi;
	std::array<std_complex, N> PhiPrime;

	prepareZPhi(Z.data(), ArrPhi.data(), Zp.data(), Zpp.data(), PhiPrime.data(), h, omega, t, rho, N);

	std_complex* devV1; // device pointer for V1
	std_complex* devV2; // device pointer for V2

	cudaMalloc(&devV1, N * N * sizeof(std_complex)); // V1 is a matrix of size N*N
	cudaMalloc(&devV2, N * sizeof(std_complex)); // V2 is a vector of size N

	cudaMemcpy(devZ, Z.data(), N * sizeof(std_complex), cudaMemcpyHostToDevice);
	cudaMemcpy(devZp, Zp.data(), N * sizeof(std_complex), cudaMemcpyHostToDevice);
	cudaMemcpy(devZpp, Zpp.data(), N * sizeof(std_complex), cudaMemcpyHostToDevice);
	cudaMemcpy(devPhi, ArrPhi.data(), N * sizeof(std_complex), cudaMemcpyHostToDevice);
	cudaMemcpy(devPhiPrime, PhiPrime.data(), N * sizeof(std_complex), cudaMemcpyHostToDevice);

	//
	std::vector<std::complex<double>> V1(N*N);
	std::array<std::complex<double>, N> V2;

	createVelocityMatrix(V1.data(), V2.data(), h, omega, t, N);
	createVelocityMatrices<<<matrix_blocks, matrix_threads >>>(devZ, devZp, devZpp, N, devV1, devV2, true);

	cudaDeviceSynchronize();

	std::vector<std_complex> V1Calculated(N*N);
	std::array<std_complex, N> V2Calculated;

	cudaMemcpy(V1Calculated.data(), devV1, N * N * sizeof(std_complex), cudaMemcpyDeviceToHost);
	cudaMemcpy(V2Calculated.data(), devV2, N * sizeof(std_complex), cudaMemcpyDeviceToHost);
	
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) 
		{
			int indx = i + j * N;
			EXPECT_NEAR(V1Calculated[indx].real(), V1[indx].real(), 1e-12) << "V1 real part mismatch at (" << i << ", " << j << ")";
			EXPECT_NEAR(V1Calculated[indx].imag(), V1[indx].imag(), 1e-12) << "V1 imag part mismatch at (" << i << ", " << j << ")";

			if (i == j) {
				EXPECT_NEAR(V2Calculated[i].real(), V2[i].real(), 1e-12) << "V2 real part mismatch at index " << i;
				EXPECT_NEAR(V2Calculated[i].imag(), V2[i].imag(), 1e-12) << "V2 imag part mismatch at index " << i;
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
	const int N = 1024;
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

	std::array<std_complex, N> Z;
	std::array<std_complex, N> Zp;
	std::array<std_complex, N> Zpp;
	std::array<std_complex, N> ArrPhi;
	std::array<std_complex, N> PhiPrime;

	prepareZPhi(Z.data(), ArrPhi.data(), Zp.data(), Zpp.data(), PhiPrime.data(), h, omega, t, properties.rho, N);
	
	// for plotting:
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


	for(int i = 0; i < N; i++)
	{
		XprimeValues[i] = Zp[i].real();
		YprimeValues[i] = Zp[i].imag();

		XppValues[i] = Zpp[i].real();
		YppValues[i] = Zpp[i].imag();

		PhiPrimeValues[i] = PhiPrime[i].real();
	}

	ZPhiDerivative<N> zPhiDerivative(properties);
	
	std_complex* devZ;
	std_complex* devZp;
	std_complex* devZpp;

	std_complex* devPhi;
	std_complex* devPhiPrime;

	cudaMalloc(&devZ, N * sizeof(std_complex));
	cudaMalloc(&devZp, N * sizeof(std_complex));
	cudaMalloc(&devZpp, N * sizeof(std_complex));

	cudaMalloc(&devPhi, N * sizeof(std_complex));
	cudaMalloc(&devPhiPrime, N * sizeof(std_complex));

	cudaMemcpy(devZ, Z.data(), N * sizeof(std_complex), cudaMemcpyHostToDevice);
	cudaMemcpy(devPhi, ArrPhi.data(), N * sizeof(std_complex), cudaMemcpyHostToDevice);

	zPhiDerivative.exec(devZ, devPhi, devZp, devPhiPrime, devZpp);

	cudaDeviceSynchronize();
	// offload the results to host
	std::array<std_complex, N> ZpCalculated;
	std::array<std_complex, N> ZppCalculated;
	std::array<std_complex, N> PhiPrimeCalculated;
	
	cudaMemcpy(ZpCalculated.data(), devZp, N * sizeof(std_complex), cudaMemcpyDeviceToHost);
	cudaMemcpy(ZppCalculated.data(), devZpp, N * sizeof(std_complex), cudaMemcpyDeviceToHost);
	cudaMemcpy(PhiPrimeCalculated.data(), devPhiPrime, N * sizeof(std_complex), cudaMemcpyDeviceToHost);
	plt::figure();
	plt::plot(XprimeValues, { {"label", "Expected X Derivative"} });
	

	for (int i = 0; i < N; i++)
	{
		XPrimeCalculatedValues[i] = ZpCalculated[i].real();
		YPrimeCalculatedValues[i] = ZpCalculated[i].imag();

		XppCalculatedValues[i] = ZppCalculated[i].real();
		YppCalculatedValues[i] = ZppCalculated[i].imag();

		PhiPrimeCalculatedValues[i] = PhiPrimeCalculated[i].real();

		EXPECT_NEAR(ZpCalculated[i].real(), XprimeValues[i], 1e-14) << "Xprime mismatch at index " << i;
		EXPECT_NEAR(ZpCalculated[i].imag(), YprimeValues[i], 1e-14) << "Yprime mismatch at index " << i;

		EXPECT_NEAR(ZppCalculated[i].real(), XppValues[i], 1e-14) << "Xpp mismatch at index " << i;
		EXPECT_NEAR(ZppCalculated[i].imag(), YppValues[i], 1e-14) << "Ypp mismatch at index " << i;

		EXPECT_NEAR(PhiPrimeCalculated[i].real(), PhiPrimeValues[i], 1e-14) << "PhiPrime mismatch at index " << i;
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
//
///// <summary>
///// Tests the accuracy of the first and second derivative of the functions X and Y using ZPhiDerivative class.
///// </summary>
///// <param name=""></param>
///// <param name=""></param>
//TEST(Kernels, ZPhiDerivativesTranslation)
//{
//	const int N = 32;
//	double j = 0;
//	double x = 0;
//	double y = 0;
//	double t = 0.1;
//	double h = 0.5;
//	double omega = 10;
//
//	ProblemProperties properties;
//	properties.rho = 0.0;
//	properties.kappa = 0.0;
//	properties.U = 0.0;
//
//	std::complex<double> z;
//	std::array<cufftDoubleComplex, 2 * N> Z;
//
//
//	std::vector<double> XprimeValues(N, 0);
//	std::vector<double> YprimeValues(N, 0);
//	std::vector<double> XPrimeCalculatedValues(N, 0);
//	std::vector<double> YPrimeCalculatedValues(N, 0);
//
//	std::vector<double> XppValues(N, 0);
//	std::vector<double> YppValues(N, 0);
//
//	std::vector<double> XppCalculatedValues(N, 0);
//	std::vector<double> YppCalculatedValues(N, 0);
//
//
//	std::vector<double> PhiPrimeValues(N, 0);
//	std::vector<double> PhiPrimeCalculatedValues(N, 0);
//
//	for (int i = 0; i < N; i++)
//	{
//		j = 2 * PI_d * i / (double)N;
//		Z[i].x = X(j, h, omega, t) + 2;
//		Z[i].y = Y(j, h, omega, t);
//
//		Z[i + N].x = Phi(j, h, omega, t, properties.rho);
//		Z[i + N].y = 0; // Imaginary part is zero since Phi is real.
//
//		XprimeValues[i] = Xprime(j, h, omega, t) * 2.0 * PI_d / N;
//		YprimeValues[i] = Yprime(j, h, omega, t) * 2.0 * PI_d / N;
//		XppValues[i] = Xpp(j, h, omega, t) * 4.0 * PI_d * PI_d / (N * N);
//		YppValues[i] = Ypp(j, h, omega, t) * 4.0 * PI_d * PI_d / (N * N);
//		PhiPrimeValues[i] = TheoreticalPhiPrime(j, h, omega, t, properties.rho) * 2.0 * PI_d / N;
//	}
//
//
//	ZPhiDerivative<N> zPhiDerivative(properties);
//
//	cufftDoubleComplex* devZ;
//	cufftDoubleComplex* devZPhiPrime;
//	cufftDoubleComplex* devZpp;
//
//	cudaMalloc(&devZ, 2 * N * sizeof(cufftDoubleComplex));
//	cudaMalloc(&devZPhiPrime, 2 * N * sizeof(cufftDoubleComplex));
//	cudaMalloc(&devZpp, N * sizeof(cufftDoubleComplex));
//
//	cudaMemcpy(devZ, Z.data(), 2 * N * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
//
//	zPhiDerivative.exec(devZ, devZPhiPrime, devZpp);
//
//	cudaDeviceSynchronize();
//	// offload the results to host
//	std::array<cufftDoubleComplex, 2 * N> ZPhiPrime;
//	std::array<cufftDoubleComplex, N> Zpp;
//
//	cudaMemcpy(ZPhiPrime.data(), devZPhiPrime, 2 * N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
//	cudaMemcpy(Zpp.data(), devZpp, N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
//	plt::figure();
//	plt::plot(XprimeValues, { {"label", "Expected X Derivative"} });
//
//
//	for (int i = 0; i < N; i++)
//	{
//		XPrimeCalculatedValues[i] = ZPhiPrime[i].x;
//		YPrimeCalculatedValues[i] = ZPhiPrime[i].y;
//
//		XppCalculatedValues[i] = Zpp[i].x;
//		YppCalculatedValues[i] = Zpp[i].y;
//
//		PhiPrimeCalculatedValues[i] = ZPhiPrime[i + N].x;
//
//		EXPECT_NEAR(ZPhiPrime[i].x, XprimeValues[i], 1e-14) << "Xprime mismatch at index " << i;
//		EXPECT_NEAR(ZPhiPrime[i].y, YprimeValues[i], 1e-14) << "Yprime mismatch at index " << i;
//
//		EXPECT_NEAR(Zpp[i].x, XppValues[i], 1e-14) << "Xpp mismatch at index " << i;
//		EXPECT_NEAR(Zpp[i].y, YppValues[i], 1e-14) << "Ypp mismatch at index " << i;
//
//		EXPECT_NEAR(ZPhiPrime[i + N].x, PhiPrimeValues[i], 1e-14) << "PhiPrime mismatch at index " << i;
//	}
//
//	plt::plot(XPrimeCalculatedValues, { {"label", "Calculated X Prime"} });
//
//	plt::plot(YprimeValues, { {"label", "Expected Y Derivative"} });
//	plt::plot(YPrimeCalculatedValues, { {"label", "Calculated Y Prime"} });
//	plt::xlabel("Index");
//	plt::ylabel("Value");
//	plt::legend();
//	plt::figure();
//
//	plt::plot(XppValues, { {"label", "Expected X Second Derivative"} });
//	plt::plot(XppCalculatedValues, { {"label", "Calculated X Second Derivative"} });
//
//	plt::plot(YppValues, { {"label", "Expected Y Second Derivative"} });
//	plt::plot(YppCalculatedValues, { {"label", "Calculated Y Second Derivative"} });
//	plt::legend();
//	plt::xlabel("Index");
//	plt::ylabel("Value");
//
//	plt::figure();
//	plt::plot(PhiPrimeValues, { {"label", "Expected Phi Prime"} });
//	plt::plot(PhiPrimeCalculatedValues, { {"label", "Calculated Phi Prime"} });
//
//	plt::legend();
//	//plt::show(false);
//}

void createRhs(double* rhs, double* Y, double* VXlower, double* VYlower, double* VXupper, double* VYupper, double rho, int N) 
{
	for(int i = 0; i < N; i++) 
	{
		rhs[i] = -(1 + rho) * Y[i] + 0.5 * (VXlower[i] * VXlower[i] + VYlower[i] * VYlower[i]) + 0.5 * rho * (VXupper[i] * VXupper[i] + VYupper[i] * VYupper[i]) - rho * (VXlower[i] * VXupper[i] + VYlower[i] * VYupper[i]);
	}
}

/// <summary>
/// Tests: compute_rhs_phi_expression
/// </summary>
/// <param name=""></param>
/// <param name=""></param>
TEST(Kernels, RhsPhi) 
{
	const int N = 32;
	const int threads = 256;
	const int blocks = (N + threads - 1) / threads;

	std::vector<double> rhs(N, 0.0);

	std::vector<double> Yvect(N, 0.0);

	std::vector<cufftDoubleComplex> ZHost(N, { 0.0, 0.0 });
	
	std::vector<cufftDoubleComplex> VLowerHost(N, { 0.0, 0.0 });
	std::vector<cufftDoubleComplex> VUpperHost(N, { 0.0, 0.0 });

	std::vector<std_complex> RhsCalculated(N);

	std::vector<double> VXlower(N, 0.0);
	std::vector<double> VYlower(N, 0.0);

	std::vector<double> VXupper(N, 0.0);
	std::vector<double> VYupper(N, 0.0);

	double omega = 10.0;
	double h = 0.5;
	double t = 0.1;

	for(int i = 0; i < N; i++) 
	{
		double j = 2 * PI_d * i / (double)N;
		Yvect[i] = Y(i, h, omega, t); // Y function values
		VXlower[i] = 2 * PI_d / N * Xprime(j, h, omega, t);
		VYlower[i] = 2 * PI_d / N * Yprime(j, h, omega, t); // lower velocities

		ZHost[i].x = X(j, h, omega, t);
		ZHost[i].y = Yvect[i];

		VLowerHost[i].x = VXlower[i];
		VLowerHost[i].y = VYlower[i]; // lower velocities as complex numbers

		VUpperHost[i].x = 0; // upper velocities are the same as lower velocities in this test
		VUpperHost[i].y = 0; // upper velocities are zero in this test
	}

	createRhs(rhs.data(), Yvect.data(), VXlower.data(), VYlower.data(), VXupper.data(), VYupper.data(), 0.0, N);

	// create the items in the device memory
	std_complex* Z;
	std_complex* V1;
	std_complex* V2;
	std_complex* result;

	cudaMalloc(&Z, N * sizeof(std_complex));
	cudaMalloc(&V1, N * sizeof(std_complex));
	cudaMalloc(&V2, N * sizeof(std_complex));
	cudaMalloc(&result, N * sizeof(std_complex));

	cudaMemcpy(Z, ZHost.data(), N * sizeof(std_complex), cudaMemcpyHostToDevice);
	cudaMemcpy(V1, VLowerHost.data(), N * sizeof(std_complex), cudaMemcpyHostToDevice);
	cudaMemcpy(V2, VUpperHost.data(), N * sizeof(std_complex), cudaMemcpyHostToDevice);

	compute_rhs_phi_expression << <blocks, threads >> > (Z, V1, V2, result, 0.0, N);

	cudaDeviceSynchronize();

	cudaMemcpy(RhsCalculated.data(), result, N * sizeof(std_complex), cudaMemcpyDeviceToHost);

	for(int i = 0; i < N; i++) 
	{
		EXPECT_NEAR(RhsCalculated[i].real(), rhs[i], 1e-14) << "Rhs mismatch at index " << i;
		EXPECT_NEAR(RhsCalculated[i].imag(), 0.0, 1e-14) << "Rhs imaginary part mismatch at index " << i;
	}
}