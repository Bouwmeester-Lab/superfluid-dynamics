#pragma once
#include <gtest/gtest.h>
#include "constants.cuh"
#include "utilities.cuh"
#include "array"
#include "complex"

__global__ void complexSinKernel(cuDoubleComplex* zs, cuDoubleComplex* out, int N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N) {
		sin(zs[idx], out[idx]);
	}
}

__global__ void complexCosKernel(cuDoubleComplex* zs, cuDoubleComplex* out, int N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N) {
		cos(zs[idx], out[idx]);
	}
}

TEST(ComplexFunctionsTests, ComplexSin) {
	const int N = 256;

	std::array<cuDoubleComplex, N> zs;
	std::array<std::complex<double>, N> zs_std;
	std::array<std::complex<double>, N> sin_expected;
	std::array<cuDoubleComplex, N> sin_result;
	double x, y;
	
	for(int i = 0; i < N; ++i) {
		x, y = 2 * PI_d / N * i;
		zs[i] = make_cuDoubleComplex(x, y);
		zs_std[i] = std::complex<double>(x, y);
		sin_expected[i] = std::sin(zs_std[i]);
	}

	cuDoubleComplex* zs_d;
	cudaMalloc(&zs_d, N * sizeof(cuDoubleComplex));
	cudaMemcpy(zs_d, zs.data(), N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

	const int threadsPerBlock = 256;
	const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

	complexSinKernel << <blocks, threadsPerBlock >> > (zs_d, zs_d, N);
	cudaDeviceSynchronize();

	cudaMemcpy(sin_result.data(), zs_d, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
	cudaFree(zs_d);

	for (int i = 0; i < N; ++i) {
		EXPECT_DOUBLE_EQ(cuCreal(sin_result[i]), sin_expected[i].real());
		EXPECT_DOUBLE_EQ(cuCimag(sin_result[i]), sin_expected[i].imag());
	}
}

TEST(ComplexFunctionsTests, ComplexCos) {
	// Similar implementation for complex cosine function
	// This can be implemented similarly to the complexSinKernel
	// and the test can be structured like the ComplexSin test.
	const int N = 256;

	std::array<cuDoubleComplex, N> zs;
	std::array<std::complex<double>, N> zs_std;
	std::array<std::complex<double>, N> cos_expected;
	std::array<cuDoubleComplex, N> cos_result;
	double x, y;

	for (int i = 0; i < N; ++i) {
		x, y = 2 * PI_d / N * i;
		zs[i] = make_cuDoubleComplex(x, y);
		zs_std[i] = std::complex<double>(x, y);
		cos_expected[i] = std::cos(zs_std[i]);
	}

	cuDoubleComplex* zs_d;
	cudaMalloc(&zs_d, N * sizeof(cuDoubleComplex));
	cudaMemcpy(zs_d, zs.data(), N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

	const int threadsPerBlock = 256;
	const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

	complexCosKernel << <blocks, threadsPerBlock >> > (zs_d, zs_d, N);
	cudaDeviceSynchronize();

	cudaMemcpy(cos_result.data(), zs_d, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
	cudaFree(zs_d);

	for (int i = 0; i < N; ++i) {
		EXPECT_DOUBLE_EQ(cuCreal(cos_result[i]), cos_expected[i].real());
		EXPECT_DOUBLE_EQ(cuCimag(cos_result[i]), cos_expected[i].imag());
	}
}

/// <summary>
/// Test for complex cotangent function.
/// It suffers from having a slightly less precision than the sine and cosine functions. EXPECT_DOUBLE_EQ is always failing, hence using EXPECT_NEAR.
/// </summary>
/// <param name=""></param>
/// <param name=""></param>
TEST(ComplexFunctionsTests, ComplexCotangent) 
{
	const int N = 256;

	std::array<cuDoubleComplex, N> zs;
	std::array<std::complex<double>, N> zs_std;
	std::array<std::complex<double>, N> cot_expected;
	std::array<cuDoubleComplex, N> cot_result;
	double x, y;

	for (int i = 0; i < N; ++i) {
		x, y = 2 * PI_d / (N+1) * (i + 1);
		zs[i] = make_cuDoubleComplex(x, y);
		zs_std[i] = std::complex<double>(x, y);
		cot_expected[i] = std::cos(zs_std[i]) / std::sin(zs_std[i]);
	}

	cuDoubleComplex* zs_d;
	cudaMalloc(&zs_d, N * sizeof(cuDoubleComplex));
	cudaMemcpy(zs_d, zs.data(), N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

	const int threadsPerBlock = 256;
	const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

	cotangent_complex << <blocks, threadsPerBlock >> > (zs_d, zs_d, N);
	cudaDeviceSynchronize();

	cudaMemcpy(cot_result.data(), zs_d, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
	cudaFree(zs_d);

	for (int i = 0; i < N; ++i) {
		EXPECT_NEAR(cuCreal(cot_result[i]), cot_expected[i].real(), 1e-15);
		EXPECT_NEAR(cuCimag(cot_result[i]), cot_expected[i].imag(), 1e-15);
	}
}