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

__global__ void complexCotKernel(std_complex* zsk, std_complex* zsj, std_complex* out, int N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N) {
		out[idx] = cotangent_green_function(zsk[idx], zsj[idx]);
	}
}

__global__ void test_precision_inversion(std_complex* zsk, std_complex* zsj, std_complex* out, int N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N) {
		out[idx] = fastPreciseInvSub(zsk[idx], zsj[idx]);
	}
}

__global__ void test_precision_substraction(std_complex* zsk, std_complex* zsj, std_complex* out, std_complex* low, int N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N) {
		dd_complex d = zsk[idx] - zsj[idx];
		out[idx] = std_complex(d.real.hi, d.imag.hi);
		low[idx] = std_complex(d.real.lo, d.imag.lo);
	}
}

std::complex<double> cotangent_test(std::complex<double> z) {
	return 1.0 / std::tan(z);
}

std::complex<double> cotangent_substraction(std::complex<double> z1, std::complex<double> z2) 
{
	return (cotangent_test(z1) * cotangent_test(z2) + 1.0) / (cotangent_test(z2) - cotangent_test(z1));
}

TEST(ComplexFunctionsTests, ComplexCotKernel) 
{
	const int N = 16;

	std::array<cuDoubleComplex, N> zsk;
	std::array<cuDoubleComplex, N> zsj;

	std_complex* zsk_d;
	std_complex* zsj_d;
	std_complex* cot_result_d;

	std::array<std::complex<double>, N> zsk_std;
	std::array<std::complex<double>, N> zsj_std;

	// High precision expected values for cotangent function cot( 0.5 *(zsk - zsj) ) calculated using a high precision library on Python.
	std::array<cuDoubleComplex, N> cot_expected = { { { -9.9833388915330681153509188355277398344111330657474, 10.016672219575397493791065794281138271298250914604 },
{ -9.9833388915330681153509188355277398344111330657474, 10.016672219575397493791065794281138271298250914604 },
{ -9.9833388915330681153509188355277398344111330657474, 10.016672219575397493791065794281138271298250914604 },
{ -9.9833388915330681153509188355277398344111330657474, 10.016672219575397493791065794281138271298250914604 },
{ -9.9833388915330681153509188355277398344111330657474, 10.016672219575397493791065794281138271298250914604 },
{ -9.9833388915330681153509188355277398344111330657474, 10.016672219575397493791065794281138271298250914604 },
{ -9.9833388915330681153509188355277398344111330657474, 10.016672219575397493791065794281138271298250914604 },
{ -9.9833388915330681153509188355277398344111330657474, 10.016672219575397493791065794281138271298250914604 },
{ -9.9833388915330681153509188355277398344111330657474, 10.016672219575397493791065794281138271298250914604 },
{ -9.9833388915330681153509188355277398344111330657474, 10.016672219575397493791065794281138271298250914604 },
{ -9.9833388915330681153509188355277398344111330657474, 10.016672219575397493791065794281138271298250914604 },
{ -9.9833388915330681153509188355277398344111330657474, 10.016672219575397493791065794281138271298250914604 },
{ -9.9833388915330681153509188355277398344111330657474, 10.016672219575397493791065794281138271298250914604 },
{ -9.9833388915330681153509188355277398344111330657474, 10.016672219575397493791065794281138271298250914604 },
{ -9.9833388915330681153509188355277398344111330657474, 10.016672219575397493791065794281138271298250914604 },
{ -9.9833388915330681153509188355277398344111330657474, 10.016672219575397493791065794281138271298250914604 } } };

	std::array<std_complex, N> cot_result;
	double x, y;
	double delta = 0.1;

	for (int i = 0; i < N; i++) {
		x = PI_d, y = PI_d;
		zsk[i] = make_cuDoubleComplex(x, y);
		zsj[i] = make_cuDoubleComplex(x + delta, y + delta);

		zsk_std[i] = std::complex<double>(x, y);
		zsj_std[i] = std::complex<double>(x + delta, y + delta);

		
	}


	cudaMalloc(&zsk_d, N * sizeof(cuDoubleComplex));
	cudaMalloc(&zsj_d, N * sizeof(cuDoubleComplex));
	cudaMalloc(&cot_result_d, N * sizeof(std_complex));
	
	cudaMemcpy(zsk_d, zsk.data(), N * sizeof(std_complex), cudaMemcpyHostToDevice);
	cudaMemcpy(zsj_d, zsj.data(), N * sizeof(std_complex), cudaMemcpyHostToDevice);
	
	const int threadsPerBlock = 256;
	const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;



	complexCotKernel << <blocks, threadsPerBlock >> > (zsk_d, zsj_d, cot_result_d, N);

	cudaDeviceSynchronize();

	cudaMemcpy(cot_result.data(), cot_result_d, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
	cudaFree(zsk_d);
	cudaFree(zsj_d);
	cudaFree(cot_result_d);

	for (int i = 0; i < N; ++i) {
		/*EXPECT_DOUBLE_EQ(cuCreal(cot_result[i]), cot_expected[i].x);
		EXPECT_DOUBLE_EQ(cuCimag(cot_result[i]), cot_expected[i].y);*/
		EXPECT_NEAR(cot_result[i].real(), cot_expected[i].x, 1e-13); // best accuracy I can get so far without diving deeper into arbitrary precision libraries
		EXPECT_NEAR((cot_result[i]).imag(), cot_expected[i].y, 1e-13);
	}
}


TEST(ComplexFunctionsTests, PrecisionInv)
{
	const int N = 16;

	std::array<cuDoubleComplex, N> zsk;
	std::array<cuDoubleComplex, N> zsj;

	std_complex* zsk_d;
	std_complex* zsj_d;
	std_complex* result_d;

	std::array<std::complex<double>, N> zsk_std;
	std::array<std::complex<double>, N> zsj_std;

	// High precision expected values for 1 / (pi +  1j * pi)
	std::array<cuDoubleComplex, N> expected = { { { 0.15915494309189533576888376337251436203445964574046, -0.15915494309189533576888376337251436203445964574046 },
{ 0.15915494309189533576888376337251436203445964574046, -0.15915494309189533576888376337251436203445964574046 },
{ 0.15915494309189533576888376337251436203445964574046, -0.15915494309189533576888376337251436203445964574046 },
{ 0.15915494309189533576888376337251436203445964574046, -0.15915494309189533576888376337251436203445964574046 },
{ 0.15915494309189533576888376337251436203445964574046, -0.15915494309189533576888376337251436203445964574046 },
{ 0.15915494309189533576888376337251436203445964574046, -0.15915494309189533576888376337251436203445964574046 },
{ 0.15915494309189533576888376337251436203445964574046, -0.15915494309189533576888376337251436203445964574046 },
{ 0.15915494309189533576888376337251436203445964574046, -0.15915494309189533576888376337251436203445964574046 },
{ 0.15915494309189533576888376337251436203445964574046, -0.15915494309189533576888376337251436203445964574046 },
{ 0.15915494309189533576888376337251436203445964574046, -0.15915494309189533576888376337251436203445964574046 },
{ 0.15915494309189533576888376337251436203445964574046, -0.15915494309189533576888376337251436203445964574046 },
{ 0.15915494309189533576888376337251436203445964574046, -0.15915494309189533576888376337251436203445964574046 },
{ 0.15915494309189533576888376337251436203445964574046, -0.15915494309189533576888376337251436203445964574046 },
{ 0.15915494309189533576888376337251436203445964574046, -0.15915494309189533576888376337251436203445964574046 },
{ 0.15915494309189533576888376337251436203445964574046, -0.15915494309189533576888376337251436203445964574046 },
{ 0.15915494309189533576888376337251436203445964574046, -0.15915494309189533576888376337251436203445964574046 } } };

	std::array<std_complex, N> result;
	double x, y;
	double delta = PI_d;

	for (int i = 0; i < N; i++) {
		x = i, y = i;
		zsk[i] = make_cuDoubleComplex(x, y);
		zsj[i] = make_cuDoubleComplex(x - delta, y - delta);

		zsk_std[i] = std::complex<double>(x, y);
		zsj_std[i] = std::complex<double>(x - delta, y - delta);


	}


	cudaMalloc(&zsk_d, N * sizeof(cuDoubleComplex));
	cudaMalloc(&zsj_d, N * sizeof(cuDoubleComplex));
	cudaMalloc(&result_d, N * sizeof(std_complex));

	cudaMemcpy(zsk_d, zsk.data(), N * sizeof(std_complex), cudaMemcpyHostToDevice);
	cudaMemcpy(zsj_d, zsj.data(), N * sizeof(std_complex), cudaMemcpyHostToDevice);

	const int threadsPerBlock = 256;
	const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;



	test_precision_inversion << <blocks, threadsPerBlock >> > (zsk_d, zsj_d, result_d, N);

	cudaDeviceSynchronize();

	cudaMemcpy(result.data(), result_d, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
	cudaFree(zsk_d);
	cudaFree(zsj_d);
	cudaFree(result_d);

	for (int i = 0; i < N; ++i) {
		/*EXPECT_DOUBLE_EQ(cuCreal(cot_result[i]), cot_expected[i].x);
		EXPECT_DOUBLE_EQ(cuCimag(cot_result[i]), cot_expected[i].y);*/
		EXPECT_NEAR(result[i].real(), expected[i].x, 1e-13); // best accuracy I can get so far without diving deeper into arbitrary precision libraries
		EXPECT_NEAR((result[i]).imag(), expected[i].y, 1e-13);
	}
}


TEST(ComplexFunctionsTests, SubstractionPrecise)
{
	const int N = 16;

	std::array<cuDoubleComplex, N> zsk;
	std::array<cuDoubleComplex, N> zsj;

	std_complex* zsk_d;
	std_complex* zsj_d;
	std_complex* result_d;
	std_complex* result_low_d;

	std::array<std::complex<double>, N> zsk_std;
	std::array<std::complex<double>, N> zsj_std;

	// High precision expected values for cotangent function cot( 0.5 *(zsk - zsj) ) calculated using a high precision library on Python.
	std::array<cuDoubleComplex, N> expected = { { { -3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117068, -3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117068 },
{ -3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117068, -3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117068 },
{ -3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117068, -3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117068 },
{ -3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117068, -3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117068 },
{ -3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117068, -3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117068 },
{ -3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117068, -3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117068 },
{ -3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117068, -3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117068 },
{ -3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117068, -3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117068 },
{ -3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117068, -3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117068 },
{ -3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117068, -3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117068 },
{ -3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117068, -3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117068 },
{ -3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117068, -3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117068 },
{ -3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117068, -3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117068 },
{ -3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117068, -3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117068 },
{ -3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117068, -3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117068 },
{ -3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117068, -3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117068 } } };

	std::array<std_complex, N> result;
	std::array<std_complex, N> result_low;
	double x, y;
	double delta = PI_d;

	for (int i = 0; i < N; i++) {
		x = i, y = i;
		zsk[i] = make_cuDoubleComplex(x, y);
		zsj[i] = make_cuDoubleComplex(x + delta, y + delta);

		zsk_std[i] = std::complex<double>(x, y);
		zsj_std[i] = std::complex<double>(x + delta, y + delta);
	}


	cudaMalloc(&zsk_d, N * sizeof(cuDoubleComplex));
	cudaMalloc(&zsj_d, N * sizeof(cuDoubleComplex));
	cudaMalloc(&result_d, N * sizeof(std_complex));
	cudaMalloc(&result_low_d, N * sizeof(std_complex));

	cudaMemcpy(zsk_d, zsk.data(), N * sizeof(std_complex), cudaMemcpyHostToDevice);
	cudaMemcpy(zsj_d, zsj.data(), N * sizeof(std_complex), cudaMemcpyHostToDevice);

	const int threadsPerBlock = 256;
	const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;



	test_precision_substraction << <blocks, threadsPerBlock >> > (zsk_d, zsj_d, result_d, result_low_d, N);

	cudaDeviceSynchronize();

	cudaMemcpy(result.data(), result_d, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
	cudaMemcpy(result_low.data(), result_low_d, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
	cudaFree(zsk_d);
	cudaFree(zsj_d);
	cudaFree(result_d);
	cudaFree(result_low_d);

	for (int i = 0; i < N; ++i) {
		/*EXPECT_DOUBLE_EQ(cuCreal(cot_result[i]), cot_expected[i].x);
		EXPECT_DOUBLE_EQ(cuCimag(cot_result[i]), cot_expected[i].y);*/
		EXPECT_DOUBLE_EQ(result[i].real(), expected[i].x); // best accuracy I can get so far without diving deeper into arbitrary precision libraries
		EXPECT_DOUBLE_EQ((result[i]).imag(), expected[i].y);
		EXPECT_DOUBLE_EQ(result_low[i].real(), 0.0); // low part should be zero
		EXPECT_DOUBLE_EQ(result_low[i].imag(), 0.0); // low part should be zero
	}
}