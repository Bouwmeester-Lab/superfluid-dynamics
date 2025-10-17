#pragma once
#ifndef UTILITIES_H
#define UTILITIES_H

#include <stdio.h>
#include <cufft.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cusolverDn.h>
#include <iostream>
#include "cuDoubleComplexOperators.cuh"
#include <cuda/std/complex>
#include "PrecisionMath.cuh"



__global__ void first_derivative_multiplication(
    const cufftDoubleComplex* a,
    cufftDoubleComplex* result,
    const int n
);
__global__ void set_mode_to_imaginary(cufftDoubleComplex* a, double img, int n);
/// <summary>
/// Substracts two vectors out = Re(a) - b + iIm(a) and stores the result in out.
/// </summary>
/// <param name="a">Complex vector</param>
/// <param name="b">Real part to substract from a.</param>
/// <param name="out"></param>
/// <param name="n"></param>
__global__ void vector_subtract_complex_real(const cufftDoubleComplex* a, const double* b, cufftDoubleComplex* out, int n);
/// <summary>
/// Element wise addition. It will add b to the real part of a for every element in a.
/// </summary>
/// <param name="a"></param>
/// <param name="b"></param>
/// <param name="out"></param>
/// <param name="n"></param>
__global__ void vector_scalar_add_complex_real(const cufftDoubleComplex* a, const double b, cufftDoubleComplex* out, int n, int start);
__device__ cufftDoubleComplex cotangent_complex(cufftDoubleComplex a);
__device__ cuDoubleComplex cotangent_green_function(cuDoubleComplex Zk, cuDoubleComplex Zj);
__global__ void cotangent_complex(const cufftDoubleComplex* a, cufftDoubleComplex* out, int n);
__device__ void cos(cufftDoubleComplex z, cufftDoubleComplex& out);
__device__ void sin(cufftDoubleComplex z, cufftDoubleComplex& zout);
__device__ cufftDoubleComplex fromReal(double a);
__global__ void real_to_complex(const double* x, cuDoubleComplex* x_c, int N);
__global__ void complex_to_real(const cuDoubleComplex* x_c, double* x, int N);
__device__ inline cufftDoubleComplex cMulScalar(double a, cufftDoubleComplex z);

__global__ void conjugate_vector(cuDoubleComplex* x, cuDoubleComplex* z, int N);

__global__ void add_k_vectors(cufftDoubleComplex* k1, cufftDoubleComplex* k2, cufftDoubleComplex* k3, cufftDoubleComplex* k4, cufftDoubleComplex* result, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        result[i].x = k1[i].x + 2.0 * k2[i].x + 2.0 * k3[i].x + k4[i].x;
        result[i].y = k1[i].y + 2.0 * k2[i].y + 2.0 * k3[i].y + k4[i].y;
    }
}
__global__ void add_k_vectors(std_complex* k1, std_complex* k2, std_complex* k3, std_complex* k4, std_complex* result, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        result[i] = k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i];
    }
}


void checkCuda(cudaError_t result);
void checkCusolver(cusolverStatus_t status);



/// <summary>
/// Multiplies the coefficients of two complex vectors element-wise
/// </summary>
/// <param name="a"></param>
/// <param name="b"></param>
/// <param name="result"></param>
/// <param name="n"></param>
/// <returns></returns>
__global__ void first_derivative_multiplication(
    const cufftDoubleComplex* a,
    cufftDoubleComplex* result,
    const int n,
	const int batchSize = 1
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= n * batchSize) return; // out of bounds

	int  i = tid % n; // index within the vector
	//printf("i: %d, tid: %d\n", i, tid);
	//printf("i: %d coeff: (%f, %f)\n", i, a[tid].x, a[tid].y);
    double x, y;
    if (i < n / 2) 
    {
        x = a[tid].x; // it's important to do this copy since if you don't you will be using the new value in the expression instead of the old one.
		y = a[tid].y;
        result[tid].x = - i * y / static_cast<double>(n);
		result[tid].y = i * x / static_cast<double>(n);
    }
    else if (i == n / 2) 
    {
        x = a[tid].x;
        result[tid].x = -PI_d * i * a[tid].y / static_cast<double>(n);
        result[tid].y = -PI_d * i * x / static_cast<double>(n); // -PI_d * a[i].x / n; // we want to treat the Nyquist frequency as exp(i*pi*j) which means
        // that the inverse fft of the fft of exp(i*pi*j) should give i pi * exp(i * pi *j). This happens when the coeff[n/2] = -pi.
		// Usually this coefficient should be -pi *n but cuFFT will NOT normalize by n, so when we do normalize manually by dividing by n, we get -pi.
    }
    else if (i == n /2 +1) 
    {
        x = a[tid].x;
		y = a[tid].y;
        // this is the Nyquist frequency, we want to set it to zero since we don't want to have any imaginary part in the result
        result[tid].x = 0;// -PI_d * i * y / static_cast<double>(n);
        result[tid].y = 0; //-PI_d * i * x / static_cast<double>(n);// PI_d* x / n; // this is the same as setting the imaginary part to zero
	}
    else if (i < n) {
        x = a[tid].x;
        y = a[tid].y;
		result[tid].x = -(i - n) * y / static_cast<double>(n); // - because you multiply by i (imaginary i) => z = x+iy => iz = ix - y
		result[tid].y = (i - n) * x / static_cast<double>(n); // https://math.mit.edu/~stevenj/fft-deriv.pdf
    }
}

__global__ void multiply_element_wise(const cufftDoubleComplex*a, cufftDoubleComplex* b, cufftDoubleComplex* result, int n) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        result[i] = cuCmul(a[i], b[i]);
    }
}

/// <summary>
/// Filters the coefficients using a tanh filter. This is used to remove high frequency noise from the signal. It assumes that the coefficients are in the order of the frequencies of [0, N/2, -N/2, ..., -1].
/// </summary>
/// <param name="k"></param>
/// <param name="N"></param>
/// <param name="eps"></param>
/// <param name="d"></param>
/// <returns></returns>
__device__ __host__ inline double filterTanh(int k, int N, double eps, double d) 
{
    return 0.5 + 0.5 * tanh((2.0 * PI_d / N * k - eps) / d);
}

/// <summary>
/// Calculates the coefficients needed to run an ifft and get the second derivative from the original function: d2/dt f(t)
/// </summary>
/// <param name="coeffsFft"></param>
/// <param name="result"></param>
/// <param name="n"></param>
/// <returns></returns>
__global__ void second_derivative_fft(const cufftDoubleComplex* coeffsFft, cufftDoubleComplex* result, const int n, const int batchSize = 1) 
{

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= n * batchSize) return; // out of bounds

	int i = tid % n; // index within the vector
    if (i < n / 2) 
    {
        result[tid].x = - i * i * coeffsFft[tid].x / (n);
		result[tid].y = - i * i * coeffsFft[tid].y / (n);
    }
    else if (i == n / 2) 
    {
        result[tid].x = - i * i * coeffsFft[tid].x / (n); // real part, this is the same idea as in the first derivative. We want the second derivative of exp(i*pi*j) to be -pi^2 * exp(i*pi*j), this will happen only if this coefficient is this value
		result[tid].y = -i * i * coeffsFft[tid].y / (n);// setting this to zero seems fine? although I wonder if it's better to leave the same value as the theoretical value: N/2 * N/2 * coeffsFft[i].y / N ?
    }
    else if(i < n)
    {
		result[tid].x = -(i - n) * (i - n) * coeffsFft[tid].x / (n); //https://math.mit.edu/~stevenj/fft-deriv.pdf
		result[tid].y = -(i - n) * (i - n) * coeffsFft[tid].y / (n);
	}
}

__global__ void force_real_only(cufftDoubleComplex* a, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        a[i].y = 0; // set the imaginary part to 0
    }
}

__global__ void set_mode_to_imaginary(cufftDoubleComplex* a, double img, int n)
{
	a[n].x = 0; // set the real part of the Nyquist frequency to 0
	a[n].y = img; // set the imaginary part of the Nyquist frequency to 1
}

__global__ void vector_subtract_complex_real(const std_complex* a, const double* b, std_complex* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] - b[i];
    }
}
/// <summary>
/// Performs batched subtraction of a real single vector (size N) from a complex vector on the GPU (NxbatchSize), storing the result in an output array.
/// </summary>
/// <param name="a">Pointer to the input array of complex numbers (batched vectors).</param>
/// <param name="b">Pointer to the input array of real numbers (single vector).</param>
/// <param name="out">Pointer to the output array where the results are stored.</param>
/// <param name="n">The length of each vector.</param>
/// <param name="batchSize">The number of vectors in the batch.</param>
/// <returns>This is a CUDA kernel function and does not return a value. The results are written to the output array 'out'.</returns>
__global__ void batched_vector_subtract_singletime_complex_real(const std_complex* a, const double* b, std_complex* out, int n, size_t batchSize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= n * batchSize) return; // out of bounds
	int i = tid % n; // index within the vector

    if (i < n) {
        out[tid] = a[tid] - b[i];
    }
}
/// <summary>
/// Adds a real scalar value to the real part of each element in a complex vector using CUDA parallelization.
/// </summary>
/// <param name="a">Pointer to the input array of complex numbers.</param>
/// <param name="b">The real scalar value to add to the real part of each complex number.</param>
/// <param name="out">Pointer to the output array where the results are stored.</param>
/// <param name="n">The number of elements to process.</param>
/// <param name="start">The starting index in the arrays for processing.</param>
/// <returns>This function does not return a value; results are written to the output array.</returns>
__global__ void vector_scalar_add_complex_real(const std_complex* a, const double b, std_complex* out, int n, int start)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
		out[start + i] = a[start + i] + b; // adds the scalar to the real part of the complex number
    }
}

__global__ void vector_mutiply_scalar(const std_complex* a, const double b, std_complex* out, int n, int start)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
		out[start + i] = a[start + i] * b; // multiplies the complex number by the scalar
    }
}
/// <summary>
/// Calculates the cotangent using cot(z) = cos(z)/sin(z).
/// </summary>
/// <param name="a"></param>
/// <param name="out"></param>
/// <param name="n"></param>
__global__ void cotangent_complex(const cufftDoubleComplex* a, cufftDoubleComplex* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
		out[i] = cotangent_complex(a[i]); // using the device function to calculate the cotangent
  //      cufftDoubleComplex cs;
  //      cufftDoubleComplex cc;

  //      cos(a[i], cc);
  //      sin(a[i], cs);

		//auto z = cuCdiv(cc, cs); // https://dlmf.nist.gov/4.21#E40 maybe useful for the future, I've tied to use this but it didn't provide the same simply dividing cos/sin

  //      out[i].x = z.x;
  //      out[i].y = z.y;
    }
}

__global__ void real_to_complex(const double* x, cuDoubleComplex* x_c, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N)
        x_c[idx] = make_cuDoubleComplex(x[idx], 0.0);
}

__global__ void real_to_complex(const double* x, std_complex* x_c, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N)
        x_c[idx] = std_complex(x[idx], 0.0);
}

__global__ void complex_to_real(const cuDoubleComplex* x_c, double* x, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N)
        x[idx] = x_c[idx].x; // only copy the real part
}

__global__ void complex_to_real(const std_complex* x_c, double* x, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N)
        x[idx] = x_c[idx].real(); // only copy the real part
}

__device__ __forceinline__ std_complex cot(std_complex z) 
{
	return 1.0 / tan(z); // using the standard complex library to calculate the cotangent
}

__device__ inline cufftDoubleComplex cotangent_complex(cufftDoubleComplex a)
{
	auto a_std = reinterpret_cast<cuda::std::complex<double>*>(&a);
	// Using the standard complex library to calculate the cotangent
	auto t = 1.0 / tan(*a_std);
	return make_cuDoubleComplex(t.real(), t.imag());

    /*cufftDoubleComplex cs;
    cufftDoubleComplex cc;

    cos(a, cc);
    sin(a, cs);

    return cuCdiv(cc, cs);*/
    //return  make_cuDoubleComplex(sin(2 * a.x)/ (cosh(2 * a.y) - cos(2 * a.x)), -sinh(2* a.y)/ (cosh(2 * a.y) - cos(2 * a.x))); // https://dlmf.nist.gov/4.21#E40
	//double coeff = 1.0 / ; // this is the normalization factor

	//return make_cuDoubleComplex(top.x * coeff, top.y * coeff);
}

__device__ cuDoubleComplex multiply_by_i(cuDoubleComplex z)
{
    return make_cuDoubleComplex(-z.y, z.x); // multiply by i is equivalent to rotating the complex number by 90 degrees counter-clockwise
}

__device__ std_complex multiply_by_i(std_complex z)
{
    return std_complex(-z.imag(), z.real()); // multiply by i is equivalent to rotating the complex number by 90 degrees counter-clockwise
}

__device__ std_complex cotangent_green_function(std_complex Zk, std_complex Zj)
{
    return cot(0.5 * (Zk - Zj));

 //   std_complex cotZk = cot(0.5 * Zk);
	//std_complex cotZj = cot(0.5 * Zj);

	//// Using the cotangent subtraction formula: cot(Zk - Zj) = (cot(Zk) * cot(Zj) + 1) / (cot(Zj) - cot(Zk))
	//std_complex top = cotZk * cotZj + 1.0;
	//std_complex invBottom = PrecisionMath::fastPreciseInvSub(cotZj, cotZk); // using the precise inverse substraction using double double precision to avoid numerical issues
	//return top * invBottom; // this is the cotangent green function
}

__device__ void cos(cufftDoubleComplex z, cufftDoubleComplex& out)
{
    out.x = cos(z.x) * cosh(z.y);
    out.y = -sin(z.x) * sinh(z.y);
}

__device__ void sin(cufftDoubleComplex z, cufftDoubleComplex& zout) {
    zout.x = sin(z.x) * cosh(z.y);
    zout.y = cos(z.x) * sinh(z.y);
}

__device__ cufftDoubleComplex fromReal(double a)
{
    cufftDoubleComplex out;
    out.x = a;
    out.y = 0.0;
    return out;
}

__device__ inline
cufftDoubleComplex cMulScalar(double a, cufftDoubleComplex z)
{
    cufftDoubleComplex out(z);

    out.x = a * out.x;
    out.y = a * out.y;

    return out;
}

__global__ void conjugate_vector(cuDoubleComplex* x, cuDoubleComplex* z, int N) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
		z[i].x = x[i].x; // copy the real part
        z[i].y = -x[i].y; // conjugate
    }
}

__global__ void conjugate_vector(std_complex* x, std_complex* z, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
		z[i] = cuda::std::conj(x[i]); // use the standard library to conjugate the complex number
    }
}


void checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(result) << std::endl;
        exit(EXIT_FAILURE);
    }
}

inline void checkCuda(cudaError_t result,
    const char* func,
    const char* file,
    int line)
{
    if (result != cudaSuccess) {
        std::cerr << "CUDA Error at " << file << ":" << line
            << " (" << func << "): "
            << cudaGetErrorString(result) << std::endl;
        throw std::runtime_error(cudaGetErrorString(result));
    }
}
#define CHECK_CUDA(val) checkCuda((val), #val, __FILE__, __LINE__)

void checkCusolver(cusolverStatus_t status) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "cuSolver Error" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void checkCublas(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS Error: " << status << std::endl;
		exit(EXIT_FAILURE);
    }
}


#endif