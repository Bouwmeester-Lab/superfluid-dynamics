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

/// <summary>
/// Computes the RHS of Phi on the GPU using the expression: -(1+rho) * Im(Z) + 0.5 * abs(V1)^2 + 0.5 * rho * abs(V2)^2. - rho * V1 dot V2
/// It assumes Kappa = 0, and there's no surface tension.
/// The imag part of result is 0. Since this is purely real, but it's useful to stick to complex for consistency.
/// </summary>
/// <param name="Z"></param>
/// <param name="V"></param>
/// <param name="result"></param>
/// <param name="alpha"></param>
/// <param name="N"></param>
/// <returns></returns>
__global__ void compute_rhs_phi_expression(
    const cuDoubleComplex* Z,
    const cuDoubleComplex* V1,
	const cuDoubleComplex* V2,
    cuDoubleComplex* result,
	double rho,
    int N
);

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
    const int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double x, y;
    if (i < n / 2) 
    {
        x = a[i].x; // it's important to do this copy since if you don't you will be using the new value in the expression instead of the old one.
		y = a[i].y;
        result[i].x = - i * y / static_cast<double>(n);
		result[i].y = i * x / static_cast<double>(n);
    }
    else if (i == n / 2) 
    {
        x = a[i].x;
        result[i].x = 0;
        result[i].y = - PI_d * x / n; // -PI_d * a[i].x / n; // we want to treat the Nyquist frequency as exp(i*pi*j) which means
        // that the inverse fft of the fft of exp(i*pi*j) should give i pi * exp(i * pi *j). This happens when the coeff[n/2] = -pi.
		// Usually this coefficient should be -pi *n but cuFFT will NOT normalize by n, so when we do normalize manually by dividing by n, we get -pi.
    }
    else if (i == n /2 +1) 
    {
        // this is the Nyquist frequency, we want to set it to zero since we don't want to have any imaginary part in the result
        result[i].x = 0;
        result[i].y = 0; // this is the same as setting the imaginary part to zero
	}
    else if (i < n) {
        x = a[i].x;
        y = a[i].y;
		result[i].x = (n - i) * y / static_cast<double>(n);
		result[i].y = (i- n) * x / static_cast<double>(n); // https://math.mit.edu/~stevenj/fft-deriv.pdf
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
__global__ void second_derivative_fft(const cufftDoubleComplex* coeffsFft, cufftDoubleComplex* result, const int n) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n / 2) 
    {
        result[i].x = - i * i * coeffsFft[i].x / n;
		result[i].y = - i * i * coeffsFft[i].y / n;
    }
    else if (i == n / 2) 
    {
        result[i].x = -i * i * coeffsFft[i].x / n; // real part, this is the same idea as in the first derivative. We want the second derivative of exp(i*pi*j) to be -pi^2 * exp(i*pi*j), this will happen only if this coefficient is this value
		result[i].y = -i * i * coeffsFft[i].y / n;// setting this to zero seems fine? although I wonder if it's better to leave the same value as the theoretical value: N/2 * N/2 * coeffsFft[i].y / N ?
    }
    else if(i < n)
    {
		result[i].x = -(i - n) * (i - n) * coeffsFft[i].x / n; //https://math.mit.edu/~stevenj/fft-deriv.pdf
		result[i].y = -(i - n) * (i - n) * coeffsFft[i].y / n;
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

__device__ cuDoubleComplex cotangent_series(cuda::std::complex<double>& a)
{
	auto __z = 1.0 / a - a / 3.0 + a * a * a / 45.0 - a * a * a * a * a / 945.0 + a * a * a * a * a * a * a / 4725.0 - 2.0 * cuda::std::pow(a, 9.0)/93555.0 - cuda::std::pow(a, 11.0) / 638512875.0; // this is the series expansion of cotangent
	return make_cuDoubleComplex(__z.real(), __z.imag());
}


__device__ std_complex cotangent_green_function(std_complex Zk, std_complex Zj, const double scale)
{
    return scale / cuda::std::tan(0.5 * (Zk - Zj));
}

__device__ cuDoubleComplex multiply_by_i(cuDoubleComplex z)
{
    return make_cuDoubleComplex(-z.y, z.x); // multiply by i is equivalent to rotating the complex number by 90 degrees counter-clockwise
}

__device__ std_complex multiply_by_i(std_complex z)
{
    return std_complex(-z.imag(), z.real()); // multiply by i is equivalent to rotating the complex number by 90 degrees counter-clockwise
}

__device__ std_complex cotangent_green_function(std_complex Zk, std_complex Zj, const cuda::std::complex<double> scale)
{
    std_complex eps = 0.5 * (Zk - Zj); // this is the difference between the two Z

    auto __z = scale / cuda::std::tan(eps);

    return __z;
}

__device__ std_complex cotangent_green_function(std_complex Zk, std_complex Zj)
{
    return cotangent_green_function(Zk, Zj, 1.0); // default scale is 1.0
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

__global__ void compute_rhs_phi_expression(const std_complex* Z, const std_complex* V1, const std_complex* V2, std_complex* result, double rho, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		// Calculate the right-hand side of the phi equation
        double Z_imag = Z[i].imag();
		double V1_abs2 = V1[i].real() * V1[i].real() + V1[i].imag() * V1[i].imag();
		double V2_abs2 = V2[i].real() * V2[i].real() + V2[i].imag() * V2[i].imag();
		double V1_dot_V2 = V1[1].real() * V2[i].real() + V1[i].imag() * V2[i].imag();
		result[i] = -(1 + rho) * Z_imag + 0.5 * V1_abs2 + 0.5 * rho * V2_abs2 - rho * V1_dot_V2;
	}
}

void checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(result) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void checkCusolver(cusolverStatus_t status) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "cuSolver Error" << std::endl;
        exit(EXIT_FAILURE);
    }
}


#endif