#pragma once
#ifndef UTILITIES_H
#define UTILITIES_H

#include <stdio.h>
#include <cufft.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cusolverDn.h>
#include <iostream>

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
        result[i].x = 0;// -PI_d * a[i].y / n;
        result[i].y = -PI_d * a[i].x / static_cast<double>(n); // -PI_d * a[i].x / n; // we want to treat the Nyquist frequency as exp(i*pi*j) which means
        // that the inverse fft of the fft of exp(i*pi*j) should give i pi * exp(i * pi *j). This happens when the coeff[n/2] = -pi.
		// Usually this coefficient should be -pi *n but cuFFT will NOT normalize by n, so when we do normalize manually by dividing by n, we get -pi.
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
        result[i].x = PI_d * PI_d * result[i].y / n; // real part, this is the same idea as in the first derivative. We want the second derivative of exp(i*pi*j) to be -pi^2 * exp(i*pi*j), this will happen only if this coefficient is this value
		result[i].y = 0;// setting this to zero seems fine? although I wonder if it's better to leave the same value as the theoretical value: N/2 * N/2 * coeffsFft[i].y / N ?
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

__global__ void vector_subtract_complex_real(const cufftDoubleComplex* a, const double* b, cufftDoubleComplex* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i].x = a[i].x - b[i]; // -b[i]; // modifies only the real part.
		out[i].y = a[i].y; // keeps the imaginary part unchanged.
    }
}

__global__ void vector_scalar_add_complex_real(const cufftDoubleComplex* a, const double b, cufftDoubleComplex* out, int n, int start)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[start + i].x = a[start + i].x + b; // modifies only the real part.
		out[start + i].y = a[start + i].y; // keeps the imaginary part unchanged.
    }
}

__global__ void vector_mutiply_scalar(const cufftDoubleComplex* a, const double b, cufftDoubleComplex* out, int n, int start)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[start + i].x = a[start + i].x*b;
        out[start + i].y = a[start + i].y*b;
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

__global__ void complex_to_real(const cuDoubleComplex* x_c, double* x, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N)
        x[idx] = x_c[idx].x; // only copy the real part
}

__device__ inline cufftDoubleComplex cotangent_complex(cufftDoubleComplex a)
{

    cufftDoubleComplex cs;
    cufftDoubleComplex cc;

    cos(a, cc);
    sin(a, cs);

    return cuCdiv(cc, cs);
    //return  make_cuDoubleComplex(sin(2 * a.x)/ (cosh(2 * a.y) - cos(2 * a.x)), -sinh(2* a.y)/ (cosh(2 * a.y) - cos(2 * a.x))); // https://dlmf.nist.gov/4.21#E40
	//double coeff = 1.0 / ; // this is the normalization factor

	//return make_cuDoubleComplex(top.x * coeff, top.y * coeff);
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

__global__ void compute_rhs_phi_expression(const cuDoubleComplex* Z, const cuDoubleComplex* V1, const cuDoubleComplex* V2, cuDoubleComplex* result, double rho, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		// Calculate the right-hand side of the phi equation
		double Z_imag = cuCimag(Z[i]);
		double V1_abs2 = V1[i].x * V1[i].x + V1[i].y * V1[i].y;
		double V2_abs2 = V2[i].x * V2[i].x + V2[i].y * V2[i].y;
		double V1_dot_V2 = V1[1].x * V2[i].x + V1[i].y * V2[i].y;
		result[i].x = -(1 + rho) * Z_imag + 0.5 * V1_abs2 + 0.5 * rho * V2_abs2 - rho * V1_dot_V2;
        result[i].y = 0;
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