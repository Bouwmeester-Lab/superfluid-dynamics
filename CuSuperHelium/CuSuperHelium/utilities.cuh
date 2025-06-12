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
    if (i < n / 2) 
    {
        result[i] = cuCmul(a[i], make_cuDoubleComplex(0, static_cast<double>(i) / n)); // normalize by n since ifft doesn't
    }
    else if (n % 2 == 0 &&  i == n / 2) 
    {
        result[i].x = 0;
		result[i].y = -PI_d; // we want to treat the Nyquist frequency as exp(i*pi*j) which means
        // that the inverse fft of the fft of exp(i*pi*j) should give i pi * exp(i * pi *j). This happens when the coeff[n/2] = -pi.
		// Usually this coefficient should be -pi *n but cuFFT will NOT normalize by n, so when we do normalize manually by dividing by n, we get -pi.
    }
    else if (i < n) {
        result[i] = cuCmul(a[i], make_cuDoubleComplex(0, (static_cast<double>(i) - n) / n));
    }
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
        result[i].x = i * i * coeffsFft[i].x /n;
		result[i].y = i * i * coeffsFft[i].y /n;
    }
    else if (i == n / 2) 
    {
		result[i].x = -PI_d * PI_d; // real part, this is the same idea as in the first derivative. We want the second derivative of exp(i*pi*j) to be -pi^2 * exp(i*pi*j), this will happen only if this coefficient is this value
		result[i].y = 0;// setting this to zero seems fine? although I wonder if it's better to leave the same value as the theoretical value: N/2 * N/2 * coeffsFft[i].y / N ?
    }
    else if(i < n)
    {
		result[i].x = -(i - n) * (i - n) * coeffsFft[i].x/n; //https://math.mit.edu/~stevenj/fft-deriv.pdf
		result[i].y = -(i - n) * (i - n) * coeffsFft[i].y/n;
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
        cufftDoubleComplex cs;
        cufftDoubleComplex cc;

        cos(a[i], cc);
        sin(a[i], cs);

        auto z = cuCdiv(cc, cs);

        out[i].x = z.x;
        out[i].y = z.y;
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

__device__ cufftDoubleComplex cotangent_complex(cufftDoubleComplex a)
{

    cufftDoubleComplex cs;
    cufftDoubleComplex cc;

    cos(a, cc);
    sin(a, cs);

    return cuCdiv(cc, cs);
}

__device__ void cos(cufftDoubleComplex z, cufftDoubleComplex& out)
{
    out.x = cos(z.x) * cosh(z.y);
    out.y = -sin(z.x) * sinh(z.y);
}

__device__ void sin(cufftDoubleComplex z, cufftDoubleComplex& zout) {
    zout.x = sinh(z.x) * cos(z.y);
    zout.y = cosh(z.x) * sin(z.y);
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
        cuDoubleComplex xi = x[i];
        z[i] = make_cuDoubleComplex(cuCreal(xi), -cuCimag(xi));
    }
}

__global__ void compute_rhs_phi_expression(const cuDoubleComplex* Z, const cuDoubleComplex* V1, const cuDoubleComplex* V2, cuDoubleComplex* result, double rho, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		// Calculate the right-hand side of the phi equation
		double Z_imag = cuCimag(Z[i]);
		double V1_abs2 = cuCabs(V1[i]) * cuCabs(V1[i]);
		double V2_abs2 = cuCabs(V2[i]) * cuCabs(V2[i]);
		double V1_dot_V2 = cuCreal(V1[1]) * cuCreal(V2[i]) + cuCimag(V1[i]) * cuCimag(V2[i]);
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