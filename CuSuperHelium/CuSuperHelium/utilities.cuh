#pragma once
#ifndef UTILITIES_H
#define UTILITIES_H

#include <stdio.h>
#include <cufft.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cusolverDn.h>
#include <iostream>

__global__ void complex_pointwise_mul(
    const cufftDoubleComplex* a,
    const cufftDoubleComplex* b,
    cufftDoubleComplex* result,
    const int n
);
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




__global__ void complex_pointwise_mul(
    const cufftDoubleComplex* a,
    const cufftDoubleComplex* b,
    cufftDoubleComplex* result,
    const int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        cufftDoubleComplex x = a[i];
        cufftDoubleComplex y = b[i];
        result[i].x = x.x * y.x - x.y * y.y;  // real
        result[i].y = x.x * y.y + x.y * y.x;  // imag
    }
}

__global__ void vector_subtract_complex_real(const cufftDoubleComplex* a, const double* b, cufftDoubleComplex* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i].x = a[i].x - b[i]; // modifies only the real part.
    }
}

__global__ void vector_scalar_add_complex_real(const cufftDoubleComplex* a, const double b, cufftDoubleComplex* out, int n, int start)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[start + i].x = a[start + i].x + b; // modifies only the real part.
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