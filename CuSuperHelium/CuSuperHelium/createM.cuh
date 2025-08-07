#pragma once

#ifndef CREATE_M_H
#define CREATE_M_H

#include "cuda_runtime.h"

#include "device_launch_parameters.h"

#include <stdio.h>
//#include <thrust/complex.h>
#include <cuda/std/complex>
#include "constants.cuh"
#include <cufft.h>
#include "utilities.cuh"

__global__ void createMKernel(double* A, std_complex* Z, std_complex* Zp, std_complex* Zpp, double rho, int n);
__global__ void createFiniteDepthMKernel(double* A, std_complex* Z, std_complex* Zp, std_complex* Zpp, double h, int n);

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
__global__ void compute_rhs_phi_expression(const std_complex* Z, const std_complex* V1, const std_complex* V2, std_complex* result, double rho, int N);
__global__ void compute_rhs_helium_phi_expression(const std_complex* Z, const std_complex* V1, std_complex* result, double h, int N);

__device__ __forceinline__ double InfiniteDepthMij(size_t i, size_t j, const std_complex* Z, const std_complex* Zp, const std_complex* Zpp, double rho)
{
    if (i == j)
    {
        // we are on the diagonal:
        return 0.5 * (1 + rho) + 0.25 * (1 - rho) / PI_d * (Zpp[i] / Zp[i]).imag(); // imaginary part
    }
    else
    {
        return 0.25 * (1 - rho) / PI_d * (Zp[i] * cotangent_green_function(Z[i], Z[j])).imag();
    }
}
/// <summary>
/// Creates the matrix M used in eq. 2.9 from Roberts 1983
/// </summary>
/// <param name="A">Matrix to fill</param>
/// <param name="diag">The precalculated diagonal using the expression for Mkk</param>
/// <param name="n">The size of the matrix (nxn)</param>
/// <returns></returns>
__global__ void createMKernel(double* A, std_complex* Z, std_complex* Zp, std_complex* Zpp, double rho, int n)
{
    int j = blockIdx.y * blockDim.y + threadIdx.y; // row
    int k = blockIdx.x * blockDim.x + threadIdx.x; // col

    if (k < n && j < n) {
        int indx = k + j * n; // column major index
		A[indx] = InfiniteDepthMij(j, k, Z, Zp, Zpp, rho);
    }
}

__device__ __forceinline__ double FiniteDepthMij(size_t i, size_t j, const std_complex* Z, const std_complex* Zp, const std_complex* Zpp, double h)
{
    if(i == j)
    {
        // we are on the diagonal:
        return 0.5 + 0.25 / PI_d * (Zpp[i] / Zp[i]).imag() - 0.25 / PI_d * cot(std_complex(0, Z[i].imag() + h)).imag(); // imaginary part
    }
    else
    {
        std_complex cotTerm = 0.5 * (Z[i] - cuda::std::conj(Z[j])) + std_complex(0, h);
        return 0.25 / PI_d * (Zp[i] * cotangent_green_function(Z[i], Z[j])).imag() - 0.25 / PI_d * cot(cotTerm).imag();
    }
}

__global__ void createFiniteDepthMKernel(double* A, std_complex* Z, std_complex* Zp, std_complex* Zpp, double h, int n)
{
    int j = blockIdx.y * blockDim.y + threadIdx.y; // row
    int k = blockIdx.x * blockDim.x + threadIdx.x; // col

    if (k < n && j < n) {
        int indx = k + j * n; // column major index
		A[indx] = FiniteDepthMij(j, k, Z, Zp, Zpp, h);
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

__global__ void compute_rhs_helium_phi_expression(const std_complex* Z, const std_complex* V1, std_complex* result, double h, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        // Calculate the right-hand side of the phi equation
        double Z_imag = Z[i].imag();
        double V1_abs2 = V1[i].real() * V1[i].real() + V1[i].imag() * V1[i].imag();
        result[i] = h / 3.0 * (1.0/cuda::std::pow(1.0+Z_imag/h, 3) - 1) + 0.5 * V1_abs2;
    }
}

#endif // !CREATE_M_H


