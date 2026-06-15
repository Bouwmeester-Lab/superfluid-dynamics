#pragma once
#ifndef MATRIX_KERNELS_H
#define MATRIX_KERNELS_H

#include "../constants.cuh"
#include "../Math/ComplexMath.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cuda/std/complex>

static __global__ void createMKernel(double* A, const std_complex* Z, const std_complex* Zp, const std_complex* Zpp, double rho, int n, size_t batchSize);
static __global__ void createFiniteDepthMKernel(double* A, const std_complex* Z, const std_complex* Zp, const std_complex* Zpp, double h, int n, size_t batchSize, bool infinite_depth = false);

/// <summary>
/// Creates the matrix M used in eq. 2.9 from Roberts 1983
/// </summary>
/// <param name="A">Matrix to fill</param>
/// <param name="diag">The precalculated diagonal using the expression for Mkk</param>
/// <param name="n">The size of the matrix (nxn)</param>
/// <returns></returns>
static __global__ void createMKernel(double* A, const std_complex* const Z, const std_complex* const Zp, const std_complex* const Zpp, double rho, int n, size_t batchSize)
{
    int j = blockIdx.y * blockDim.y + threadIdx.y; // row
    int k = blockIdx.x * blockDim.x + threadIdx.x; // col
    int b = blockIdx.z; // batch index

    if (b >= batchSize) return; // out of bounds check for batch dimension

    if (k < n && j < n) {
        int indx = k + j * n + b * n * n; // column major index
        if (k == j)
        {
            // we are on the diagonal:
            A[indx] = 0.5 * (1 + rho) + 0.25 * (1 - rho) / PI_d * (Zpp[k + b * n] / Zp[k + b * n]).imag(); // imaginary part
        }
        else
        {
            A[indx] = 0.25 * (1 - rho) / PI_d * (Zp[k + b * n] * cotangent_green_function(Z[k + b * n], Z[j + b * n])).imag();// cuCmul(ZPhiPrime[k], cotangent_complex(cMulScalar(0.5, cuCsub(ZPhi[k], ZPhi[j])))).y; // 0.25 * (1 - rho) / PI_d * (cuCmul(ZPhiPrime[k], cotangent_complex(cMulScalar(0.5, cuCsub(ZPhi[k], ZPhi[j]))))).y;
        }
    }
}

static __global__ void createFiniteDepthMKernel(double* A, const std_complex* const Z, const std_complex* const Zp, const std_complex* const Zpp, double h, int n, size_t batchSize, bool infinite_depth)
{
    const int j = blockIdx.y * blockDim.y + threadIdx.y; // row
    const int k = blockIdx.x * blockDim.x + threadIdx.x; // col
    const int b = blockIdx.z; // batch index

    if (b >= batchSize) return; // out of bounds check for batch dimension

    if (k < n && j < n) {
        int indx = k + j * n + b * n * n; // column major index
        if (k == j)
        {
            // we are on the diagonal:
            A[indx] = 0.5 + 0.25 / PI_d * (Zpp[k + b * n] / Zp[k + b * n]).imag(); // imaginary part
            if (!infinite_depth) // finite depth correction if needed
                A[indx] -= 0.25 / PI_d * cot(std_complex(0, Z[k + b * n].imag() + h)).imag(); // finite depth correction
        }
        else
        {
            A[indx] = 0.25 / PI_d * (Zp[k + b * n] * cotangent_green_function(Z[k + b * n], Z[j + b * n])).imag();
            if (!infinite_depth)
            {
                std_complex cotTerm = 0.5 * (Z[k + b * n] - cuda::std::conj(Z[j + b * n])) + std_complex(0, h);
                A[indx] -= 0.25 / PI_d * cot(cotTerm).imag();
            }
        }
    }
}

#endif
