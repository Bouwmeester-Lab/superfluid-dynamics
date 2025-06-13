#pragma once

#ifndef CREATE_M_H
#define CREATE_M_H

#include "cuda_runtime.h"

#include "device_launch_parameters.h"

#include <stdio.h>
//#include <thrust/complex.h>
#include "constants.cuh"
#include <cufft.h>
#include "utilities.cuh"

__global__ void createMKernel(double* A, cufftDoubleComplex* ZPhi, cufftDoubleComplex* ZPhiPrime, cufftDoubleComplex* Zpp, double rho, int n);

/// <summary>
/// Creates the matrix M used in eq. 2.9 from Roberts 1983
/// </summary>
/// <param name="A">Matrix to fill</param>
/// <param name="diag">The precalculated diagonal using the expression for Mkk</param>
/// <param name="n">The size of the matrix (nxn)</param>
/// <returns></returns>
__global__ void createMKernel(double* A, cufftDoubleComplex* ZPhi, cufftDoubleComplex* ZPhiPrime, cufftDoubleComplex* Zpp, double rho, int n)
{
    int k = blockIdx.y * blockDim.y + threadIdx.y; // row
    int j = blockIdx.x * blockDim.x + threadIdx.x; // col

    if (k < n && j < n) {
        int indx = k + j * n; // column major index
        if (k == j)
        {
            // we are on the diagonal:
            A[indx] = 0.5 * (1 + rho) + 0.25 * (1 - rho) / PI_d * cuCdiv(Zpp[k], ZPhiPrime[k]).y; // imaginary part
        }
        else
        {
            A[indx] = 0.25 * (1 - rho) / PI_d * (cuCmul(ZPhiPrime[k], cotangent_complex(cMulScalar(0.5, cuCsub(ZPhi[k], ZPhi[j]))))).y;
        }
    }
}

#endif // !CREATE_M_H


