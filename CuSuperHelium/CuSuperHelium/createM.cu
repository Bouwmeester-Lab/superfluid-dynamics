#pragma once
#include "cuda_runtime.h"

#include "device_launch_parameters.h"

#include <stdio.h>
//#include <thrust/complex.h>
#include "constants.cuh"
#include <cufft.h>
#include "utilities.cuh"

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
            // we are in the diagonal:
            A[indx] = 0.5 * (1 + rho) + 0.25 * (1 - rho) / PI_d * cuCdiv(Zpp[k], ZPhi[k]).y; // imaginary part
        }
        else 
        {
            A[indx] = 0.25 * (1 - rho) / PI_d * (cuCmul(ZPhiPrime[k],  cotangent_complex( cuCmul(cuCsub(ZPhi[k], ZPhi[j]), make_cuDoubleComplex(0.5, 0))))).y;
        }
    }
}