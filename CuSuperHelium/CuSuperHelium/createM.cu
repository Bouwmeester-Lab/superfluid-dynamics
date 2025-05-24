#pragma once
#include "cuda_runtime.h"

#include "device_launch_parameters.h"

#include <stdio.h>
//#include <thrust/complex.h>
#include "constants.cuh"

/// <summary>
/// Creates the matrix M used in eq. 2.9 from Roberts 1983
/// </summary>
/// <param name="A">Matrix to fill</param>
/// <param name="diag">The precalculated diagonal using the expression for Mkk</param>
/// <param name="n">The size of the matrix (nxn)</param>
/// <returns></returns>
//__global__ void createMKernel(double* A, thrust::complex<double>* Z, thrust::complex<double>* Zp, thrust::complex<double>* Zpp, double rho, int n)
//{
//    int k = blockIdx.y * blockDim.y + threadIdx.y; // row
//    int j = blockIdx.x * blockDim.x + threadIdx.x; // col
//
//    if (k < n && j < n) {
//        int indx = k + j * n; // column major index
//        if (k == j) 
//        {
//            // we are in the diagonal:
//            A[indx] = 0.5 * (1 + rho) + 0.25 * (1 - rho) / PI_d * (Zpp[k] / Zp[k]).imag();
//        }
//        else 
//        {
//            A[indx] = 0.25 * (1 - rho) / PI_d * (Zp[k] * cotangent((Z[k] - Z[j]) * 0.5)).imag();
//        }
//    }
//}