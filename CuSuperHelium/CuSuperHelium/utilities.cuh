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
void vector_subtract_complex_real(const cufftDoubleComplex* a, const double* b, cufftDoubleComplex* out, int n);
/// <summary>
/// Element wise addition. It will add b to the real part of a for every element in a.
/// </summary>
/// <param name="a"></param>
/// <param name="b"></param>
/// <param name="out"></param>
/// <param name="n"></param>
void vector_scalar_add_complex_real(const cufftDoubleComplex* a, const double b, cufftDoubleComplex* out, int n, int start);
cufftDoubleComplex cotangent_complex(cufftDoubleComplex a);
void cotangent_complex(const cufftDoubleComplex* a, cufftDoubleComplex* out, int n);
void cos(cufftDoubleComplex z, cufftDoubleComplex& out);
void sin(cufftDoubleComplex z, cufftDoubleComplex& zout);
static cufftDoubleComplex fromReal(double a);
void real_to_complex(const double* x, cuDoubleComplex* x_c, int N);
cufftDoubleComplex cMulScalar(double a, cufftDoubleComplex z);
void checkCuda(cudaError_t result);
void checkCusolver(cusolverStatus_t status);

#endif