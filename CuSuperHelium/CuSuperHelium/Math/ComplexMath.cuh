#pragma once
#ifndef COMPLEX_MATH_H
#define COMPLEX_MATH_H

#include "../constants.cuh"
#include "../cuDoubleComplexOperators.cuh"
#include "../PrecisionMath.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cufft.h>
#include <cuda/std/complex>

__device__ cufftDoubleComplex cotangent_complex(cufftDoubleComplex a);
__device__ cuDoubleComplex cotangent_green_function(cuDoubleComplex Zk, cuDoubleComplex Zj);
__device__ void cos(cufftDoubleComplex z, cufftDoubleComplex& out);
__device__ void sin(cufftDoubleComplex z, cufftDoubleComplex& zout);
__device__ cufftDoubleComplex fromReal(double a);
__device__ inline cufftDoubleComplex cMulScalar(double a, cufftDoubleComplex z);

__device__ __forceinline__ std_complex cot(std_complex z)
{
    return 1.0 / tan(z); // using the standard complex library to calculate the cotangent
}

__device__ inline cufftDoubleComplex cotangent_complex(cufftDoubleComplex a)
{
    auto a_std = reinterpret_cast<cuda::std::complex<double>*>(&a);
    // Using the standard complex library to calculate the cotangent
    auto t = 1.0 / tan(*a_std);
    return make_cuDoubleComplex(t.real(), t.imag());
}

__device__ inline cuDoubleComplex multiply_by_i(cuDoubleComplex z)
{
    return make_cuDoubleComplex(-z.y, z.x); // multiply by i is equivalent to rotating the complex number by 90 degrees counter-clockwise
}

__device__ inline std_complex multiply_by_i(std_complex z)
{
    return std_complex(-z.imag(), z.real()); // multiply by i is equivalent to rotating the complex number by 90 degrees counter-clockwise
}

__device__ inline std_complex cotangent_green_function(std_complex Zk, std_complex Zj)
{
    return cot(0.5 * (Zk - Zj));
}

__device__ inline void cos(cufftDoubleComplex z, cufftDoubleComplex& out)
{
    out.x = cos(z.x) * cosh(z.y);
    out.y = -sin(z.x) * sinh(z.y);
}

__device__ inline void sin(cufftDoubleComplex z, cufftDoubleComplex& zout)
{
    zout.x = sin(z.x) * cosh(z.y);
    zout.y = cos(z.x) * sinh(z.y);
}

__device__ inline cufftDoubleComplex fromReal(double a)
{
    cufftDoubleComplex out;
    out.x = a;
    out.y = 0.0;
    return out;
}

__device__ inline cufftDoubleComplex cMulScalar(double a, cufftDoubleComplex z)
{
    cufftDoubleComplex out(z);

    out.x = a * out.x;
    out.y = a * out.y;

    return out;
}

#endif
