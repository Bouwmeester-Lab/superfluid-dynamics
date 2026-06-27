#pragma once
#ifndef CONSTANTS_H
#define CONSTANTS_H

#include "math_constants.h"
#include <cuda/std/complex>
//#include <thrust/complex.h>

constexpr double PI_d = CUDART_PI;
constexpr double PI_OVER_2_d = 1.570796326794896558;
constexpr double ONE_OVER_PI_d = 0.31830988618379069122;
constexpr double hbar_d = 1.054571817e-34; // Planck's constant over 2*pi in J*s
constexpr double alpha_hamaker_d = 3.5e-24; //  6.3 https://arxiv.org/html/2504.13001v1#S5
typedef cuda::std::complex<double> std_complex;

//__device__ thrust::complex<double> cotangent(thrust::complex<double> z)
//{
//	return 1.0 / thrust::tan(z);
//}

#endif // CONSTANTS_H

