#pragma once
#ifndef CONSTANTS_H
#define CONSTANTS_H

#include "math_constants.h"
#include <cuda/std/complex>
//#include <thrust/complex.h>

constexpr double PI_d = CUDART_PI;
typedef cuda::std::complex<double> std_complex;

//__device__ thrust::complex<double> cotangent(thrust::complex<double> z)
//{
//	return 1.0 / thrust::tan(z);
//}

#endif // CONSTANTS_H

