#pragma once
#include "cuda_runtime.h"

#include "device_launch_parameters.h"

#include <stdio.h>
//#include <thrust/complex.h>
#include "constants.cuh"
#include <cufft.h>
#include "utilities.cuh"

void createMKernel(double* A, cufftDoubleComplex* ZPhi, cufftDoubleComplex* ZPhiPrime, cufftDoubleComplex* Zpp, double rho, int n);