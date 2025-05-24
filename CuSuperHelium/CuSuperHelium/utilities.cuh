#pragma once
#ifndef UTILITIES_H
#define UTILITIES_H

#include <stdio.h>
#include <cufft.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void complex_pointwise_mul(
    const cufftDoubleComplex* a,
    const cufftDoubleComplex* b,
    cufftDoubleComplex* result,
    const int n
);

#endif