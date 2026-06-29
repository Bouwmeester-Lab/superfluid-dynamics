#pragma once
#ifndef RADIAL_KERNELS_H
#define RADIAL_KERNELS_H

#include "../constants.cuh"
#include "cuda_runtime.h"
#include "../ElipticalGreenFunctions.cuh"



static __global__ void form_radial_kernel(double* matrixA, double* matrixB, const double* r, const double* z, const double* zprime, size_t num_elements, size_t batch_size)
{
    int j = blockIdx.y * blockDim.y + threadIdx.y; // row // used as integration variable
	int k = blockIdx.x * blockDim.x + threadIdx.x; // col // used as the evaluation point for the kernel
    int b = blockIdx.z; // batch index

    if (b >= batch_size) return; // out of bounds check for batch dimension

    if (k < num_elements && j < num_elements) {
        int indx = k + j * num_elements + b * num_elements * num_elements; // column major index
        if (k == j)
        {
            // we are on the diagonal:
            matrixA[indx] = 1.0; // TODO: figure out the PV of the integral here.
			matrixB[indx] = 1.0 - ElipticalGreenFunctions::calculate_c(j, num_elements, r[j], z[j]); // TODO: replace 1.0 with the PV of the integral here.
        }
        else
        {
            double nnorm = 1.0 / sqrt(1.0 + zprime[j] * zprime[j]);
			double nrho = -zprime[j] * nnorm; // normal vector component in the rho direction
            double neta = 1.0 * nnorm;
            double delta_s_j = sqrt(1.0 + zprime[j] * zprime[j]) * 2.0 * PI_d / num_elements;

            matrixA[indx] = r[j] * ElipticalGreenFunctions::calculate_G(r[k], z[k], r[j], z[j]) * delta_s_j; // 2.0 * pi / N represents delta r_j the spacing between collocations points.
            matrixB[indx] = r[j] * ElipticalGreenFunctions::calculate_dGdn(r[k], z[k], r[j], z[j], nrho, neta) * delta_s_j;
        }
    }
}
#endif
