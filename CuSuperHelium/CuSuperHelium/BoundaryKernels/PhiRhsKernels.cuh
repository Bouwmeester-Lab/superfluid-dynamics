#pragma once
#ifndef PHI_RHS_KERNELS_H
#define PHI_RHS_KERNELS_H

#include "../constants.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cuda/std/complex>

/// <summary>
/// Computes the RHS of Phi on the GPU using the expression: -(1+rho) * Im(Z) + 0.5 * abs(V1)^2 + 0.5 * rho * abs(V2)^2. - rho * V1 dot V2
/// It assumes Kappa = 0, and there's no surface tension.
/// The imag part of result is 0. Since this is purely real, but it's useful to stick to complex for consistency.
/// </summary>
/// <param name="Z"></param>
/// <param name="V"></param>
/// <param name="result"></param>
/// <param name="alpha"></param>
/// <param name="N"></param>
/// <returns></returns>
__global__ void compute_rhs_phi_expression(const std_complex* Z, const std_complex* V1, const std_complex* V2, std_complex* result, double rho, int N);
__global__ void compute_rhs_helium_phi_expression(const std_complex* Z, const std_complex* V1, std_complex* result, double h, int N);

__global__ void compute_rhs_phi_expression(const std_complex* Z, const std_complex* V1, const std_complex* V2, std_complex* result, double rho, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        // Calculate the right-hand side of the phi equation
        double Z_imag = Z[i].imag();
        double V1_abs2 = V1[i].real() * V1[i].real() + V1[i].imag() * V1[i].imag();
        double V2_abs2 = V2[i].real() * V2[i].real() + V2[i].imag() * V2[i].imag();
        double V1_dot_V2 = V1[i].real() * V2[i].real() + V1[i].imag() * V2[i].imag();
        result[i] = -(1 + rho) * Z_imag + 0.5 * V1_abs2 + 0.5 * rho * V2_abs2 - rho * V1_dot_V2;
    }
}

__global__ void compute_rhs_helium_phi_expression(const std_complex* Z, const std_complex* V1, std_complex* result, double h, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double vdw = h / 3.0; //  20.447761896665416 *
        //printf("coeff before van der waals term: %.10e\n", vdw);
        result[i] = vdw * cuda::std::pow(1.0 + Z[i].imag() / h, -3.0) - vdw + 0.5 * V1[i].real() * V1[i].real() + 0.5 * V1[i].imag() * V1[i].imag(); // we can try to add the surface tension term
    }
}

__global__ void compute_rhs_helium_phi_expression_expansion_terms(const std_complex* Z, const std_complex* V1, std_complex* result, double h, int N, int order = 2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        //double vdw = 20.447761896665416 * h / 3.0;
        double kinetic_contribution = 0.5 * V1[i].real() * V1[i].real() + 0.5 * V1[i].imag() * V1[i].imag();
        double vdw = 0.0;
        switch (order)
        {
        case 3:
            vdw += -10.0 / 3.0 * cuda::std::pow(Z[i].imag(), 3.0) / (h * h);
        case 2:
            vdw += 2.0 * cuda::std::pow(Z[i].imag(), 2.0) / h;
        case 1:
            vdw += -Z[i].imag();
        default:
            break;
        }
        double prefactor = 1.0; // 20.447761896665416;
        //printf("coeff before van der waals term: %.10e\n", vdw);
        result[i] = prefactor * vdw + kinetic_contribution; // we can try to add the surface tension term
    }
}

__device__ __forceinline__ double inverse_radius_of_curvature(const std_complex Zp, const std_complex Zpp)
{
    return (Zp.real() * Zpp.imag() - Zp.imag() * Zpp.real()) / cuda::std::pow(Zp.real() * Zp.real() + Zp.imag() * Zp.imag(), 1.5);
}

__global__ void compute_rhs_helium_phi_expression_with_surface_tension(const std_complex* Z, const std_complex* Zp, const std_complex* Zpp, const std_complex* V1, std_complex* result, double h, double kappa, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        // Calculate the right-hand side of the phi equation
        double Z_imag = Z[i].imag();
        double V1_abs2 = V1[i].real() * V1[i].real() + V1[i].imag() * V1[i].imag();
        //double curvature_term = ;
        //printf("Curvature term at index %d: %f, Xp %f, Yp %f\n", i, curvature_term, Zp[i].real(), Zp[i].imag());
        result[i] = 20.447761896665416 * h / 3.0 * (1.0 / cuda::std::pow(1.0 + Z_imag / h, 3) - 1) + 0.5 * V1_abs2 + kappa * inverse_radius_of_curvature(Zp[i], Zpp[i]);
    }
}

#endif
