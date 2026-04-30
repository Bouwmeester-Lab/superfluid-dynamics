#pragma once

#ifndef CREATE_M_H
#define CREATE_M_H

#include "cuda_runtime.h"

#include "device_launch_parameters.h"

#include <stdio.h>
//#include <thrust/complex.h>
#include <cuda/std/complex>
#include "constants.cuh"
#include <cufft.h>
#include "utilities.cuh"
#include "DelayedIntensityTerm.cuh"
#include "LightIntensity.cuh"

__global__ void createMKernel(double* A, const std_complex* Z, const std_complex* Zp, const std_complex* Zpp, double rho, int n, size_t batchSize);
__global__ void createFiniteDepthMKernel(double* A, const std_complex* Z, const std_complex* Zp, const std_complex* Zpp, double h, int n, size_t batchSize, bool infinite_depth = false);

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
/// <summary>
/// Creates the matrix M used in eq. 2.9 from Roberts 1983
/// </summary>
/// <param name="A">Matrix to fill</param>
/// <param name="diag">The precalculated diagonal using the expression for Mkk</param>
/// <param name="n">The size of the matrix (nxn)</param>
/// <returns></returns>
__global__ void createMKernel(double* A, const std_complex* const Z, const std_complex* const Zp, const std_complex* const Zpp, double rho, int n, size_t batchSize)
{
    int j = blockIdx.y * blockDim.y + threadIdx.y; // row
    int k = blockIdx.x * blockDim.x + threadIdx.x; // col
	int b = blockIdx.z; // batch index

	if (b >= batchSize) return; // out of bounds check for batch dimension

    if (k < n && j < n) {
        int indx = k + j * n + b*n*n; // column major index
        if (k == j)
        {
            // we are on the diagonal:
            A[indx] = 0.5 * (1 + rho) + 0.25 * (1 - rho) / PI_d * (Zpp[k + b*n] / Zp[k + b*n]).imag(); // imaginary part
        }
        else
        {
            A[indx] = 0.25 * (1 - rho) / PI_d * (Zp[k + b*n] * cotangent_green_function(Z[k+b*n], Z[j+b*n])).imag();// cuCmul(ZPhiPrime[k], cotangent_complex(cMulScalar(0.5, cuCsub(ZPhi[k], ZPhi[j])))).y; // 0.25 * (1 - rho) / PI_d * (cuCmul(ZPhiPrime[k], cotangent_complex(cMulScalar(0.5, cuCsub(ZPhi[k], ZPhi[j]))))).y;
        }
    }
}

__global__ void createFiniteDepthMKernel(double* A, const std_complex* const Z, const std_complex* const Zp, const std_complex* const Zpp, double h, int n, size_t batchSize, bool infinite_depth)
{
    const int j = blockIdx.y * blockDim.y + threadIdx.y; // row
    const int k = blockIdx.x * blockDim.x + threadIdx.x; // col
	const int b = blockIdx.z; // batch index

	if (b >= batchSize) return; // out of bounds check for batch dimension

    if (k < n && j < n) {
        int indx = k + j * n + b * n*n; // column major index
        if (k == j)
        {
            // we are on the diagonal:
            A[indx] = 0.5 + 0.25 / PI_d * (Zpp[k + b*n] / Zp[k + b*n]).imag(); // imaginary part
			if (!infinite_depth) // finite depth correction if needed
                A[indx] -= 0.25 / PI_d * cot(std_complex(0, Z[k + b * n].imag() + h)).imag(); // finite depth correction
        }
        else
        {
            A[indx] = 0.25 / PI_d * (Zp[k + b*n] * cotangent_green_function(Z[k + b*n], Z[j + b*n])).imag();
            if (!infinite_depth) 
            {
                std_complex cotTerm = 0.5 * (Z[k + b * n] - cuda::std::conj(Z[j + b * n])) + std_complex(0, h);
                A[indx] -= 0.25 / PI_d * cot(cotTerm).imag();
            }                
        }
    }
}



__global__ void compute_rhs_phi_expression(const std_complex* Z, const std_complex* V1, const std_complex* V2, std_complex* result, double rho, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        // Calculate the right-hand side of the phi equation
        double Z_imag = Z[i].imag();
        double V1_abs2 = V1[i].real() * V1[i].real() + V1[i].imag() * V1[i].imag();
        double V2_abs2 = V2[i].real() * V2[i].real() + V2[i].imag() * V2[i].imag();
        double V1_dot_V2 = V1[1].real() * V2[i].real() + V1[i].imag() * V2[i].imag();
        result[i] = -(1 + rho) * Z_imag + 0.5 * V1_abs2 + 0.5 * rho * V2_abs2 - rho * V1_dot_V2;
    }
}

__global__ void compute_rhs_helium_phi_expression(const std_complex* Z, const std_complex* V1, std_complex* result, double h, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double vdw = h / 3.0; //  20.447761896665416 *
        //printf("coeff before van der waals term: %.10e\n", vdw);
		result[i] = vdw * cuda::std::pow(1.0 + Z[i].imag() / h, -3.0) - vdw + 0.5 * V1[i].real() * V1[i].real() + 0.5* V1[i].imag() * V1[i].imag(); // we can try to add the surface tension term
    }
}

template <size_t N>
__global__ void add_optical_field_drive_terms(std_complex* result, double currentTime, const std_complex* Z, const std_complex* lowerVelocities, DelayedIntensityTermDevice<N> delayedIntensityTerm, OptomechanicalVariables variables, bool saveProgress = false)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
		double intensity = LightIntensity::compute_intensity(Z[i].imag(), Z[i].real(), variables);

        double delayedIntensity = delayedIntensityTerm.calculate_new_delayed_intensity(currentTime, intensity, i);

        if (saveProgress) {
			delayedIntensityTerm.save_value(delayedIntensity, currentTime, i);
        }
        result[i] += variables.DampingStrength * lowerVelocities[i].imag(); // this is the damping term
        result[i] += variables.Beta * delayedIntensity;
		result[i] += LightIntensity::get_current_intensity_drive_strength(variables) * intensity; // add the current intensity as well, since the delayed term only accounts for the past contribution
        //result[i] += 1e8;
    }
}

template <size_t N>
__global__ void add_optical_field_drive_terms_no_time_depence(std_complex* result, const std_complex* Z, const std_complex* lowerVelocities, OptomechanicalVariables variables)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double intensity = LightIntensity::compute_intensity(Z[i].imag(), Z[i].real(), variables);

        result[i] += variables.DampingStrength * lowerVelocities[i].imag(); // this is the damping term
        result[i] += LightIntensity::get_current_intensity_drive_strength(variables) * intensity; // add the current intensity as well, since the delayed term only accounts for the past contribution
        //result[i] += 1e8;
    }
}

template <size_t N>
__global__ void add_delayed_intensity_phi_rhs(std_complex* result, const std_complex* delayed, OptomechanicalVariables variables)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) 
    {
		result[i] += delayed[i] +1e3; // add the delayed intensity contribution to the RHS of the phi equation. This is the term that accounts for the past contribution of the optical field to the superfluid dynamics.
    }
}

template <size_t N>
__global__ void calculate_intensity_delayed_rhs(std_complex* result, const std_complex*Z, const std_complex* delayed, OptomechanicalVariables variables)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double intensity = LightIntensity::compute_intensity(Z[i].imag(), Z[i].real(), variables);
		result[i] = variables.Beta * intensity - 1.0 / variables.Tau * delayed[i]; // RHS of the delayed intensity term in the augmented system. Removes explicit time dependence.
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
			vdw += -10.0/3.0 * cuda::std::pow(Z[i].imag(), 3.0) / (h * h);
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



#endif // !CREATE_M_H


