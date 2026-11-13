#pragma once
#ifndef OPTOMECHANICAL_DRIVE_KERNELS_H
#define OPTOMECHANICAL_DRIVE_KERNELS_H

#include "Core/constants.cuh"
#include "Optomechanics/DelayedIntensityTerm.cuh"
#include "Optomechanics/LightIntensity.cuh"
#include "Optomechanics/OptomechanicalVariables.h"
#include "Physics/DimensionlessParameters.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

static __global__ void add_optical_field_drive_terms(const size_t N, std_complex* result, double currentTime, const double* frequency_shift, const std_complex* Z, const std_complex* lowerVelocities, DelayedIntensityTermDevice delayedIntensityTerm, OptomechanicalVariables variables, DimensionlessParameters properties, bool saveProgress = false)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        // TODO: update this to make use of the frequency shift computed in LightIntensity, instead of recomputing the intensity from Z. This will require passing the frequency shift as an argument to this function, and updating the way we compute the intensity to use that frequency shift instead of recomputing it from Z.
        double intensity = LightIntensity::compute_intensity(*frequency_shift, Z[i].real(), variables) * LightIntensity::compute_x_profile(Z[i].real(), variables.location_x0_mode, variables.sigma_optical_mode);

        double delayedIntensity = delayedIntensityTerm.calculate_new_delayed_intensity(currentTime, intensity, i);

        if (saveProgress) {
            delayedIntensityTerm.save_value(delayedIntensity, currentTime, i);
        }
        result[i] += variables.DampingStrength * lowerVelocities[i].imag(); // this is the damping term
        result[i] += variables.Beta * LightIntensity::get_current_intensity_drive_strength(variables, variables.sigma_thermal_mode, properties) * delayedIntensity;
        result[i] += LightIntensity::get_current_intensity_drive_strength(variables, variables.sigma_optical_mode, properties) * intensity; // add the current intensity as well, since the delayed term only accounts for the past contribution
        //result[i] += 1e8;
    }
}

static __device__ void add_optical_field_drive_terms_no_time_depence(const size_t N, std_complex* result, const double prefix, const double* frequency_shift, const std_complex* Z, const std_complex* lowerVelocities, OptomechanicalVariables variables, DimensionlessParameters properties)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double intensity = LightIntensity::compute_intensity(*frequency_shift, Z[i].real(), variables) * LightIntensity::compute_x_profile(Z[i].real(), variables.location_x0_mode, variables.sigma_optical_mode);

        result[i] += variables.DampingStrength * lowerVelocities[i].imag(); // this is the damping term
        result[i] += prefix * LightIntensity::get_current_intensity_drive_strength(variables, variables.sigma_optical_mode, properties) * intensity; // add the current intensity as well, since the delayed term only accounts for the past contribution
        //result[i] += 1e8;
    }
}

static __global__ void add_optical_drive_sine_wave_autonomous(const size_t N, std_complex* rhs, const std_complex* devTheta, const double* frequency_shift, const std_complex* Z, const std_complex* lowerVelocities, OptomechanicalVariables variables, DimensionlessParameters properties)
{
    add_optical_field_drive_terms_no_time_depence(N, rhs, 0.5*(1.0+sin((*devTheta).real())), frequency_shift, Z, lowerVelocities, variables, properties);
}

static __global__ void add_optical_field_drive_terms_no_time_depence(const size_t N, std_complex* result, const std_complex* devPrefix, const double* frequency_shift, const std_complex* Z, const std_complex* lowerVelocities, OptomechanicalVariables variables, DimensionlessParameters properties)
{
    add_optical_field_drive_terms_no_time_depence(N, result, (*devPrefix).real(), frequency_shift, Z, lowerVelocities, variables, properties);
}

static __global__ void add_optical_field_drive_terms_no_time_depence(const size_t N, std_complex* result, const double* frequency_shift, const std_complex* Z, const std_complex* lowerVelocities, OptomechanicalVariables variables, DimensionlessParameters properties)
{
    add_optical_field_drive_terms_no_time_depence(N, result, 1.0, frequency_shift, Z, lowerVelocities, variables, properties);
}

static __global__ void add_delayed_intensity_phi_rhs(const size_t N, std_complex* result, const std_complex* delayed, OptomechanicalVariables variables, DimensionlessParameters properties)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        result[i] += variables.Beta * LightIntensity::get_current_intensity_drive_strength(variables, variables.sigma_thermal_mode, properties) * delayed[i]; // add the delayed intensity contribution to the RHS of the phi equation. This is the term that accounts for the past contribution of the optical field to the superfluid dynamics.
    }
}

static __global__ void set_growth_intensity_rate(std_complex* rhs_intensity, const double ramp_rate)
{
    rhs_intensity[0] = std_complex(ramp_rate, 0.0); // set the growth rate of the intensity to the specified ramp rate.
}

static __global__ void set_theta_rhs(std_complex* rhs_theta, const double omega)
{
    rhs_theta[0] = std_complex(omega, 0.0); // set the growth rate of the intensity to the specified ramp rate.
}

static __global__ void calculate_intensity_delayed_rhs(const size_t N, std_complex* result, const double* frequency_shift, const std_complex* Z, const std_complex* delayed, OptomechanicalVariables variables)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double intensity = LightIntensity::compute_intensity(*frequency_shift, Z[i].real(), variables) * LightIntensity::compute_x_profile(Z[i].real(), variables.location_x0_mode, variables.sigma_thermal_mode);
        result[i] = (intensity - delayed[i]) / variables.Tau; // RHS of the delayed intensity term in the augmented system. Removes explicit time dependence.
    }
}

static __global__ void calculate_ramped_intensity_delayed_rhs(const size_t N, std_complex* result, const std_complex* rampedIntensity, const double* frequency_shift, const std_complex* Z, const std_complex* delayed, OptomechanicalVariables variables)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double intensity = (*rampedIntensity).real() * LightIntensity::compute_intensity(*frequency_shift, Z[i].real(), variables) * LightIntensity::compute_x_profile(Z[i].real(), variables.location_x0_mode, variables.sigma_thermal_mode);
        result[i] = (intensity - delayed[i]) / variables.Tau; // RHS of the delayed intensity term in the augmented system. Removes explicit time dependence.
    }
}

static __global__ void calculate_sine_modulated_intensity_delayed_rhs(const size_t N, std_complex* result, const std_complex* theta, const double* frequency_shift, const std_complex* Z, const std_complex* delayed, OptomechanicalVariables variables)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double intensity = 0.5*(1.0 + sin( (*theta).real())) * LightIntensity::compute_intensity(*frequency_shift, Z[i].real(), variables) * LightIntensity::compute_x_profile(Z[i].real(), variables.location_x0_mode, variables.sigma_thermal_mode);
        result[i] = (intensity - delayed[i]) / variables.Tau; // RHS of the delayed intensity term in the augmented system. Removes explicit time dependence.
    }
}

#endif
