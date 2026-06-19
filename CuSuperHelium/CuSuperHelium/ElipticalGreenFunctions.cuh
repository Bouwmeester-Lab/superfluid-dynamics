#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cmath>
#include <limits>
#include "constants.cuh"

__host__ __device__
double elliptic_K_parameter(double m)
{
    // K(m) = int_0^{pi/2} dtheta / sqrt(1 - m sin^2 theta)
    // valid for 0 <= m < 1

    if (m < 0.0) return NAN;
    if (m == 1.0) return INFINITY;
    if (m > 1.0) return NAN;

    double a = 1.0;
    double b = sqrt(1.0 - m);

    // AGM iteration
    for (int i = 0; i < 20; ++i)
    {
        double an = 0.5 * (a + b);
        double bn = sqrt(a * b);

        if (fabs(an - bn) <= 4.0 * std::numeric_limits<double>::epsilon() * an)
        {
            a = an;
            break;
        }

        a = an;
        b = bn;
    }

    return PI_d / (2.0 * a);
}

class ElipticalGreenFunctions
{
public:
	ElipticalGreenFunctions();
	~ElipticalGreenFunctions();
	__device__ __host__ double calculate_A(double r, double z, double rho, double eta);
	__device__ __host__ double calculate_m(double r, double z, double rho, double eta);
	// derivatives
	__device__ __host__ double calculate_dmdrho(double r, double z, double rho, double eta);
	__device__ __host__ double calculate_dmdeta(double r, double z, double rho, double eta);

	__device__ __host__ double calculate_dGdrho(double r, double z, double rho, double eta);
	__device__ __host__ double calculate_dGdeta(double r, double z, double rho, double eta);

private:

};

ElipticalGreenFunctions::ElipticalGreenFunctions()
{}

ElipticalGreenFunctions::~ElipticalGreenFunctions()
{}

__device__ __host__ double ElipticalGreenFunctions::calculate_A(double r, double z, double rho, double eta)
{
	return sqrt(pow(r + rho, 2.0) + pow(z - eta, 2.0));
}

__device__ __host__ double ElipticalGreenFunctions::calculate_m(double r, double z, double rho, double eta)
{
	return 4.0 * r * rho / (pow(r + rho, 2.0) + pow(z - eta, 2.0));
}

__device__ __host__ double ElipticalGreenFunctions::calculate_dmdrho(double r, double z, double rho, double eta)
{
	return 4.0*r*rho*(z-eta)/pow(pow(r+rho, 2.0) + pow(z-eta, 2.0), 2.0);
}

__device__ __host__ double ElipticalGreenFunctions::calculate_dmdeta(double r, double z, double rho, double eta)
{
	return 8 * r * rho * (z - eta) / pow(pow(r + rho, 2.0) + pow(z - eta, 2.0), 2.0);
}
