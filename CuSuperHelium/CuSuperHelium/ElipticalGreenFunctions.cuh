#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cmath>
#include <limits>
#include "constants.cuh"
#include "ElipticalIntegrals.cuh"

class ElipticalGreenFunctions
{
public:
	ElipticalGreenFunctions();
	~ElipticalGreenFunctions();
	__device__ __host__ __inline__ static double calculate_A(double r, double z, double rho, double eta);
	__device__ __host__ __inline__ static double calculate_m(double r, double z, double rho, double eta);
	/// <summary>
	/// Calculates the complementary m: mc = 1 - m from the analytical expression. This is used to avoid numerical issues when m is close to 1.
	/// </summary>
	/// <param name="r"></param>
	/// <param name="z"></param>
	/// <param name="rho"></param>
	/// <param name="eta"></param>
	/// <returns></returns>
	__device__ __host__ __inline__ static double calculate_mc(double r, double z, double rho, double eta);
	// derivatives
	__device__ __host__ __inline__ static double calculate_dmdrho(double r, double z, double rho, double eta);
	__device__ __host__ __inline__ static double calculate_dmdeta(double r, double z, double rho, double eta);

	__device__ __host__ __inline__ static double calculate_dGdrho(double r, double z, double rho, double eta);
	__device__ __host__ __inline__ static double calculate_dGdeta(double r, double z, double rho, double eta);

	__device__ __host__ __inline__ static double calculate_dGdn(double r, double z, double rho, double eta, double nrho, double neta);
	
	__device__ __host__ __inline__ static double calculate_G(double r, double z, double rho, double eta);
	__device__ __host__ __inline__ static double calculate_c(size_t index, size_t points, double r, double z);
private:

};

ElipticalGreenFunctions::ElipticalGreenFunctions()
{}

ElipticalGreenFunctions::~ElipticalGreenFunctions()
{}

__device__ __host__ __inline__ double ElipticalGreenFunctions::calculate_A(double r, double z, double rho, double eta)
{
	return sqrt(pow(r + rho, 2.0) + pow(z - eta, 2.0));
}

__device__ __host__ __inline__ double ElipticalGreenFunctions::calculate_m(double r, double z, double rho, double eta)
{
	return 4.0 * r * rho / (pow(r + rho, 2.0) + pow(z - eta, 2.0));
}

__device__ __host__ __inline__ double ElipticalGreenFunctions::calculate_mc(double r, double z, double rho, double eta)
{
	return (pow(r - rho, 2.0) + pow(z - eta, 2.0)) / (pow(r + rho, 2.0) + pow(z - eta, 2.0));
}

__device__ __host__ __inline__ double ElipticalGreenFunctions::calculate_dmdrho(double r, double z, double rho, double eta)
{
	return 4.0*r*rho*(z-eta)/pow(pow(r+rho, 2.0) + pow(z-eta, 2.0), 2.0);
}

__device__ __host__ __inline__ double ElipticalGreenFunctions::calculate_dmdeta(double r, double z, double rho, double eta)
{
	return 8 * r * rho * (z - eta) / pow(pow(r + rho, 2.0) + pow(z - eta, 2.0), 2.0);
}

__device__ __host__ __inline__ double ElipticalGreenFunctions::calculate_dGdrho(double r, double z, double rho, double eta)
{
	double A = calculate_A(r, z, rho, eta);
	double one_over_A = 1.0 / A;
	double m = calculate_m(r, z, rho, eta);
	double mc = calculate_mc(r, z, rho, eta);

	double dmdrho = calculate_dmdrho(r, z, rho, eta);

	double K, E;
	elliptic_K_E_parameter_from_m_mc(m, mc, &K, &E);
	double dKdm = elliptic_dKdm_parameter(m, mc, K, E);


	return ONE_OVER_PI_d * (-(r + rho) / pow(A, 3.0) * K + one_over_A * dmdrho * dKdm);
}

__device__ __host__ __inline__ double ElipticalGreenFunctions::calculate_dGdeta(double r, double z, double rho, double eta)
{
	double A = calculate_A(r, z, rho, eta);
	double one_over_A = 1.0 / A;
	double m = calculate_m(r, z, rho, eta);
	double mc = calculate_mc(r, z, rho, eta);

	double dmdeta = calculate_dmdeta(r, z, rho, eta);

	double K, E;
	elliptic_K_E_parameter_from_m_mc(m, mc, &K, &E);
	double dKdm = elliptic_dKdm_parameter(m, mc, K, E);

	return ONE_OVER_PI_d * ((z - eta) / pow(A, 3.0) * K + one_over_A * dmdeta * dKdm);
}

__device__ __host__ __inline__ double ElipticalGreenFunctions::calculate_dGdn(double r, double z, double rho, double eta, double nrho, double neta)
{
	return calculate_dGdrho(r, z, rho, eta) * nrho + calculate_dGdeta(r, z, rho, eta) * neta;
}

__device__ __host__ __inline__ double ElipticalGreenFunctions::calculate_G(double r, double z, double rho, double eta)
{
	double A = calculate_A(r, z, rho, eta);
	double m = calculate_m(r, z, rho, eta);
	double mc = calculate_mc(r, z, rho, eta);
	double K, E;
	elliptic_K_E_parameter_from_m_mc(m, mc, &K, &E);
	return ONE_OVER_PI_d * K / A;
	
}

__device__ __host__ __inline__ double ElipticalGreenFunctions::calculate_c(size_t index, size_t points, double r, double z)
{
	if (index == points - 1) 
	{
		// this is the last point which is on the boundary:
		return 0.5;
	}
	else {
		return 0.5;
	}
}
