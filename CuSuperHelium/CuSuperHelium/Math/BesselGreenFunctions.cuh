#pragma once
#ifndef BESSEL_GREEN_FUNCTIONS_H
#define BESSEL_GREEN_FUNCTIONS_H

#include "../Common/CudaChecks.cuh"

#include "cuda_runtime.h"
#include <vector>
#include <boost/math/special_functions/bessel.hpp>
#include <iterator>

#include "RadialModels.h"

static __global__ void calculateBnMatrix(const double* dev_r, const double* devKappa, double* Bn, size_t Nb, size_t N_collocations)
{
	size_t k = blockIdx.x * blockDim.x + threadIdx.x; // bessel function n
	size_t j = blockIdx.y * blockDim.y + threadIdx.y; // collocation point j
	if (j < N_collocations && k < Nb)
	{
		Bn[k * N_collocations + j] = j0(devKappa[k] * dev_r[j]);
	}
}

static __global__ void calculateJ1CollocationMatrix(const double* dev_r, const double* devKappa, double* BPrimeN, size_t Nb, size_t N_collocations)
{
	size_t k = blockIdx.x * blockDim.x + threadIdx.x; // bessel function n
	size_t j = blockIdx.y * blockDim.y + threadIdx.y; // collocation point j
	if (j < N_collocations && k < Nb)
	{
		BPrimeN[k * N_collocations + j] = j1(devKappa[k] * dev_r[j]);
	}
}


template <size_t Nb, size_t N_collocations>
class DirichletNeumannBesselGreenFunctions
{
public:
	DirichletNeumannBesselGreenFunctions(double R = 1.0);
	~DirichletNeumannBesselGreenFunctions();
	void initialize(RadialPointers pointers, RadialProperties properties);
	void calculateMatrices(RadialPointers pointers, RadialProperties properties);
	/// Calculates the green function at the collocation point k using for the source point j using the Bessel function of the first kind of order 0 and the normalization factor Wn.
	/// </summary>
	/// <param name="k">Represents the field point where the green function is evaluated.</param>
	/// <param name="j">Represents the source point affecting the green function is evaluated.</param>
	/// <param name="pointers">Structure containing device pointers to the radial and z coordinates of the collocation points.</param>
	/// <param name="depth">The depth of the domain.</param>
	/// <param name="Nb">The number of modes.</param>
	/// <returns></returns>
	__device__ __inline__ double calculateGreenFunction(size_t k, size_t j, RadialPointers pointers, RadialProperties properties);
	__device__ __inline__ double calculatedGdn(size_t k, size_t j, RadialPointers pointers, RadialProperties properties);
private:
	double* dev_r; // Device pointer to store the radial coordinates of the collocation points.

	double* devZerosJ0; // Device pointer to store the zeros of the Bessel function J0, kappa_n = \beta_n / R, where beta_n is the n-th zero of J0.
	double* devKappa; // Device pointer to store the values of kappa_n = \beta_n / R, where beta_n is the n-th zero of J0.
	double* devWn; // Device pointer to store the normalization of the green function: Wn = 1/(R^2 * J1(\beta_n)^2 *\kappa_n), where J1 is the Bessel function of the first kind of order 1.

	double* devBn; // Device pointer to the matrix storing Bkj = J_0(\beta_k * r_j / R), where r_j is the j-th collocation point and \beta_k is the k-th zero of J0.
	double* devJ1; // Device pointer to the matrix storing B'kj = J_1(\beta_k * r_j / R), where r_j is the j-th collocation point and \beta_k is the k-th zero of J0.
	dim3 matrix_threads = dim3(16, 16);
	// Number of blocks in the grid for the kernel launch.
	dim3 matrix_blocks = dim3((Nb + 15) / 16, (N_collocations + 15) / 16); 

	/// <summary>
	/// Access the Bkj value from the device memory.
	/// </summary>
	/// <param name="n">Represents the mode number.</param>
	/// <param name="j">Represents the collocation point.</param>
	/// <returns></returns>
	__device__ __inline__ double getBnj(size_t n, size_t j) const;

	__device__ __inline__ double getBPrimenj(size_t n, size_t j) const { return devJ1[n * N_collocations + j]; }

	/// <summary>
	/// Obtain the normalization factor Wn from the device memory.
	/// </summary>
	/// <param name="n">Represents the mode number.</param>
	/// <returns></returns>
	__device__ __inline__ double getWn(size_t n) const;

	/// <summary>
	/// Calculates the z component of the Green's function.
	/// </summary>
	/// <param name="k">Field collocation point to use</param>
	/// <param name="j">Source collocation point</param>
	/// <param name="n">Mode number</param>
	/// <param name="pointers">Structure containing device pointers to the radial and z coordinates of the collocation points.</param>
	/// <param name="depth">The depth of the domain.</param>
	/// <returns></returns>
	__device__ __inline__ double calculate_g(size_t k, size_t j, size_t n, RadialPointers pointers, double depth) const;
	__device__ __inline__ double calculate_g_prime(size_t k, size_t j, size_t n, RadialPointers pointers, double depth) const;

	/// <summary>
	/// Calculates nr: the rth component of the normal vector at the collocation point k. This is not normalized to avoid calculating twice the normalization factor.
	/// </summary>
	/// <param name="k"></param>
	/// <param name="pointers"></param>
	/// <returns></returns>
	__device__ __inline__ double calculate_nr(size_t k, RadialPointers pointers) const;
	__device__ __inline__ double calculate_nz(size_t k, RadialPointers pointers) const { return 1.0; }
	/// <summary>
	/// Calculates the norm of the normal vector at the collocation point k.
	/// </summary>
	/// <param name="k"></param>
	/// <param name="pointers"></param>
	/// <returns></returns>
	__device__ __inline__ double calculate_norm_n(size_t k, RadialPointers pointers) const;

	

};



template<size_t Nb, size_t N_collocations>
DirichletNeumannBesselGreenFunctions<Nb, N_collocations>::DirichletNeumannBesselGreenFunctions(double R)
{
	CHECK_CUDA(cudaMalloc((void**)&devZerosJ0, Nb * sizeof(double)));
	CHECK_CUDA(cudaMalloc((void**)&devKappa, Nb * sizeof(double)));
	CHECK_CUDA(cudaMalloc((void**)&devWn, Nb * sizeof(double)));
	CHECK_CUDA(cudaMalloc((void**)&devBn, N_collocations * Nb * sizeof(double)));
	CHECK_CUDA(cudaMalloc((void**)&devJ1, N_collocations * Nb * sizeof(double)));
}

template<size_t Nb, size_t N_collocations>
void DirichletNeumannBesselGreenFunctions<Nb, N_collocations>::initialize(RadialPointers pointers, RadialProperties properties)
{
	const double* dev_r = pointers.dev_r;
	const double R = properties.R;

	std::vector<double> zeros_J0_host;
	std::vector<double> kappa_host;
	std::vector<double> Wn_host;
	boost::math::cyl_bessel_j_zero(0.0, 1, Nb, std::back_inserter(zeros_J0_host));

	double j1 = 0.0;
	//double kappa_n;
	for (size_t n = 0; n < Nb; ++n)
	{
		//kappa_host.push_back(zeros_J0_host[n] / R);
		kappa_host.push_back(zeros_J0_host[n] / R);
		j1 = boost::math::cyl_bessel_j(1.0, zeros_J0_host[n]);
		//Nn_host.push_back(0.5 * R * R * j1 * j1);
		Wn_host.push_back(1.0 / (R * R * j1 * j1 * kappa_host[n]));
	}

	CHECK_CUDA(cudaMemcpy(devZerosJ0, zeros_J0_host.data(), Nb * sizeof(double), cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(devKappa, kappa_host.data(), Nb * sizeof(double), cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(devWn, Wn_host.data(), Nb * sizeof(double), cudaMemcpyHostToDevice));
}
	

template<size_t Nb, size_t N_collocations>
void DirichletNeumannBesselGreenFunctions<Nb, N_collocations>::calculateMatrices(RadialPointers pointers, RadialProperties properties)
{
	calculateBnMatrix << <matrix_blocks, matrix_threads >> > (dev_r, devKappa, devBn, Nb, N_collocations);
	CHECK_CUDA(cudaGetLastError());
	calculateJ1CollocationMatrix << <matrix_blocks, matrix_threads >> > (dev_r, devKappa, devJ1, Nb, N_collocations);
	CHECK_CUDA(cudaGetLastError());
	CHECK_CUDA(cudaDeviceSynchronize());

}

template<size_t Nb, size_t N_collocations>
DirichletNeumannBesselGreenFunctions<Nb, N_collocations>::~DirichletNeumannBesselGreenFunctions()
{
	checkCuda(cudaFree(devZerosJ0));
	checkCuda(cudaFree(devKappa));
	checkCuda(cudaFree(devWn));
	checkCuda(cudaFree(devJ1));
	checkCuda(cudaFree(devBn));
}

template<size_t Nb, size_t N_collocations>
__device__ __inline__ double DirichletNeumannBesselGreenFunctions<Nb, N_collocations>::calculateGreenFunction(size_t k, size_t j, RadialPointers pointers, RadialProperties properties)
{
	if (k == j) 
	{
		return 0.0; // TODO: handle the singularity case when k == j, possibly using a limit.
	}
	else 
	{
		double sum = 0.0;
		for (size_t n = 0; n < Nb; ++n)
		{
			sum += getBnj(n, k) * getBnj(n, j) * getWn(n) * calculate_g(k, j, n, pointers, properties.depth);
		}
		return sum;
	}
}

template<size_t Nb, size_t N_collocations>
__device__ __inline__ double DirichletNeumannBesselGreenFunctions<Nb, N_collocations>::calculatedGdn(size_t k, size_t j, RadialPointers pointers, RadialProperties properties)
{
	if (k == j)
	{
		return 0.0; // TODO: handle the singularity case when k == j, possibly using a limit.
	}

	double sumr = 0.0;
	double sumz = 0.0;
	for (size_t n = 0; n < Nb; ++n) 
	{
		sumr -= getWn(n) * calculate_g(k, j, n, pointers, properties.depth) * getBPrimenj(n, j) * devKappa[n] * getBnj(n, k);
		sumz += getWn(n) * getBnj(n, k) * getBnj(n, j) * calculate_g_prime(k, j, n, pointers, properties.depth);
	}
	return  (calculate_nr(j, pointers) * sumr + calculate_nz(j, pointers) * sumz) / calculate_norm_n(j, pointers);
}


template<size_t Nb, size_t N_collocations>
inline __device__ __inline__ double DirichletNeumannBesselGreenFunctions<Nb, N_collocations>::getBnj(size_t n, size_t j) const
{
	return devBn[n * N_collocations + j];
}

template<size_t Nb, size_t N_collocations>
inline __device__ __inline__ double DirichletNeumannBesselGreenFunctions<Nb, N_collocations>::getWn(size_t n) const
{
	return devWn[n];
}

template<size_t Nb, size_t N_collocations>
inline __device__ __inline__ double DirichletNeumannBesselGreenFunctions<Nb, N_collocations>::calculate_g(size_t k, size_t j, size_t n, RadialPointers pointers, double depth) const
{
	return exp(-devKappa[n] * fabs(pointers.dev_z[k] - pointers.dev_z[j])) + exp(-devKappa[n] * (pointers.dev_z[k] + pointers.dev_z[j] + 2.0 * depth));
}

template<size_t Nb, size_t N_collocations>
__device__ __inline__ double DirichletNeumannBesselGreenFunctions<Nb, N_collocations>::calculate_g_prime(size_t k, size_t j, size_t n, RadialPointers pointers, double depth) const
{

	const double kappa = devKappa[n];
	const double zk = pointers.dev_z[k];
	const double zj = pointers.dev_z[j];

	if (k == j)
	{
		return 0.0; // TODO: singular/self term
	}

	const double dz = zk - zj;
	const double direct = exp(-kappa * fabs(dz));
	const double image = exp(-kappa * (zk + zj + 2.0 * depth));

	if (dz > 0.0)
	{
		return kappa * (direct - image);
	}
	else if (dz < 0.0)
	{
		return -kappa * (direct + image);
	}
	else
	{
		// k != j but same height: symmetric / PV convention.
		return -kappa * image;
	}
}

template<size_t Nb, size_t N_collocations>
__device__ __inline__ double DirichletNeumannBesselGreenFunctions<Nb, N_collocations>::calculate_nr(size_t k, RadialPointers pointers) const
{
	return -pointers.dev_z_prime[k];
}

template<size_t Nb, size_t N_collocations>
__device__ __inline__ double DirichletNeumannBesselGreenFunctions<Nb, N_collocations>::calculate_norm_n(size_t k, RadialPointers pointers) const
{
	return sqrt(1.0 + pointers.dev_z_prime[k] * pointers.dev_z_prime[k]);
}


#endif
