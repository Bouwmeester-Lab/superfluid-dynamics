#pragma once
#include "ProblemProperties.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include "constants.cuh"
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

template <int N>
class EnergyBase
{
public:
	EnergyBase(ProblemProperties& properties);
	virtual ~EnergyBase();
	double getEnergy() const;
protected:
	ProblemProperties& properties;
	double* devEnergy;

	void* tempStorage = nullptr; // Temporary storage for CUB operations
	size_t tempStorageBytes = 0;

	virtual double scaleEnergy(double energy) const = 0; // Pure virtual function to scale energy
};

template <int N>
class KineticEnergy : public EnergyBase<N>
{
public:
	using EnergyBase<N>::EnergyBase; // Inherit constructor from EnergyBase
	void CalculateEnergy(std_complex* devPhi, std_complex* devZ, std_complex* devZp, std_complex* velocitiesLower);
	virtual double scaleEnergy(double energy) const override
	{
		return energy * 0.25 / PI_d;
	}
};

template <int N>
class GravitationalEnergy : public EnergyBase<N>
{
public:
	using EnergyBase<N>::EnergyBase; // Inherit constructor from EnergyBase
	void CalculateEnergy(std_complex* devZ, std_complex* devZp);
	virtual double scaleEnergy(double energy) const override
	{
		return energy * 0.25 * (1.0 + this->properties.rho) / PI_d;
	}
};

template <int N>
class SurfaceEnergy : public EnergyBase<N>
{
public:
	using EnergyBase<N>::EnergyBase; // Inherit constructor from EnergyBase
	void CalculateEnergy(std_complex* devZp);
	virtual double scaleEnergy(double energy) const override
	{
		return (energy - 2.0 * PI_d) * this->properties.kappa / (2.0 * PI_d);
	}
};

template <int N>
class VolumeFlux : public EnergyBase<N>
{
public:
	using EnergyBase<N>::EnergyBase; // Inherit constructor from EnergyBase
	void CalculateEnergy(std_complex* devZp, std_complex* velocities);
	virtual double scaleEnergy(double energy) const override
	{
		return energy * 0.5 / PI_d;
	}
};


struct KineticEnergyCombination
{
	const std_complex* Z;
	const std_complex* Phi;
	const std_complex* Zp;
	const std_complex* velocitiesLower;
	const std_complex* velocitiesUpper;
	const ProblemProperties properties;

	__host__ __device__ KineticEnergyCombination(const std_complex* z, const std_complex* phi, const std_complex* zp, const std_complex* velocitiesLower, const std_complex* velocitiesUpper, const ProblemProperties properties)
		: Z(z), Phi(phi), Zp(zp), velocitiesLower(velocitiesLower), velocitiesUpper(velocitiesUpper), properties(properties) {
	}

	__device__ double operator()(int k) const
	{
		return (Phi[k].real() + 0.5 * properties.U * (1.0 + properties.rho) * Z[k].real()) * (-1.0 * Zp[k].imag() * velocitiesLower[k].real() + Zp[k].real() * velocitiesLower[k].imag()) \
			- 0.5 * properties.U * ((velocitiesLower[k].real() + properties.rho * velocitiesUpper[k].real()) * Zp[k].real() + (velocitiesLower[k].imag() + properties.rho * velocitiesUpper[k].imag()) * Zp[k].imag() \
				+ 0.5 * properties.U * (1.0 - properties.rho) * Zp[k].real()) * Z[k].imag();
	}
};

struct GravitationalEnergyCombination
{
	const std_complex* Z;
	const std_complex* Zp;
	__host__ __device__ GravitationalEnergyCombination(const std_complex* z, const std_complex* zp)
		: Z(z), Zp(zp) {
	}
	__device__ double operator()(int k) const
	{
		return Z[k].imag() * Z[k].imag() * Zp[k].real();
	}
};

struct SurfaceEnergyCombination
{
	const std_complex* Zp;
	__host__ __device__ SurfaceEnergyCombination(const std_complex* zp)
		: Zp(zp) {}
	__device__ double operator()(int k) const
	{
		return sqrt(Zp[k].real() * Zp[k].real() + Zp[k].imag() * Zp[k].imag());
	}
};

struct VolumeFluxCombination
{
	const std_complex* Zp;
	const std_complex* velocities;
	__host__ __device__ VolumeFluxCombination(const std_complex* zp, const std_complex* velocities)
		: Zp(zp), velocities(velocities) {}
	__device__ double operator()(int k) const
	{
		return (velocities[k].imag() * Zp[k].real() + velocities[k].real() * Zp[k].imag());
	}
};

template <int N>
EnergyBase<N>::EnergyBase(ProblemProperties& properties) : properties(properties)
{
	cudaMalloc(&devEnergy, sizeof(double));
}

template <int N>
EnergyBase<N>::~EnergyBase()
{
	cudaFree(devEnergy);
}

template <int N>
double EnergyBase<N>::getEnergy() const
{
	cudaDeviceSynchronize();
	double e;
	cudaMemcpy(&e, devEnergy, sizeof(double), cudaMemcpyDeviceToHost);
	return this->scaleEnergy(e);
}

template<int N>
void KineticEnergy<N>::CalculateEnergy(std_complex* devPhi, std_complex* devZ, std_complex* devZp, std_complex* velocitiesLower)
{
	thrust::counting_iterator<int> countingIterator(0);

	// create a transform iterator that applies the KineticEnergyCombination functor
	KineticEnergyCombination kineticEnergyCombination(devZ, devPhi, devZp, velocitiesLower, velocitiesLower, this->properties);

	auto transformIterator = thrust::make_transform_iterator(countingIterator, kineticEnergyCombination);

	// get temporary storage size
	cub::DeviceReduce::Sum(this->tempStorage, this->tempStorageBytes, transformIterator, this->devEnergy, N);

	// allocate temporary storage
	cudaMalloc(&this->tempStorage, this->tempStorageBytes);

	// perform the reduction
	cub::DeviceReduce::Sum(this->tempStorage, this->tempStorageBytes, transformIterator, this->devEnergy, N);
}



template<int N>
void GravitationalEnergy<N>::CalculateEnergy(std_complex* devZ, std_complex* devZp)
{
	thrust::counting_iterator<int> countingIterator(0);
	// create a transform iterator that applies the GravitationalEnergyCombination functor
	GravitationalEnergyCombination gravitationalEnergyCombination(devZ, devZp);
	auto transformIterator = thrust::make_transform_iterator(countingIterator, gravitationalEnergyCombination);
	// get temporary storage size
	cub::DeviceReduce::Sum(this->tempStorage, this->tempStorageBytes, transformIterator, this->devEnergy, N);
	// allocate temporary storage
	cudaMalloc(&this->tempStorage, this->tempStorageBytes);
	// perform the reduction
	cub::DeviceReduce::Sum(this->tempStorage, this->tempStorageBytes, transformIterator, this->devEnergy, N);
}

template<int N>
void SurfaceEnergy<N>::CalculateEnergy(std_complex* devZp)
{
	thrust::counting_iterator<int> countingIterator(0);
	// create a transform iterator that applies the SurfaceEnergyCombination functor
	SurfaceEnergyCombination surfaceEnergyCombination(devZp);
	auto transformIterator = thrust::make_transform_iterator(countingIterator, surfaceEnergyCombination);
	// get temporary storage size
	cub::DeviceReduce::Sum(this->tempStorage, this->tempStorageBytes, transformIterator, this->devEnergy, N);
	// allocate temporary storage
	cudaMalloc(&this->tempStorage, this->tempStorageBytes);
	// perform the reduction
	cub::DeviceReduce::Sum(this->tempStorage, this->tempStorageBytes, transformIterator, this->devEnergy, N);
}

template<int N>
void VolumeFlux<N>::CalculateEnergy(std_complex* devZp, std_complex* velocities)
{
	thrust::counting_iterator<int> countingIterator(0);
	// create a transform iterator that applies the VolumeFluxCombination functor
	VolumeFluxCombination volumeFluxCombination(devZp, velocities);
	auto transformIterator = thrust::make_transform_iterator(countingIterator, volumeFluxCombination);
	// get temporary storage size
	cub::DeviceReduce::Sum(this->tempStorage, this->tempStorageBytes, transformIterator, this->devEnergy, N);
	// allocate temporary storage
	cudaMalloc(&this->tempStorage, this->tempStorageBytes);
	// perform the reduction
	cub::DeviceReduce::Sum(this->tempStorage, this->tempStorageBytes, transformIterator, this->devEnergy, N);
}