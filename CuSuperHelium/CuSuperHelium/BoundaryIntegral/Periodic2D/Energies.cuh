#pragma once
#include "BoundaryIntegral/Periodic2D/InterfaceParameters.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include "Core/constants.cuh"
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include "Core/DevicePointers.cuh"

class EnergyBase
{
public:
	EnergyBase(InterfaceParameters& properties, int n);
	virtual ~EnergyBase();
	double getEnergy() const;
	virtual void CalculateEnergy(const DevicePointers& devPoints, cudaStream_t stream = cudaStreamPerThread) = 0;
protected:
	InterfaceParameters& properties;
	const int N; ///< Number of points summed by the reduction
	double* devEnergy;

	void* tempStorage = nullptr; // Temporary storage for CUB operations, this is a device pointer
	size_t tempStorageBytes = 0;

	virtual double scaleEnergy(double energy) const = 0; // Pure virtual function to scale energy
	
};

class EnergyContainer {
public:
	EnergyBase* kineticEnergy;
	EnergyBase* potentialEnergy;
	EnergyBase* surfaceEnergy;

	void CalculateEnergy(const DevicePointers& devPoints, cudaStream_t stream) {
		kineticEnergy->CalculateEnergy(devPoints, stream);
		potentialEnergy->CalculateEnergy(devPoints, stream);
		surfaceEnergy->CalculateEnergy(devPoints, stream);
	}

	EnergyContainer(EnergyBase* kinetic, EnergyBase* potential, EnergyBase* surface)
		: kineticEnergy(kinetic), potentialEnergy(potential), surfaceEnergy(surface) {
	}
	~EnergyContainer() {
		delete kineticEnergy;
		delete potentialEnergy;
		delete surfaceEnergy;
	}
};

class KineticEnergy : public EnergyBase
{
public:
	using EnergyBase::EnergyBase; // Inherit constructor from EnergyBase
	

	void CalculateEnergy(const std_complex* devPhi, const std_complex* devZ, const std_complex* devZp, const std_complex* velocitiesLower, cudaStream_t stream);
	virtual double scaleEnergy(double energy) const override
	{
		return energy * 0.25 / PI_d;
	}
	virtual void CalculateEnergy(const DevicePointers& devPoints, cudaStream_t stream) override
	{
		CalculateEnergy(devPoints.Phi, devPoints.Z, devPoints.Zp, devPoints.LowerVelocities, stream);
	}
};

class GravitationalEnergy : public EnergyBase
{
public:
	using EnergyBase::EnergyBase; // Inherit constructor from EnergyBase
	void CalculateEnergy(const std_complex* devZ, const std_complex* devZp, cudaStream_t stream);
	virtual double scaleEnergy(double energy) const override
	{
		return energy * 0.25 * (1.0 + this->properties.rho) / PI_d;
	}
	virtual void CalculateEnergy(const DevicePointers& devPoints, cudaStream_t stream) override
	{
		this->CalculateEnergy(devPoints.Z, devPoints.Zp, stream);
	}
};

class VanDerWaalsEnergy : public EnergyBase
{
public:
	using EnergyBase::EnergyBase; // Inherit constructor from EnergyBase
	void CalculateEnergy(const std_complex* devZ, cudaStream_t stream);
	virtual double scaleEnergy(double energy) const override
	{
		return energy * cuda::std::pow(this->properties.depth, 2) / 6.0;
	}
	virtual void CalculateEnergy(const DevicePointers& devPoints, cudaStream_t stream) override
	{
		this->CalculateEnergy(devPoints.Z, stream);
	}
};

class SurfaceEnergy : public EnergyBase
{
public:
	using EnergyBase::EnergyBase; // Inherit constructor from EnergyBase
	void CalculateEnergy(const std_complex* devZp, cudaStream_t stream);
	virtual double scaleEnergy(double energy) const override
	{
		return (energy - 2.0 * PI_d) * this->properties.kappa / (2.0 * PI_d);
	}
	virtual void CalculateEnergy(const DevicePointers& devPoints, cudaStream_t stream) override
	{
		this->CalculateEnergy(devPoints.Zp, stream);
	}
};

class VolumeFlux : public EnergyBase
{
public:
	using EnergyBase::EnergyBase; // Inherit constructor from EnergyBase
	void CalculateEnergy(const std_complex* devZp, const std_complex* velocities, cudaStream_t stream);
	virtual double scaleEnergy(double energy) const override
	{
		return energy * 0.5 / PI_d;
	}
	virtual void CalculateEnergy(const DevicePointers& devPoints, cudaStream_t stream) override
	{
		this->CalculateEnergy(devPoints.Zp, devPoints.LowerVelocities, stream);
	}
};


struct KineticEnergyCombination
{
	const std_complex* Z;
	const std_complex* Phi;
	const std_complex* Zp;
	const std_complex* velocitiesLower;
	const std_complex* velocitiesUpper;
	const InterfaceParameters properties;

	__host__ __device__ KineticEnergyCombination(const std_complex* z, const std_complex* phi, const std_complex* zp, const std_complex* velocitiesLower, const std_complex* velocitiesUpper, const InterfaceParameters properties)
		: Z(z), Phi(phi), Zp(zp), velocitiesLower(velocitiesLower), velocitiesUpper(velocitiesUpper), properties(properties) {
	}

	__device__ double operator()(int k) const
	{
		return (Phi[k].real() + 0.5 * properties.U * (1.0 + properties.rho) * Z[k].real()) * (-1.0 * Zp[k].imag() * velocitiesLower[k].real() + Zp[k].real() * velocitiesLower[k].imag()) \
			- 0.5 * properties.U * ((velocitiesLower[k].real() + properties.rho * velocitiesUpper[k].real()) * Zp[k].real() + (velocitiesLower[k].imag() + properties.rho * velocitiesUpper[k].imag()) * Zp[k].imag() \
				+ 0.5 * properties.U * (1.0 - properties.rho) * Zp[k].real()) * Z[k].imag();
	}
};

struct VanDerWaalsEnergyCombination
{
	const std_complex* Z;
	const InterfaceParameters properties;
	__host__ __device__ VanDerWaalsEnergyCombination(const std_complex* z, const InterfaceParameters properties)
		: Z(z), properties(properties) {
	}
	__device__ double operator()(int k) const
	{
		return (1 / cuda::std::pow(1 + Z[k].imag() / properties.depth, 2) - 1.0);
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

inline EnergyBase::EnergyBase(InterfaceParameters& properties, int n) : properties(properties), N(n)
{
	CUSH_CHECK(cudaMalloc(&devEnergy, sizeof(double)));
}

inline EnergyBase::~EnergyBase()
{
	CUSH_CHECK_NOTHROW(cudaGetLastError());

	if(devEnergy) CUSH_CHECK_NOTHROW(cudaFree(devEnergy));
	CUSH_CHECK_NOTHROW(cudaGetLastError());

	if(tempStorage) CUSH_CHECK_NOTHROW(cudaFree(tempStorage));
	CUSH_CHECK_NOTHROW(cudaGetLastError());
	/*auto error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("CUDA Error in EnergyBase destructor: %s\n", cudaGetErrorString(error));
	}*/
}

inline double EnergyBase::getEnergy() const
{
	CUSH_CHECK(cudaDeviceSynchronize());
	double e;
	CUSH_CHECK(cudaMemcpy(&e, devEnergy, sizeof(double), cudaMemcpyDeviceToHost));
	return this->scaleEnergy(e);
}

inline void KineticEnergy::CalculateEnergy(const std_complex* devPhi, const std_complex* devZ, const std_complex* devZp, const std_complex* velocitiesLower, cudaStream_t stream)
{
	thrust::counting_iterator<int> countingIterator(0);

	// create a transform iterator that applies the KineticEnergyCombination functor
	KineticEnergyCombination kineticEnergyCombination(devZ, devPhi, devZp, velocitiesLower, velocitiesLower, this->properties);

	auto transformIterator = thrust::make_transform_iterator(countingIterator, kineticEnergyCombination);

	// get temporary storage size
	cub::DeviceReduce::Sum(this->tempStorage, this->tempStorageBytes, transformIterator, this->devEnergy, N);

	// allocate temporary storage
	CUSH_CHECK(cudaMalloc(&this->tempStorage, this->tempStorageBytes));

	// perform the reduction
	cub::DeviceReduce::Sum(this->tempStorage, this->tempStorageBytes, transformIterator, this->devEnergy, N);

	CUSH_CHECK(cudaFreeAsync(this->tempStorage, stream));
	this->tempStorage = nullptr;
}



inline void GravitationalEnergy::CalculateEnergy(const std_complex* devZ, const std_complex* devZp, cudaStream_t stream)
{
	thrust::counting_iterator<int> countingIterator(0);
	// create a transform iterator that applies the GravitationalEnergyCombination functor
	GravitationalEnergyCombination gravitationalEnergyCombination(devZ, devZp);
	auto transformIterator = thrust::make_transform_iterator(countingIterator, gravitationalEnergyCombination);
	// get temporary storage size
	cub::DeviceReduce::Sum(this->tempStorage, this->tempStorageBytes, transformIterator, this->devEnergy, N);
	// allocate temporary storage
	CUSH_CHECK(cudaMalloc(&this->tempStorage, this->tempStorageBytes));
	// perform the reduction
	cub::DeviceReduce::Sum(this->tempStorage, this->tempStorageBytes, transformIterator, this->devEnergy, N);
	CUSH_CHECK(cudaFreeAsync(this->tempStorage, stream));
	this->tempStorage = nullptr;
}

inline void VanDerWaalsEnergy::CalculateEnergy(const std_complex* devZ, cudaStream_t stream)
{
	thrust::counting_iterator<int> countingIterator(0);
	// create a transform iterator that applies the GravitationalEnergyCombination functor
	VanDerWaalsEnergyCombination vanDerWaalsEnergyCombination(devZ, this->properties);
	auto transformIterator = thrust::make_transform_iterator(countingIterator, vanDerWaalsEnergyCombination);
	// get temporary storage size
	cub::DeviceReduce::Sum(this->tempStorage, this->tempStorageBytes, transformIterator, this->devEnergy, N);
	// allocate temporary storage
	CUSH_CHECK(cudaMalloc(&this->tempStorage, this->tempStorageBytes));
	// perform the reduction
	cub::DeviceReduce::Sum(this->tempStorage, this->tempStorageBytes, transformIterator, this->devEnergy, N);
	CUSH_CHECK(cudaFreeAsync(this->tempStorage, stream));
	this->tempStorage = nullptr;
}

inline void SurfaceEnergy::CalculateEnergy(const std_complex* devZp, cudaStream_t stream)
{
	thrust::counting_iterator<int> countingIterator(0);
	// create a transform iterator that applies the SurfaceEnergyCombination functor
	SurfaceEnergyCombination surfaceEnergyCombination(devZp);
	auto transformIterator = thrust::make_transform_iterator(countingIterator, surfaceEnergyCombination);
	// get temporary storage size
	cub::DeviceReduce::Sum(this->tempStorage, this->tempStorageBytes, transformIterator, this->devEnergy, N);
	// allocate temporary storage
	CUSH_CHECK(cudaMalloc(&this->tempStorage, this->tempStorageBytes));
	// perform the reduction
	cub::DeviceReduce::Sum(this->tempStorage, this->tempStorageBytes, transformIterator, this->devEnergy, N);
	CUSH_CHECK(cudaFreeAsync(this->tempStorage, stream));
	this->tempStorage = nullptr;
}

inline void VolumeFlux::CalculateEnergy(const std_complex* devZp, const std_complex* velocities, cudaStream_t stream)
{
	thrust::counting_iterator<int> countingIterator(0);
	// create a transform iterator that applies the VolumeFluxCombination functor
	VolumeFluxCombination volumeFluxCombination(devZp, velocities);
	auto transformIterator = thrust::make_transform_iterator(countingIterator, volumeFluxCombination);
	// get temporary storage size
	cub::DeviceReduce::Sum(this->tempStorage, this->tempStorageBytes, transformIterator, this->devEnergy, N);
	// allocate temporary storage
	CUSH_CHECK(cudaMalloc(&this->tempStorage, this->tempStorageBytes));
	// perform the reduction
	cub::DeviceReduce::Sum(this->tempStorage, this->tempStorageBytes, transformIterator, this->devEnergy, N);
	CUSH_CHECK(cudaFreeAsync(this->tempStorage, stream));
	this->tempStorage = nullptr;
}