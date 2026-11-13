#pragma once
#ifndef DEVICE_PARTICLE_LOADING_H
#define DEVICE_PARTICLE_LOADING_H

#include "cuda_runtime.h"
#include <complex>
#include <vector>

#include "Core/constants.cuh"
#include "Core/CudaChecks.cuh"
#include "Core/DeviceBuffer.cuh"
#include "Export/ExportTypes.cuh"
#include "Physics/SimulationOptions.h"

// Host-to-device loading of particle positions and potentials into deviceData.buffer:
// batchSize * N complex positions (x + i y), then batchSize * N potentials (phi + 0 i).
// The buffer frees itself when deviceData goes out of scope. Errors throw cush::CudaError.

inline int loadDataToDevice(const double* x, const double* y, const double* phi, DeviceParticleData& deviceData, const size_t N, const size_t batchSize = 1, cudaStream_t stream = cudaStreamPerThread)
{
	deviceData.buffer = DeviceBuffer<std_complex>(2 * N * batchSize);

	std::vector<std_complex> ZHost(batchSize * N);
	std::vector<std_complex> PhiHost(batchSize * N);
	for (size_t i = 0; i < batchSize * N; ++i)
	{
		ZHost[i] = std_complex(x[i], y[i]);
		PhiHost[i] = std_complex(phi[i], 0.0);
	}

	CUSH_CHECK(cudaMemcpyAsync(deviceData.devZ(), ZHost.data(), batchSize * N * sizeof(std_complex), cudaMemcpyHostToDevice, stream));
	CUSH_CHECK(cudaMemcpyAsync(deviceData.devZ() + batchSize * N, PhiHost.data(), batchSize * N * sizeof(std_complex), cudaMemcpyHostToDevice, stream));
	return 0;
}

inline int loadDataToDevice(const c_double* Z, const c_double* phi, DeviceParticleData& deviceData, size_t N, cudaStream_t stream = cudaStreamPerThread)
{
	deviceData.buffer = DeviceBuffer<std_complex>(2 * N);
	CUSH_CHECK(cudaMemcpyAsync(deviceData.devZ(), Z, N * sizeof(std_complex), cudaMemcpyHostToDevice, stream));
	CUSH_CHECK(cudaMemcpyAsync(deviceData.devZ() + N, phi, N * sizeof(std_complex), cudaMemcpyHostToDevice, stream));
	return 0;
}

inline int loadDataToDevice(ParticleData& data, DeviceParticleData& deviceData, const size_t N)
{
	deviceData.buffer = DeviceBuffer<std_complex>(2 * N);

	std::vector<std_complex> ZHost(N);
	std::vector<std_complex> potentialComplex(N);
	for (size_t i = 0; i < N; ++i)
	{
		ZHost[i] = std_complex(data.Z[i].real(), data.Z[i].imag());
		potentialComplex[i] = std_complex(data.Potential[i], 0.0);
	}

	CUSH_CHECK(cudaMemcpy(deviceData.devZ(), ZHost.data(), N * sizeof(std_complex), cudaMemcpyHostToDevice));
	CUSH_CHECK(cudaMemcpy(deviceData.devZ() + N, potentialComplex.data(), N * sizeof(std_complex), cudaMemcpyHostToDevice));
	CUSH_CHECK(cudaDeviceSynchronize());
	return 0;
}

/// <summary>
/// Frees the device memory early. Optional: the buffer is also freed when deviceData is destroyed.
/// </summary>
inline int freeDeviceData(DeviceParticleData& deviceData)
{
	deviceData.buffer.reset();
	return 0;
}

#endif // DEVICE_PARTICLE_LOADING_H
