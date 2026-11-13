#pragma once
#ifndef DEVICE_PARTICLE_LOADING_H
#define DEVICE_PARTICLE_LOADING_H

#include "cuda_runtime.h"
#include <complex>
#include <cstdio>
#include <vector>

#include "constants.cuh"
#include "ExportTypes.cuh"
#include "SimulationOptions.h"
#include "Common/CudaChecks.cuh"

// Host-to-device loading of particle positions and potentials (devZ: N positions, then N potentials).
// Free with freeDeviceData.
inline int loadDataToDevice(const double* x, const double* y, const double* phi, DeviceParticleData& deviceData, const size_t N, const size_t batchSize = 1, cudaStream_t stream = cudaStreamPerThread)
{
    cudaError_t cudaStatus;
    // Allocate GPU buffers for three vectors (two input, one output)    .
	//std::cout << "Allocating device memory..." << std::endl;
    cudaStatus = cudaMallocAsync(&deviceData.devZ, 2 * N * batchSize * sizeof(std_complex), stream); // we need space for both positions and potentials
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return cudaStatus;
    }
	//std::cout << "Device memory allocated." << std::endl;
    // Copy input vectors from host memory to GPU buffers.
    std::vector<std_complex> ZHost(batchSize * N);
	std::vector<std_complex> PhiHost(batchSize * N);

	//std::cout << "Loading data to host complex vectors..." << std::endl;
    for (size_t i = 0; i < batchSize * N; ++i)
    {
        ZHost[i] = std_complex(x[i], y[i]);
		PhiHost[i] = std_complex(phi[i], 0.0);
    }

    cudaStatus = cudaMemcpyAsync(deviceData.devZ, ZHost.data(), batchSize * N * sizeof(std_complex), cudaMemcpyHostToDevice, stream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return cudaStatus;
    }

    // copy to device
    cudaStatus = cudaMemcpyAsync(deviceData.devZ + batchSize * N, PhiHost.data(), batchSize * N * sizeof(std_complex), cudaMemcpyHostToDevice, stream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return cudaStatus;
    }
	//std::cout << "Data loaded to device." << std::endl;
    //cudaDeviceSynchronize();
    //std::vector<std_complex> checkZ(2 * N);
    //cudaStatus = cudaMemcpy(checkZ.data(), deviceData.devZ, 2 * N * sizeof(std_complex), cudaMemcpyDeviceToHost);
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaMemcpy failed!");
    //    return cudaStatus;
    //}

    //std::vector<double> X(N);
    //std::vector<double> Y(N);
    //std::vector<double> Phi(N);
    //std::vector<double> PhiImag(N);

    //for (size_t i = 0; i < N; ++i)
    //{
    //    X[i] = checkZ[i].real();
    //    Y[i] = checkZ[i].imag();
    //    Phi[i] = checkZ[i + N].real();
    //    PhiImag[i] = checkZ[i + N].imag(); // should be 0
    //}
    return 0;
}

inline int loadDataToDevice(const c_double* Z, const c_double* phi, DeviceParticleData& deviceData, size_t N, cudaStream_t stream = cudaStreamPerThread)
{
    cudaError_t cudaStatus;
    // Allocate GPU buffers for three vectors (two input, one output)    .
    checkCuda(cudaMallocAsync(&deviceData.devZ, 2 * N * sizeof(std_complex), stream)); // we need space for both positions and potentials
    checkCuda(cudaMemcpyAsync(deviceData.devZ, Z, N * sizeof(std_complex), cudaMemcpyHostToDevice, stream));

    checkCuda(cudaMemcpyAsync(deviceData.devZ + N, phi, N * sizeof(std_complex), cudaMemcpyHostToDevice, stream));
	
    /*cudaDeviceSynchronize();
	std::vector<c_double> ZHost(2 * N);
	checkCuda(cudaMemcpy(ZHost.data(), deviceData.devZ, 2 * N * sizeof(c_double), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        if (Z[i].re != ZHost[i].re || Z[i].im != ZHost[i].im)
        {
			std::cout << "Mismatch at " << i << ": " << Z[i].re << " != " << ZHost[i].re << std::endl;
			std::cout << "Mismatch at " << i << ": " << Z[i].im << " != " << ZHost[i].im << std::endl;
        }
        if (phi[i].re != ZHost[i + N].re || phi[i].im != ZHost[i + N].im)
        {
			std::cout << "Mismatch at " << i << ": " << phi[i].re << " != " << ZHost[i + N].re << std::endl;
			std::cout << "Mismatch at " << i << ": " << phi[i].im << " != " << ZHost[i + N].im << std::endl;
        }
    }*/

	return 0;
}

inline int loadDataToDevice(ParticleData& data, DeviceParticleData& deviceData, const size_t N) 
{
    cudaError_t cudaStatus;
    // Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc(&deviceData.devZ, 2 * N * sizeof(std_complex)); // we need space for both positions and potentials
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return cudaStatus;
    }
    // Copy input vectors from host memory to GPU buffers.
	std::vector<std_complex> ZHost(N);
    for (size_t i = 0; i < N; ++i)
    {
		ZHost[i] = std_complex(data.Z[i].real(), data.Z[i].imag());
    }

    cudaStatus = cudaMemcpy(deviceData.devZ,  ZHost.data(), N * sizeof(std_complex), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return cudaStatus;
    }

	// copy the potentials
	std::vector<std_complex> potentialComplex(N);
    for (size_t i = 0; i < N; ++i) 
    {
		potentialComplex[i] = std_complex(data.Potential[i], 0.0);
    }
	// copy to device
	cudaStatus = cudaMemcpy(deviceData.devZ + N, potentialComplex.data(), N * sizeof(std_complex), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return cudaStatus;
    }
	cudaDeviceSynchronize();
	std::vector<std_complex> checkZ(2*N);
	cudaStatus = cudaMemcpy(checkZ.data(), deviceData.devZ, 2 * N * sizeof(std_complex), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return cudaStatus;
	}

	//std::vector<double> X(N);
	//std::vector<double> Y(N);
	//std::vector<double> Phi(N);
	//std::vector<double> PhiImag(N);

 //   for (size_t i = 0; i < N; ++i)
 //   {
	//	X[i] = checkZ[i].real();
	//	Y[i] = checkZ[i].imag();
	//	Phi[i] = checkZ[i + N].real();
	//	PhiImag[i] = checkZ[i + N].imag(); // should be 0
 //   }
    return 0;
}


inline int freeDeviceData(DeviceParticleData& deviceData) 
{
    cudaError_t cudaStatus;
    cudaStatus = cudaFree(deviceData.devZ);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaFree failed!");
        return cudaStatus;
    }
    return 0;
}

#endif // DEVICE_PARTICLE_LOADING_H
