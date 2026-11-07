#pragma once
#ifndef DENSE_LU_SOLVER_H
#define DENSE_LU_SOLVER_H

#include <cusolverDn.h>
#include <stdexcept>
#include <string>

#include "CudaChecks.cuh"

/// <summary>
/// Solves a single dense system M a = b (n x n, column-major, double) with cuSOLVER LU factorization.
/// Same algorithm as MatrixSolver&lt;N, 1&gt; but with the size n chosen at runtime.
/// </summary>
class DenseLuSolver final
{
public:
	explicit DenseLuSolver(int n) : n(n)
	{
		if (n <= 0)
		{
			throw std::invalid_argument("DenseLuSolver requires a positive matrix size");
		}
		checkCusolver(cusolverDnCreate(&handle));
		CHECK_CUDA(cudaMalloc(&devPivot, n * sizeof(int)));
		CHECK_CUDA(cudaMalloc(&devInfo, sizeof(int)));
		checkCusolver(cusolverDnDgetrf_bufferSize(handle, n, n, nullptr, n, &workSize));
		CHECK_CUDA(cudaMalloc(&devWork, workSize * sizeof(double)));
		checkCusolver(cusolverDnSetStream(handle, stream));
	}

	~DenseLuSolver()
	{
		if (devPivot) cudaFree(devPivot);
		if (devInfo) cudaFree(devInfo);
		if (devWork) cudaFree(devWork);
		if (handle) cusolverDnDestroy(handle);
	}

	DenseLuSolver(const DenseLuSolver&) = delete;
	DenseLuSolver& operator=(const DenseLuSolver&) = delete;

	int size() const { return n; }

	/// <summary>
	/// Solves devM * deva = devb. devM is overwritten by its LU factors. devb is left untouched when deva != devb.
	/// All pointers are device pointers.
	/// </summary>
	void solve(double* devM, const double* devb, double* deva)
	{
		if (deva != devb)
		{
			CHECK_CUDA(cudaMemcpyAsync(deva, devb, n * sizeof(double), cudaMemcpyDeviceToDevice, stream));
		}
		const cusolverStatus_t getrfStatus = cusolverDnDgetrf(handle, n, n, devM, n, devWork, devPivot, devInfo);
		if (getrfStatus != CUSOLVER_STATUS_SUCCESS)
		{
			throw std::runtime_error("cusolverDnDgetrf failed with status " + std::to_string(static_cast<int>(getrfStatus)));
		}
		checkInfo("cusolverDnDgetrf");

		const cusolverStatus_t getrsStatus = cusolverDnDgetrs(handle, CUBLAS_OP_N, n, 1, devM, n, devPivot, deva, n, devInfo);
		if (getrsStatus != CUSOLVER_STATUS_SUCCESS)
		{
			throw std::runtime_error("cusolverDnDgetrs failed with status " + std::to_string(static_cast<int>(getrsStatus)));
		}
		checkInfo("cusolverDnDgetrs");
	}

private:
	int n;
	cudaStream_t stream = cudaStreamLegacy;
	cusolverDnHandle_t handle = nullptr;
	int* devPivot = nullptr;
	int* devInfo = nullptr;
	double* devWork = nullptr;
	int workSize = -1;

	void checkInfo(const char* operation)
	{
		int hostInfo = 0;
		CHECK_CUDA(cudaMemcpyAsync(&hostInfo, devInfo, sizeof(int), cudaMemcpyDeviceToHost, stream));
		CHECK_CUDA(cudaStreamSynchronize(stream));
		if (hostInfo < 0)
		{
			throw std::runtime_error(std::string(operation) + " failed: argument " + std::to_string(-hostInfo) + " had an illegal value");
		}
		if (hostInfo > 0)
		{
			throw std::runtime_error(std::string(operation) + " failed: matrix is singular at U(" + std::to_string(hostInfo) + "," + std::to_string(hostInfo) + ")");
		}
	}
};

#endif // DENSE_LU_SOLVER_H
