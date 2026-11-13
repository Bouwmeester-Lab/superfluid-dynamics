#pragma once
#ifndef DENSE_LU_SOLVER_H
#define DENSE_LU_SOLVER_H

#include <cusolverDn.h>
#include <stdexcept>
#include <string>

#include "Core/CudaChecks.cuh"

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
		CUSH_CHECK(cusolverDnCreate(&handle));
		CUSH_CHECK(cudaMalloc(&devPivot, n * sizeof(int)));
		CUSH_CHECK(cudaMalloc(&devInfo, sizeof(int)));
		CUSH_CHECK(cusolverDnDgetrf_bufferSize(handle, n, n, nullptr, n, &workSize));
		CUSH_CHECK(cudaMalloc(&devWork, workSize * sizeof(double)));
		CUSH_CHECK(cusolverDnSetStream(handle, stream));
	}

	~DenseLuSolver()
	{
		if (devPivot) CUSH_CHECK_NOTHROW(cudaFree(devPivot));
		if (devInfo) CUSH_CHECK_NOTHROW(cudaFree(devInfo));
		if (devWork) CUSH_CHECK_NOTHROW(cudaFree(devWork));
		if (handle) CUSH_CHECK_NOTHROW(cusolverDnDestroy(handle));
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
			CUSH_CHECK(cudaMemcpyAsync(deva, devb, n * sizeof(double), cudaMemcpyDeviceToDevice, stream));
		}
		CUSH_CHECK(cusolverDnDgetrf(handle, n, n, devM, n, devWork, devPivot, devInfo));
		checkInfo("cusolverDnDgetrf");

		CUSH_CHECK(cusolverDnDgetrs(handle, CUBLAS_OP_N, n, 1, devM, n, devPivot, deva, n, devInfo));
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
		CUSH_CHECK(cudaMemcpyAsync(&hostInfo, devInfo, sizeof(int), cudaMemcpyDeviceToHost, stream));
		CUSH_CHECK(cudaStreamSynchronize(stream));
		if (hostInfo < 0)
		{
			cush::throwNumericalError("cuSOLVER", hostInfo, std::string(operation) + ": argument " + std::to_string(-hostInfo) + " had an illegal value");
		}
		if (hostInfo > 0)
		{
			cush::throwNumericalError("cuSOLVER", hostInfo, std::string(operation) + ": matrix is singular at U(" + std::to_string(hostInfo) + "," + std::to_string(hostInfo) + ")");
		}
	}
};

#endif // DENSE_LU_SOLVER_H
