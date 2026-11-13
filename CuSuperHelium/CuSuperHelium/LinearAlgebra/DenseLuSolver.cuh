#pragma once
#ifndef DENSE_LU_SOLVER_H
#define DENSE_LU_SOLVER_H

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <stdexcept>
#include <string>
#include <vector>

#include "Core/CudaChecks.cuh"

/// <summary>
/// Solves dense systems M a = b (n x n, column-major, double) by LU factorization, with n chosen at runtime.
/// batch == 1: one system, cuSOLVER getrf/getrs.
/// batch &gt; 1: `batch` independent systems stored back to back (matrices n*n apart, vectors n apart),
/// cuBLAS getrfBatched/getrsBatched.
/// </summary>
class DenseLuSolver final
{
public:
	explicit DenseLuSolver(int n, size_t batch = 1) : n(n), batch(batch), hostInfoArray(batch, 0)
	{
		if (n <= 0)
		{
			throw std::invalid_argument("DenseLuSolver requires a positive matrix size");
		}
		if (batch == 0)
		{
			throw std::invalid_argument("DenseLuSolver requires a positive batch size");
		}
		CUSH_CHECK(cusolverDnCreate(&handle));
		CUSH_CHECK(cudaMalloc(&devPivot, n * sizeof(int)));
		CUSH_CHECK(cudaMalloc(&devInfo, sizeof(int)));
		CUSH_CHECK(cusolverDnDgetrf_bufferSize(handle, n, n, nullptr, n, &workSize));
		CUSH_CHECK(cudaMalloc(&devWork, workSize * sizeof(double)));
		if (batch > 1)
		{
			CUSH_CHECK(cublasCreate(&blas));
			CUSH_CHECK(cudaMalloc(&devMarray, batch * sizeof(double*)));
			CUSH_CHECK(cudaMalloc(&devbarray, batch * sizeof(double*)));
			CUSH_CHECK(cudaMalloc(&devPivotArray, batch * n * sizeof(int)));
			CUSH_CHECK(cudaMalloc(&devInfoArray, batch * sizeof(int)));
		}
		// Match surrounding kernel launches that use the default stream.
		setStream(cudaStreamLegacy);
	}

	~DenseLuSolver()
	{
		if (devPivot) CUSH_CHECK_NOTHROW(cudaFree(devPivot));
		if (devInfo) CUSH_CHECK_NOTHROW(cudaFree(devInfo));
		if (devWork) CUSH_CHECK_NOTHROW(cudaFree(devWork));
		if (handle) CUSH_CHECK_NOTHROW(cusolverDnDestroy(handle));
		if (blas) CUSH_CHECK_NOTHROW(cublasDestroy(blas));
		if (devMarray) CUSH_CHECK_NOTHROW(cudaFree(devMarray));
		if (devbarray) CUSH_CHECK_NOTHROW(cudaFree(devbarray));
		if (devPivotArray) CUSH_CHECK_NOTHROW(cudaFree(devPivotArray));
		if (devInfoArray) CUSH_CHECK_NOTHROW(cudaFree(devInfoArray));
	}

	DenseLuSolver(const DenseLuSolver&) = delete;
	DenseLuSolver& operator=(const DenseLuSolver&) = delete;

	int size() const { return n; }
	size_t batchSize() const { return batch; }

	void setStream(cudaStream_t newStream)
	{
		stream = newStream;
		CUSH_CHECK(cusolverDnSetStream(handle, stream));
		if (blas) CUSH_CHECK(cublasSetStream(blas, stream));
	}

	/// <summary>
	/// Solves devM * deva = devb for every system of the batch. devM is overwritten by its LU factors.
	/// devb is left untouched when deva != devb. All pointers are device pointers.
	/// </summary>
	void solve(double* devM, const double* devb, double* deva)
	{
		if (deva != devb)
		{
			CUSH_CHECK(cudaMemcpyAsync(deva, devb, batch * n * sizeof(double), cudaMemcpyDeviceToDevice, stream));
		}
		if (batch == 1)
		{
			CUSH_CHECK(cusolverDnDgetrf(handle, n, n, devM, n, devWork, devPivot, devInfo));
			checkInfo("cusolverDnDgetrf");

			CUSH_CHECK(cusolverDnDgetrs(handle, CUBLAS_OP_N, n, 1, devM, n, devPivot, deva, n, devInfo));
			checkInfo("cusolverDnDgetrs");
			return;
		}
		solveBatched(devM, deva);
	}

private:
	int n;
	size_t batch;
	cudaStream_t stream = cudaStreamLegacy;
	cusolverDnHandle_t handle = nullptr;
	int* devPivot = nullptr;
	int* devInfo = nullptr;
	double* devWork = nullptr;
	int workSize = -1;

	// batched mode
	cublasHandle_t blas = nullptr;
	double** devMarray = nullptr;
	double** devbarray = nullptr;
	int* devPivotArray = nullptr;
	int* devInfoArray = nullptr;
	std::vector<int> hostInfoArray;

	void solveBatched(double* devM, double* deva)
	{
		std::vector<double*> hMptrs(batch), hbptrs(batch);
		for (size_t i = 0; i < batch; ++i)
		{
			hMptrs[i] = devM + i * n * n;
			hbptrs[i] = deva + i * n;
		}
		CUSH_CHECK(cudaMemcpyAsync(devMarray, hMptrs.data(), batch * sizeof(double*), cudaMemcpyHostToDevice, stream));
		CUSH_CHECK(cudaMemcpyAsync(devbarray, hbptrs.data(), batch * sizeof(double*), cudaMemcpyHostToDevice, stream));

		const int count = static_cast<int>(batch);
		CUSH_CHECK(cublasDgetrfBatched(blas, n, devMarray, n, devPivotArray, devInfoArray, count));
		CUSH_CHECK(cudaMemcpyAsync(hostInfoArray.data(), devInfoArray, batch * sizeof(int), cudaMemcpyDeviceToHost, stream));
		CUSH_CHECK(cudaStreamSynchronize(stream));
		for (size_t i = 0; i < batch; ++i)
		{
			if (hostInfoArray[i] != 0)
			{
				cush::throwNumericalError("cuBLAS", hostInfoArray[i], "getrfBatched: LU factorization failed in batch " + std::to_string(i));
			}
		}

		CUSH_CHECK(cublasDgetrsBatched(blas, CUBLAS_OP_N, n, 1, devMarray, n, devPivotArray, devbarray, n, hostInfoArray.data(), count));
		if (hostInfoArray[0] != 0)
		{
			cush::throwNumericalError("cuBLAS", hostInfoArray[0], "getrsBatched: argument " + std::to_string(-hostInfoArray[0]) + " had an illegal value");
		}
	}

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
