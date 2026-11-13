#pragma once
#ifndef MATRIX_SOLVER_H
#define MATRIX_SOLVER_H

#include <cusolverDn.h>
#include <array>
#include <stdexcept>
#include <string>
#include <vector>
#include "Core/CudaChecks.cuh"

/// <summary>
/// Solves a systeme M a = b for a where M is an NxN matrix of real components, b reals, a reals
/// </summary>
/// <typeparam name="N">Size of the matrix</typeparam>
template <int N, size_t batchSize>
class MatrixSolver final
{
public:
    /// <summary>
    /// Constructs a MatrixSolver object and initializes cuSolver resources.
    /// </summary>
    MatrixSolver();

    /// <summary>
    /// Releases cuSolver resources and device memory.
    /// </summary>
    ~MatrixSolver();

    /// <summary>
    /// Solves a matrix system devM * deva = devb. devM, and devb are known. All the pointers are assumed to be device pointers on the gpu!
    /// </summary>
    /// <param name="devM">Device pointer to the batchSize x NxN matrices M (column-major - cublas compatible, double precision)</param>
    /// <param name="devb">Device pointer to the right-hand side vector b (length batchSize x N, double precision)</param>
    /// <param name="deva">Device pointer to the solution vector a (length batchSize x N, double precision, output)</param>
    void solve(double* devM, double* devb, double* deva);
    void setStream(cudaStream_t stream) {
		this->stream = stream;
        CUSH_CHECK(cusolverDnSetStream(handle, stream));
		CUSH_CHECK(cublasSetStream(blas, stream));
    }
private:
    cublasHandle_t blas;
    cudaStream_t stream = cudaStreamLegacy;
    cusolverDnHandle_t handle; ///< cuSolver handle for managing solver context
    void checkSolverInfo(const char* operation);
    int* devPivot;             ///< Device pointer for pivot indices (length N)
    int* devInfo;              ///< Device pointer for solver info (length 1)
    double* devWork;           ///< Device pointer for workspace memory
    int work_size = -1;        ///< Size of the workspace in bytes

	// for batch processing
	double** devMarray = nullptr;  ///< Device pointer to array of pointers to matrices M for batched operations
	double** devbarray = nullptr;  ///< Device pointer to array of pointers to vectors b for batched operations

	int* devPivotArray = nullptr; ///< Device pointer for array of pivot indices for batched operations
	int* devInfoArray = nullptr;  ///< Device pointer for solver info for batched operations

	std::array<int, batchSize> hostInfoArray; ///< Host array for solver info for batched operations
};

/// <summary>
/// Constructs a MatrixSolver object, initializes cuSolver handle, allocates device memory for pivot and info,
/// and queries the required workspace size for LU factorization.
/// </summary>
template <int N, size_t batchSize>
MatrixSolver<N, batchSize>::MatrixSolver()  
{  
    // Initialize cuSolver handle  
    CUSH_CHECK(cusolverDnCreate(&handle));  
    CUSH_CHECK(cublasCreate(&blas));
    // Allocate device memory for pivot and info  
    CUSH_CHECK(cudaMalloc(&devPivot, N * sizeof(int)));
    CUSH_CHECK(cudaMalloc(&devInfo, sizeof(int)));

    // Query working space size for LU factorization  
    CUSH_CHECK(cusolverDnDgetrf_bufferSize(handle, N, N, nullptr, N, &work_size));  
	// Allocate device memory for workspace
    CUSH_CHECK(cudaMalloc(&devWork, work_size * sizeof(double)));

    CUSH_CHECK(cudaMalloc(&devMarray, batchSize * sizeof(double*)));
    CUSH_CHECK(cudaMalloc(&devbarray, batchSize * sizeof(double*)));
    CUSH_CHECK(cudaMalloc(&devPivotArray, batchSize * N * sizeof(int)));
    CUSH_CHECK(cudaMalloc(&devInfoArray, batchSize * sizeof(int)));

	// Match surrounding kernel launches that use the default stream.
    this->setStream(cudaStreamLegacy);
}

/// <summary>
/// Destructor for MatrixSolver. Frees cuSolver handle and device memory for pivot, info, and workspace.
/// </summary>
template <int N, size_t batchSize>
MatrixSolver<N, batchSize>::~MatrixSolver()
{
    // Free device memory and cuSolver handle if allocated
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

template<int N, size_t batchSize>
inline void MatrixSolver<N, batchSize>::checkSolverInfo(const char* operation)
{
    int hostInfo = 0;
    CUSH_CHECK(cudaMemcpyAsync(&hostInfo, devInfo, sizeof(int), cudaMemcpyDeviceToHost, this->stream));
    CUSH_CHECK(cudaStreamSynchronize(this->stream));

    if (hostInfo < 0) {
        cush::throwNumericalError("cuSOLVER", hostInfo, std::string(operation) + ": argument " + std::to_string(-hostInfo) + " had an illegal value");
    }
    if (hostInfo > 0) {
        cush::throwNumericalError("cuSOLVER", hostInfo, std::string(operation) + ": matrix is singular at U(" + std::to_string(hostInfo) + "," + std::to_string(hostInfo) + ")");
    }
}

/// <summary>
/// Solves the linear system devM * deva = devb using LU factorization and substitution.
/// devM and devb are device pointers. The solution is written to deva (device pointer).
/// This will make a copy of devb into deva if they are different pointers to avoid overwriting devb since cusolverDnDgetrs will overwrite devb with the solution.
/// </summary>
/// <param name="devM">Device pointer to the NxN matrix M</param>
/// <param name="devb">Device pointer to the right-hand side vector b</param>
/// <param name="deva">Device pointer to the solution vector a (output)</param>
template<int N, size_t batchSize>
inline void MatrixSolver<N, batchSize>::solve(double* devM, double* devb, double* deva)
{
    if (deva != devb) {
        CUSH_CHECK(cudaMemcpyAsync(deva, devb, batchSize * N * sizeof(double), cudaMemcpyDeviceToDevice, this->stream)); // Copy b to a if they are different this avoids overwriting b if deva is a different pointer than devb.
        
		//std::cout << "Copied devb to deva to avoid overwriting devb." << std::endl;
    }
    if (batchSize == 1) {
		// use cuSolver to perform LU factorization of M if batchSize is 1
        CUSH_CHECK(cusolverDnDgetrf(handle, N, N, devM, N, devWork, devPivot, devInfo));
        checkSolverInfo("cusolverDnDgetrf");
        // Solve: LU * x = b -> a = x
        CUSH_CHECK(cusolverDnDgetrs(handle, CUBLAS_OP_N, N, 1, devM, N, devPivot, deva, N, devInfo)); // Use the matrix as-is (no transpose) CUBLAS_OP_N
        checkSolverInfo("cusolverDnDgetrs");
    }
    else {
        

        std::vector<double*> hMptrs(batchSize), hbptrs(batchSize);
        for (int i = 0; i < batchSize; ++i) {
            hMptrs[i] = devM + i * N * N;
            hbptrs[i] = deva + i * N;
        }
		//std::cout << "Preparing to solve batch of size " << batchSize << " with matrix size " << N << "x" << N << std::endl;
        CUSH_CHECK(cudaMemcpyAsync(devMarray, hMptrs.data(), batchSize * sizeof(double*), cudaMemcpyHostToDevice, this->stream));
        CUSH_CHECK(cudaMemcpyAsync(devbarray, hbptrs.data(), batchSize * sizeof(double*), cudaMemcpyHostToDevice, this->stream));
        
		// LU Factorization
		CUSH_CHECK(cublasDgetrfBatched(blas, N, devMarray, N, devPivotArray, devInfoArray, batchSize));
		//std::cout << "LU factorization completed for batch of size " << batchSize << std::endl;
        // check info array
		CUSH_CHECK(cudaMemcpyAsync((void*)hostInfoArray.data(), devInfoArray, batchSize * sizeof(int), cudaMemcpyDeviceToHost, this->stream));

		CUSH_CHECK(cudaStreamSynchronize(this->stream)); // ensure the info array is ready before checking it

        for(int i =0; i < batchSize; ++i) {
            // check info for each batch
			int hInfo = hostInfoArray[i];
            if (hInfo != 0) {
                cush::throwNumericalError("cuBLAS", hInfo, "getrfBatched: LU factorization failed in batch " + std::to_string(i));
            }
		}
		//std::cout << "LU factorization successful for all batches." << std::endl;
		//// print addresses of the matrices and vectors being solved for debugging
		//std::cout << "devMarray addresses: ";
		//std::cout << devMarray << std::endl;
		//std::cout << "devbarray addresses: ";
		//std::cout << devbarray << std::endl;
		//std::cout << "devPivotArray address: " << devPivotArray << std::endl;
		//std::cout << "devInfoArray address: " << devInfoArray << std::endl;

		// Solve the systems
		CUSH_CHECK(cublasDgetrsBatched(blas, CUBLAS_OP_N, N, 1, devMarray, N, devPivotArray, devbarray, N, hostInfoArray.data(), batchSize));
        //std::cout << "Solved batch of size " << batchSize << std::endl;
		// check info array again
        if(hostInfoArray[0] != 0) {
            cush::throwNumericalError("cuBLAS", hostInfoArray[0], "getrsBatched: argument " + std::to_string(-hostInfoArray[0]) + " had an illegal value");
	    }
    }
    
}
#endif // !MATRIX_SOLVER_H
