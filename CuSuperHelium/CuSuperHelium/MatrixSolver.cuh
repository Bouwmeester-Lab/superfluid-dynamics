#pragma once
#ifndef MATRIX_SOLVER_H
#define MATRIX_SOLVER_H


#include <cusolverDn.h>
#include "utilities.cuh"

/// <summary>
/// Solves a systeme M a = b for a where M is an NxN matrix of real components, b reals, a reals
/// </summary>
/// <typeparam name="N">Size of the matrix</typeparam>
template <int N>
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
    /// <param name="devM">Device pointer to the NxN matrix M (row-major, double precision)</param>
    /// <param name="devb">Device pointer to the right-hand side vector b (length N, double precision)</param>
    /// <param name="deva">Device pointer to the solution vector a (length N, double precision, output)</param>
    void solve(double* devM, double* devb, double* deva);

private:
    cusolverDnHandle_t handle; ///< cuSolver handle for managing solver context
    int* devPivot;             ///< Device pointer for pivot indices (length N)
    int* devInfo;              ///< Device pointer for solver info (length 1)
    double* devWork;           ///< Device pointer for workspace memory
    int work_size = -1;        ///< Size of the workspace in bytes
};

/// <summary>
/// Constructs a MatrixSolver object, initializes cuSolver handle, allocates device memory for pivot and info,
/// and queries the required workspace size for LU factorization.
/// </summary>
template <int N>  
MatrixSolver<N>::MatrixSolver()  
{  
    // Initialize cuSolver handle  
    checkCusolver(cusolverDnCreate(&handle));  

    // Allocate device memory for pivot and info  
    checkCuda(cudaMalloc(&devPivot, N * sizeof(int)));  
    checkCuda(cudaMalloc(&devInfo, sizeof(int)));  

    // Query working space size for LU factorization  
    checkCusolver(cusolverDnDgetrf_bufferSize(handle, N, N, nullptr, N, &work_size));  
	// Allocate device memory for workspace
    checkCuda(cudaMalloc(&devWork, work_size * sizeof(double)));
}

/// <summary>
/// Destructor for MatrixSolver. Frees cuSolver handle and device memory for pivot, info, and workspace.
/// </summary>
template <int N>
MatrixSolver<N>::~MatrixSolver()
{
    // Free device memory and cuSolver handle if allocated
    if (devPivot) cudaFree(devPivot);
    if (devInfo) cudaFree(devInfo);
    if (devWork) cudaFree(devWork);
    if (handle) cusolverDnDestroy(handle);
}

/// <summary>
/// Solves the linear system devM * deva = devb using LU factorization and substitution.
/// devM and devb are device pointers. The solution is written to deva (device pointer).
/// This will make a copy of devb into deva if they are different pointers to avoid overwriting devb since cusolverDnDgetrs will overwrite devb with the solution.
/// </summary>
/// <param name="devM">Device pointer to the NxN matrix M</param>
/// <param name="devb">Device pointer to the right-hand side vector b</param>
/// <param name="deva">Device pointer to the solution vector a (output)</param>
template<int N>
inline void MatrixSolver<N>::solve(double* devM, double* devb, double* deva)
{
    checkCusolver(cusolverDnDgetrf(handle, N, N, devM, N, devWork, devPivot, devInfo));
    if (deva != devb) 
    {
		cudaMemcpy(deva, devb, N * sizeof(double), cudaMemcpyDeviceToDevice); // Copy b to a if they are different this avoids overwriting b if deva is a different pointer than devb. 
        // If deva is the same as devb, this is a no-op since cusolverDnDgetrs will overwrite deva with the solution.
	}
    // Solve: LU * x = b -> a = x
    checkCusolver(cusolverDnDgetrs(handle, CUBLAS_OP_N, N, 1, devM, N, devPivot, deva, N, devInfo)); // Use the matrix as-is (no transpose) CUBLAS_OP_N
}
#endif // !MATRIX_SOLVER_H