#pragma once
#ifndef CUDA_CHECKS_H
#define CUDA_CHECKS_H

#include "cuda_runtime.h"
#include <cublas_v2.h>
#include <cusolverDn.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

inline cudaError_t setDevice()
{
    cudaError_t cudaStatus;
    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return cudaStatus;
    }

    return cudaStatus;
}

inline void checkCudaErrors(std::string from)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error from " << from << ": " << cudaGetErrorString(err) << std::endl;
    }
}

inline void checkCuda(cudaError_t result)
{
    if (result != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(result) << std::endl;
        exit(EXIT_FAILURE);
    }
}

inline void checkCuda(cudaError_t result,
    const char* func,
    const char* file,
    int line)
{
    if (result != cudaSuccess) {
        std::cerr << "CUDA Error at " << file << ":" << line
            << " (" << func << "): "
            << cudaGetErrorString(result) << std::endl;
        throw std::runtime_error(cudaGetErrorString(result));
    }
}

#define CHECK_CUDA(val) checkCuda((val), #val, __FILE__, __LINE__)

inline void checkCusolver(cusolverStatus_t status)
{
    if (status != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "cuSolver Error: " << static_cast<int>(status) << std::endl;
        exit(EXIT_FAILURE);
    }
}

inline void checkCublas(cublasStatus_t status)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS Error: " << static_cast<int>(status) << std::endl;
        exit(EXIT_FAILURE);
    }
}

#endif
