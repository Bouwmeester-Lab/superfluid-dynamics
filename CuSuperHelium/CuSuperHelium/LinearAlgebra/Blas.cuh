#pragma once
#ifndef CUSH_BLAS_H
#define CUSH_BLAS_H

#include <cublas_v2.h>
#include "Core/cuDoubleComplexOperators.cuh"

/// <summary>
/// Type-generic wrappers around cuBLAS level-1 calls, so templated code calls one name for every state type.
/// Each overload is a direct call to the matching cuBLAS routine, unit stride.
/// </summary>
namespace cush::blas
{
	/// y += alpha * x
	inline cublasStatus_t axpy(cublasHandle_t handle, int n, const double* alpha, const double* x, double* y)
	{
		return cublasDaxpy(handle, n, alpha, x, 1, y, 1);
	}

	inline cublasStatus_t axpy(cublasHandle_t handle, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, cuDoubleComplex* y)
	{
		return cublasZaxpy(handle, n, alpha, x, 1, y, 1);
	}

	inline cublasStatus_t axpy(cublasHandle_t handle, int n, const std_complex* alpha, const std_complex* x, std_complex* y)
	{
		return cublasZaxpy(handle, n, reinterpret_cast<const cuDoubleComplex*>(alpha), reinterpret_cast<const cuDoubleComplex*>(x), 1, reinterpret_cast<cuDoubleComplex*>(y), 1);
	}
}

#endif // !CUSH_BLAS_H
