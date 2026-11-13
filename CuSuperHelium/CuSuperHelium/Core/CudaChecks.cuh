#pragma once
#ifndef CUDA_CHECKS_H
#define CUDA_CHECKS_H

// Error checking for CUDA and the CUDA libraries. The library never calls exit(): every failure
// throws cush::CudaError, and the exports catch it and return -1.
//
//   CUSH_CHECK(expr)          expr returns cudaError_t, cublasStatus_t, cusolverStatus_t or cufftResult;
//                             throws cush::CudaError unless it is the library's success value.
//   CUSH_CHECK_LAUNCH()       after a kernel launch: throws if the launch failed (cudaGetLastError).
//                             With CUSH_SYNC_CHECKS defined it also synchronizes the device, so errors
//                             raised while the kernel runs are reported here instead of at a later call.
//   CUSH_CHECK_NOTHROW(expr)  for destructors and other noexcept code: logs a failure to stderr instead
//                             of throwing.

#include "cuda_runtime.h"
#include <cublas_v2.h>
#include <cufft.h>
#include <cusolverDn.h>

#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>

namespace cush
{
	/// <summary>
	/// A failed CUDA, cuBLAS, cuSOLVER or cuFFT call, or a numerical failure reported by a library
	/// (for example a singular matrix in getrf). what() names the library, the status, the expression
	/// and the source location.
	/// </summary>
	class CudaError : public std::runtime_error
	{
	public:
		CudaError(const std::string& library, int code, const std::string& status, const std::string& message)
			: std::runtime_error(library + " error " + status + " (" + std::to_string(code) + "): " + message),
			  library_(library), code_(code) {}

		const std::string& library() const noexcept { return library_; }
		int code() const noexcept { return code_; }

	private:
		std::string library_;
		int code_;
	};

	namespace detail
	{
		inline const char* statusName(cusolverStatus_t s)
		{
			switch (s)
			{
			case CUSOLVER_STATUS_SUCCESS: return "CUSOLVER_STATUS_SUCCESS";
			case CUSOLVER_STATUS_NOT_INITIALIZED: return "CUSOLVER_STATUS_NOT_INITIALIZED";
			case CUSOLVER_STATUS_ALLOC_FAILED: return "CUSOLVER_STATUS_ALLOC_FAILED";
			case CUSOLVER_STATUS_INVALID_VALUE: return "CUSOLVER_STATUS_INVALID_VALUE";
			case CUSOLVER_STATUS_ARCH_MISMATCH: return "CUSOLVER_STATUS_ARCH_MISMATCH";
			case CUSOLVER_STATUS_MAPPING_ERROR: return "CUSOLVER_STATUS_MAPPING_ERROR";
			case CUSOLVER_STATUS_EXECUTION_FAILED: return "CUSOLVER_STATUS_EXECUTION_FAILED";
			case CUSOLVER_STATUS_INTERNAL_ERROR: return "CUSOLVER_STATUS_INTERNAL_ERROR";
			case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED: return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
			case CUSOLVER_STATUS_NOT_SUPPORTED: return "CUSOLVER_STATUS_NOT_SUPPORTED";
			default: return "CUSOLVER_STATUS_UNKNOWN";
			}
		}

		inline const char* statusName(cufftResult r)
		{
			switch (r)
			{
			case CUFFT_SUCCESS: return "CUFFT_SUCCESS";
			case CUFFT_INVALID_PLAN: return "CUFFT_INVALID_PLAN";
			case CUFFT_ALLOC_FAILED: return "CUFFT_ALLOC_FAILED";
			case CUFFT_INVALID_TYPE: return "CUFFT_INVALID_TYPE";
			case CUFFT_INVALID_VALUE: return "CUFFT_INVALID_VALUE";
			case CUFFT_INTERNAL_ERROR: return "CUFFT_INTERNAL_ERROR";
			case CUFFT_EXEC_FAILED: return "CUFFT_EXEC_FAILED";
			case CUFFT_SETUP_FAILED: return "CUFFT_SETUP_FAILED";
			case CUFFT_INVALID_SIZE: return "CUFFT_INVALID_SIZE";
			case CUFFT_INVALID_DEVICE: return "CUFFT_INVALID_DEVICE";
			case CUFFT_NO_WORKSPACE: return "CUFFT_NO_WORKSPACE";
			case CUFFT_NOT_IMPLEMENTED: return "CUFFT_NOT_IMPLEMENTED";
			case CUFFT_NOT_SUPPORTED: return "CUFFT_NOT_SUPPORTED";
			// Codes that differ between CUDA versions are reported by number (CudaError::code()).
			default: return "CUFFT_ERROR";
			}
		}

		inline std::string where(const char* expr, const char* file, int line)
		{
			return std::string(expr) + " at " + file + ":" + std::to_string(line);
		}

		inline bool failed(cudaError_t s) { return s != cudaSuccess; }
		inline bool failed(cublasStatus_t s) { return s != CUBLAS_STATUS_SUCCESS; }
		inline bool failed(cusolverStatus_t s) { return s != CUSOLVER_STATUS_SUCCESS; }
		inline bool failed(cufftResult s) { return s != CUFFT_SUCCESS; }

		inline CudaError makeError(cudaError_t s, const char* expr, const char* file, int line)
		{
			return CudaError("CUDA", static_cast<int>(s), cudaGetErrorName(s),
				std::string(cudaGetErrorString(s)) + " in " + where(expr, file, line));
		}
		inline CudaError makeError(cublasStatus_t s, const char* expr, const char* file, int line)
		{
			return CudaError("cuBLAS", static_cast<int>(s), cublasGetStatusName(s),
				std::string(cublasGetStatusString(s)) + " in " + where(expr, file, line));
		}
		inline CudaError makeError(cusolverStatus_t s, const char* expr, const char* file, int line)
		{
			return CudaError("cuSOLVER", static_cast<int>(s), statusName(s), where(expr, file, line));
		}
		inline CudaError makeError(cufftResult s, const char* expr, const char* file, int line)
		{
			return CudaError("cuFFT", static_cast<int>(s), statusName(s), where(expr, file, line));
		}

		template <typename Status>
		inline void check(Status status, const char* expr, const char* file, int line)
		{
			if (failed(status))
			{
				throw makeError(status, expr, file, line);
			}
		}

		template <typename Status>
		inline void checkNoThrow(Status status, const char* expr, const char* file, int line) noexcept
		{
			if (failed(status))
			{
				try
				{
					std::cerr << "Error (not thrown): " << makeError(status, expr, file, line).what() << std::endl;
				}
				catch (...)
				{
				}
			}
		}

		inline void checkLaunch(const char* file, int line)
		{
			check(cudaGetLastError(), "kernel launch", file, line);
#ifdef CUSH_SYNC_CHECKS
			check(cudaDeviceSynchronize(), "kernel execution (CUSH_SYNC_CHECKS)", file, line);
#endif
		}
	}

	/// <summary>
	/// Throws CudaError for a numerical failure that a library reports through an info value rather than its
	/// status, such as cuSOLVER's devInfo. Use after copying the info value back to the host.
	/// </summary>
	inline void throwNumericalError(const std::string& library, int info, const std::string& message)
	{
		throw CudaError(library, info, "info", message);
	}
}

#define CUSH_CHECK(expr) ::cush::detail::check((expr), #expr, __FILE__, __LINE__)
#define CUSH_CHECK_LAUNCH() ::cush::detail::checkLaunch(__FILE__, __LINE__)
#define CUSH_CHECK_NOTHROW(expr) ::cush::detail::checkNoThrow((expr), #expr, __FILE__, __LINE__)

inline cudaError_t setDevice()
{
	// Choose which GPU to run on; change this on a multi-GPU system.
	return cudaSetDevice(0);
}

#endif
