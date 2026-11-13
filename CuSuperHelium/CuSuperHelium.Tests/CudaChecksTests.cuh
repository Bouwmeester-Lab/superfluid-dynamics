#pragma once

#include "gtest/gtest.h"
#include "Core/CudaChecks.cuh"

#include <string>

namespace
{
	__global__ void noopKernel() {}

	bool contains(const std::string& text, const std::string& part) { return text.find(part) != std::string::npos; }
}

TEST(CudaChecks, SuccessDoesNotThrow)
{
	EXPECT_NO_THROW(CUSH_CHECK(cudaSuccess));
	EXPECT_NO_THROW(CUSH_CHECK(CUBLAS_STATUS_SUCCESS));
	EXPECT_NO_THROW(CUSH_CHECK(CUSOLVER_STATUS_SUCCESS));
	EXPECT_NO_THROW(CUSH_CHECK(CUFFT_SUCCESS));
}

TEST(CudaChecks, EachLibraryThrowsCudaErrorWithContext)
{
	try
	{
		CUSH_CHECK(cudaErrorInvalidValue);
		FAIL() << "expected cush::CudaError";
	}
	catch (const cush::CudaError& e)
	{
		EXPECT_EQ(e.library(), "CUDA");
		EXPECT_EQ(e.code(), static_cast<int>(cudaErrorInvalidValue));
		EXPECT_TRUE(contains(e.what(), "cudaErrorInvalidValue")) << e.what();
		EXPECT_TRUE(contains(e.what(), "CudaChecksTests.cuh")) << e.what();
	}

	EXPECT_THROW(CUSH_CHECK(CUBLAS_STATUS_INVALID_VALUE), cush::CudaError);
	EXPECT_THROW(CUSH_CHECK(CUSOLVER_STATUS_INVALID_VALUE), cush::CudaError);

	try
	{
		CUSH_CHECK(CUFFT_INVALID_PLAN);
		FAIL() << "expected cush::CudaError";
	}
	catch (const cush::CudaError& e)
	{
		EXPECT_EQ(e.library(), "cuFFT");
		EXPECT_TRUE(contains(e.what(), "CUFFT_INVALID_PLAN")) << e.what();
	}
}

TEST(CudaChecks, LaunchCheckCatchesBadConfiguration)
{
	CUSH_CHECK(cudaGetLastError()); // start clean
	noopKernel<<<0, 1>>>();         // zero blocks: cudaErrorInvalidConfiguration (not sticky)
	EXPECT_THROW(CUSH_CHECK_LAUNCH(), cush::CudaError);

	noopKernel<<<1, 1>>>();
	EXPECT_NO_THROW(CUSH_CHECK_LAUNCH());
	CUSH_CHECK(cudaDeviceSynchronize());
}

TEST(CudaChecks, NoThrowVariantOnlyLogs)
{
	EXPECT_NO_THROW(CUSH_CHECK_NOTHROW(cudaErrorInvalidValue));
	EXPECT_NO_THROW(CUSH_CHECK_NOTHROW(CUFFT_INVALID_PLAN));
}

TEST(CudaChecks, NumericalErrorsUseTheSameType)
{
	EXPECT_THROW(cush::throwNumericalError("cuSOLVER", 3, "matrix is singular"), cush::CudaError);
}
