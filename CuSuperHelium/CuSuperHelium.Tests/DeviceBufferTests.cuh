#pragma once

#include "gtest/gtest.h"
#include "Common/DeviceBuffer.cuh"

#include <utility>
#include <vector>

TEST(DeviceBuffer, RoundTripsData)
{
	const std::vector<double> host = { 1.0, -2.5, 3.25, 4.0 };
	DeviceBuffer<double> buffer(host.size());
	ASSERT_NE(buffer.get(), nullptr);
	EXPECT_EQ(buffer.size(), host.size());
	EXPECT_EQ(buffer.bytes(), host.size() * sizeof(double));

	CHECK_CUDA(cudaMemcpy(buffer.get(), host.data(), buffer.bytes(), cudaMemcpyHostToDevice));
	std::vector<double> back(host.size());
	CHECK_CUDA(cudaMemcpy(back.data(), buffer.get(), buffer.bytes(), cudaMemcpyDeviceToHost));
	EXPECT_EQ(back, host);
}

TEST(DeviceBuffer, MoveTransfersOwnership)
{
	DeviceBuffer<double> a(8);
	double* raw = a.get();

	DeviceBuffer<double> b(std::move(a));
	EXPECT_EQ(b.get(), raw);
	EXPECT_EQ(b.size(), 8u);
	EXPECT_EQ(a.get(), nullptr);
	EXPECT_EQ(a.size(), 0u);

	DeviceBuffer<double> c(4);
	c = std::move(b);
	EXPECT_EQ(c.get(), raw);
	EXPECT_EQ(c.size(), 8u);
	EXPECT_EQ(b.get(), nullptr);
}

TEST(DeviceBuffer, ResetAndEmpty)
{
	DeviceBuffer<double> empty(0);
	EXPECT_EQ(empty.get(), nullptr);
	EXPECT_EQ(empty.bytes(), 0u);

	DeviceBuffer<double> buffer(16);
	buffer.reset();
	EXPECT_EQ(buffer.get(), nullptr);
	EXPECT_EQ(buffer.size(), 0u);
	buffer.reset(); // idempotent
	EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}
