#pragma once

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "Radial/BoundaryNodes.cuh"

namespace
{
struct HostSurfaceNodes
{
	std::vector<double> r, z, nr, nz, ds, curvature;
};

/// <summary>
/// Builds the surface BoundaryNodes on the GPU for the given surface and copies them back.
/// </summary>
HostSurfaceNodes buildHostSurfaceNodes(const std::vector<double>& r, const std::vector<double>& eta, const std::vector<double>& etaPrime, const std::vector<double>& etaPrimePrime, double R)
{
	const size_t N = r.size();
	const size_t bytes = N * sizeof(double);
	std::vector<double> zeros(N, 0.0);

	double* dev[6];
	const std::vector<double>* sources[6] = { &r, &eta, &etaPrime, &etaPrimePrime, &zeros, &zeros };
	for (int i = 0; i < 6; ++i)
	{
		CHECK_CUDA(cudaMalloc(&dev[i], bytes));
		CHECK_CUDA(cudaMemcpy(dev[i], sources[i]->data(), bytes, cudaMemcpyHostToDevice));
	}

	const RadialSurfaceState surface{ dev[0], dev[1], dev[2], dev[3], dev[4], dev[5], N };
	BoundaryNodeBuffers buffers(N, 0);
	buffers.buildSurface(surface, R);
	CHECK_CUDA(cudaDeviceSynchronize());

	const BoundaryNodes view = buffers.view();
	HostSurfaceNodes host{ std::vector<double>(N), std::vector<double>(N), std::vector<double>(N), std::vector<double>(N), std::vector<double>(N), std::vector<double>(N) };
	CHECK_CUDA(cudaMemcpy(host.r.data(), view.r, bytes, cudaMemcpyDeviceToHost));
	CHECK_CUDA(cudaMemcpy(host.z.data(), view.z, bytes, cudaMemcpyDeviceToHost));
	CHECK_CUDA(cudaMemcpy(host.nr.data(), view.nr, bytes, cudaMemcpyDeviceToHost));
	CHECK_CUDA(cudaMemcpy(host.nz.data(), view.nz, bytes, cudaMemcpyDeviceToHost));
	CHECK_CUDA(cudaMemcpy(host.ds.data(), view.ds, bytes, cudaMemcpyDeviceToHost));
	CHECK_CUDA(cudaMemcpy(host.curvature.data(), view.curvature, bytes, cudaMemcpyDeviceToHost));

	for (double* p : dev)
	{
		CHECK_CUDA(cudaFree(p));
	}
	return host;
}

std::vector<double> cellCentres(size_t N, double R)
{
	std::vector<double> r(N);
	for (size_t i = 0; i < N; ++i)
	{
		r[i] = R * (static_cast<double>(i) + 0.5) / static_cast<double>(N);
	}
	return r;
}
}

TEST(RadialBoundaryNodes, FlatSurfaceHasVerticalNormalsAndCellsCoverRadius)
{
	constexpr size_t N = 64;
	constexpr double R = 1.0;
	const std::vector<double> r = cellCentres(N, R);
	const std::vector<double> zeros(N, 0.0);

	const HostSurfaceNodes nodes = buildHostSurfaceNodes(r, zeros, zeros, zeros, R);

	double totalDs = 0.0;
	for (size_t i = 0; i < N; ++i)
	{
		EXPECT_DOUBLE_EQ(nodes.r[i], r[i]);
		EXPECT_DOUBLE_EQ(nodes.z[i], 0.0);
		EXPECT_DOUBLE_EQ(nodes.nr[i], 0.0);
		EXPECT_DOUBLE_EQ(nodes.nz[i], 1.0);
		EXPECT_DOUBLE_EQ(nodes.curvature[i], 0.0);
		totalDs += nodes.ds[i];
	}
	EXPECT_NEAR(totalDs, R, 1.0e-12);
}

TEST(RadialBoundaryNodes, SlopedSurfaceHasUnitNormalPerpendicularToTangent)
{
	constexpr size_t N = 50;
	constexpr double R = 2.0;
	constexpr double slope = 0.3;
	const std::vector<double> r = cellCentres(N, R);
	std::vector<double> eta(N), etaPrime(N, slope), etaPrimePrime(N, 0.0);
	for (size_t i = 0; i < N; ++i)
	{
		eta[i] = slope * r[i];
	}

	const HostSurfaceNodes nodes = buildHostSurfaceNodes(r, eta, etaPrime, etaPrimePrime, R);

	double totalDs = 0.0;
	for (size_t i = 0; i < N; ++i)
	{
		EXPECT_NEAR(nodes.nr[i] * nodes.nr[i] + nodes.nz[i] * nodes.nz[i], 1.0, 1.0e-14) << "index " << i;
		// tangent (1, eta') must be perpendicular to the normal
		EXPECT_NEAR(nodes.nr[i] * 1.0 + nodes.nz[i] * slope, 0.0, 1.0e-14) << "index " << i;
		EXPECT_GT(nodes.nz[i], 0.0) << "normal must point out of the fluid (upwards)";
		totalDs += nodes.ds[i];
	}
	EXPECT_NEAR(totalDs, R * std::sqrt(1.0 + slope * slope), 1.0e-12);
}

TEST(RadialBoundaryNodes, WallNodesAreAllocatedAfterSurfaceNodes)
{
	BoundaryNodeBuffers buffers(10, 3);
	const BoundaryNodes view = buffers.view();
	EXPECT_EQ(view.nSurface, 10u);
	EXPECT_EQ(view.nWall, 3u);
	EXPECT_EQ(view.total(), 13u);
	EXPECT_EQ(buffers.total(), 13u);
}
