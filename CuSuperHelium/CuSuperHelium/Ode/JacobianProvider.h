#pragma once
#ifndef JACOBIAN_PROVIDER_H
#define JACOBIAN_PROVIDER_H

#include <stdexcept>
#include "cuda_runtime.h"
#include "Core/CudaChecks.cuh"
#include "Core/DeviceBuffer.cuh"
#include "Ode/OdeProblem.h"

/// <summary>
/// Supplies the Jacobian df/dy of an OdeProblem&lt;double&gt; to implicit solvers (Gauss-Legendre).
/// Device pointers; the Jacobian is n x n, column-major (column j = df/dy_j), n = problem size.
/// </summary>
class JacobianProvider
{
public:
	virtual ~JacobianProvider() = default;
	virtual void calculateJacobian(const double* devState, double* devJacobian) = 0;
	virtual void setStream(cudaStream_t stream) = 0;
};

namespace cush::detail
{
	static __global__ void perturbComponent(const double* y, double* out, size_t j, double eps, size_t n)
	{
		size_t i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < n) out[i] = y[i] + (i == j ? eps : 0.0);
	}

	static __global__ void centralDifferenceColumn(const double* fPlus, const double* fMinus, double* column, double eps, size_t n)
	{
		size_t i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < n) column[i] = (fPlus[i] - fMinus[i]) / (2.0 * eps);
	}
}

/// <summary>
/// Central-difference Jacobian of any OdeProblem&lt;double&gt;: 2 n right-hand-side evaluations, one column at a time.
/// Generic, so any problem can be integrated with Gauss-Legendre. Problems with a faster structured Jacobian
/// (the batched boundary integral, JacobianCalculator) provide their own.
/// The time passed to rhs() is set with setTime; autonomous problems ignore it.
/// </summary>
class FiniteDifferenceJacobian final : public JacobianProvider
{
public:
	explicit FiniteDifferenceJacobian(OdeProblem<double>& problem, double eps = 1e-6)
		: problem(problem), n(problem.size()), eps(eps), yPlus(n), yMinus(n), fPlus(n), fMinus(n)
	{
		if (n == 0) throw std::invalid_argument("FiniteDifferenceJacobian: the problem has an empty state");
	}

	void setTime(double time) { t = time; }
	void setStream(cudaStream_t s) override { stream = s; problem.setStream(s); }

	void calculateJacobian(const double* devState, double* devJacobian) override
	{
		const int threads = 256;
		const int blocks = static_cast<int>((n + threads - 1) / threads);
		for (size_t j = 0; j < n; ++j)
		{
			cush::detail::perturbComponent<<<blocks, threads, 0, stream>>>(devState, yPlus.get(), j, eps, n);
			CUSH_CHECK_LAUNCH();
			cush::detail::perturbComponent<<<blocks, threads, 0, stream>>>(devState, yMinus.get(), j, -eps, n);
			CUSH_CHECK_LAUNCH();
			problem.rhs(yPlus.get(), t, fPlus.get());
			problem.rhs(yMinus.get(), t, fMinus.get());
			cush::detail::centralDifferenceColumn<<<blocks, threads, 0, stream>>>(fPlus.get(), fMinus.get(), devJacobian + j * n, eps, n);
			CUSH_CHECK_LAUNCH();
		}
	}

private:
	OdeProblem<double>& problem;
	const size_t n;
	double eps;
	double t = 0.0;
	cudaStream_t stream = cudaStreamPerThread;
	DeviceBuffer<double> yPlus, yMinus, fPlus, fMinus;
};

#endif // !JACOBIAN_PROVIDER_H
