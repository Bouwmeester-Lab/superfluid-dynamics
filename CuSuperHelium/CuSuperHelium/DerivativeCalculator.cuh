#pragma once
#ifndef DERIVATIVE_CALCULATOR_H
#define DERIVATIVE_CALCULATOR_H

#include "cuda_runtime.h"

class DerivativeCalculator
{
public:
	virtual void calculateFirstDerivative(const double* input, const double* x, double* output, size_t N) = 0;
};

class FiniteDifferenceDerivativeCalculator final : public DerivativeCalculator
{
public:
	virtual void calculateFirstDerivative(const double* input, const double* x, double* output, size_t N) override;
};

static __global__ void calculateFirstDerivativeKernel(const double* input, const double* x, double* output, size_t N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N)
	{
		if (idx == 0)
		{
			output[idx] = (input[idx + 1] - input[idx]) / (x[idx + 1] - x[idx]); // Forward difference
		}
		else if (idx == N - 1)
		{
			output[idx] = (input[idx] - input[idx - 1]) / (x[idx] - x[idx - 1]); // Backward difference
		}
		else
		{
			output[idx] = (input[idx + 1] - input[idx - 1]) / (2.0 * (x[idx + 1] - x[idx])); // Central difference
		}
	}
}

void FiniteDifferenceDerivativeCalculator::calculateFirstDerivative(const double* input, const double* x, double* output, size_t N)
{
	calculateFirstDerivativeKernel<<<(N + 255) / 256, 256>>>(input, x, output, N);
}

#endif // !DERIVATIVE_CALCULATOR_H


