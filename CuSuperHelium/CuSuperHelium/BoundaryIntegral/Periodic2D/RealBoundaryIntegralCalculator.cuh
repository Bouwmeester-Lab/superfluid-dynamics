#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cufft.h"
#include "Core/constants.cuh"
#include "Math/Derivatives.cuh"
#include "BoundaryIntegral/Periodic2D/WaterVelocities.cuh"
#include "LinearAlgebra/MatrixSolver.cuh"
#include "Ode/AutonomousProblem.h"
#include "BoundaryIntegral/Periodic2D/Energies.cuh"
#include "Optomechanics/DelayedIntensityTerm.cuh"
#include "BoundaryIntegral/Periodic2D/BoundaryProblem.cuh"
#include "BoundaryIntegral/Periodic2D/BaseBoundaryIntegrator.cuh"
#include "BoundaryIntegral/Periodic2D/TimedBoundaryIntegrator.cuh"

__global__ void convertToComplexStateKernel(const double* realState, std_complex* complexState, size_t N)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N)
	{
		complexState[idx] = std_complex(realState[idx], realState[N + idx]); // assuming the real state is organized as [x0, x1, ..., xN-1, y0, y1, ..., yN-1, phi0, .. phiN-1]
		complexState[N + idx] = std_complex(realState[2 * N + idx], 0.0); // phi is real, imaginary part is 0
	}
}

__global__ void convertToRealRhsKernel(const std_complex* complexRHS, double* realRHS, size_t N)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N)
	{
		realRHS[idx] = complexRHS[idx].real(); // vx component
		realRHS[N + idx] = complexRHS[idx].imag(); // vy component
		realRHS[2 * N + idx] = complexRHS[N + idx].real(); // dphi component
	}
}
/// <summary>
/// A wrapper around BoundaryIntegralCalculator to work with real states instead of complex states.
/// It converts the real state to complex, calls the complex boundary integral calculator, and then converts the complex RHS back to real.
/// It assumes the real state is organized as [x0, x1, ..., xN-1, y0, y1, ..., yN-1, phi0, .. phiN-1]
/// and the complex state is organized as [z0, z1, ..., zN-1, phi0, phi1, ..., phiN-1] where zi = xi + i*yi
/// phi is complex number with the imaginary part being zero.
/// It keeps all calculations on the GPU. No copying to/from host is performed.
/// 
/// It uses a temporary device memory for the complex state and complex RHS.
/// These are allocated in the constructor and freed in the destructor.
/// </summary>
/// <typeparam name="N"></typeparam>
template <size_t N>
class RealBoundaryIntegralCalculator final : public AutonomousProblem<double, 3 * N> 
{
	BaseBoundaryIntegralCalculator<N, 1>& boundaryIntegralCalculator;
	cudaStream_t stream = cudaStreamPerThread;

	std_complex* devComplexState = nullptr;
	std_complex* devComplexRHS = nullptr;

	const size_t blockSize = 256;
	const size_t numBlocks = (N + blockSize - 1) / blockSize;

public:
	virtual void run(double* initialState, double* rhs) override;
	virtual void setStream(cudaStream_t stream) override;

	RealBoundaryIntegralCalculator(BaseBoundaryIntegralCalculator<N, 1>& boundaryIntegralCalculator);
	~RealBoundaryIntegralCalculator();
};

template<size_t N>
void RealBoundaryIntegralCalculator<N>::run(double* initialState, double* rhs)
{
	// convert real state to complex state
	
	convertToComplexStateKernel<<<numBlocks, blockSize, 0, stream>>>(initialState, devComplexState, N);
	CUSH_CHECK_LAUNCH();
	
	// calculate complex RHS
	boundaryIntegralCalculator.run(devComplexState, devComplexRHS);
	convertToRealRhsKernel<<<numBlocks, blockSize, 0, stream >> > (devComplexRHS, rhs, N);
	CUSH_CHECK_LAUNCH();
}

template<size_t N>
void RealBoundaryIntegralCalculator<N>::setStream(cudaStream_t stream)
{
	this->stream = stream;
	boundaryIntegralCalculator.setStream(stream);
}

template<size_t N>
RealBoundaryIntegralCalculator<N>::RealBoundaryIntegralCalculator(BaseBoundaryIntegralCalculator<N, 1>& boundaryIntegralCalculator) : boundaryIntegralCalculator(boundaryIntegralCalculator)
{
	CUSH_CHECK(cudaMalloc(&devComplexState, 2 * N * sizeof(std_complex)));
	CUSH_CHECK(cudaMalloc(&devComplexRHS, 2 * N * sizeof(std_complex)));
}

template<size_t N>
RealBoundaryIntegralCalculator<N>::~RealBoundaryIntegralCalculator()
{
	CUSH_CHECK_NOTHROW(cudaFree(devComplexState));
	CUSH_CHECK_NOTHROW(cudaFree(devComplexRHS));
}
