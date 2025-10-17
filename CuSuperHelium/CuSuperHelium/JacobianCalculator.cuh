#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "AutonomousProblem.h"
#include <memory>
#include "constants.cuh"
#include "utilities.cuh"
#include "cublas.h"

template <size_t N>
__global__ void createInitialBatchedZ(const std_complex* __restrict__ initialState, std_complex* __restrict__ ZBatched, double eps)
{
	int tId = blockIdx.x * blockDim.x + threadIdx.x;
	if (tId >= N) return; // each thread calculates one column of the initial Z matrix

	for (size_t i = 0; i < 3 * N; i++) 
	{
		if(i < N) // x part
			ZBatched[i * N + tId] = initialState[tId] + std_complex(eps, 0.0); // (x + eps) + i y
		else if(i < 2 * N) // y part
			ZBatched[i * N + tId] = initialState[tId] + std_complex(0.0, eps); // x + i (y + eps)
		else // phi part
			ZBatched[i * N + tId] = initialState[tId + N] + std_complex(eps, 0.0); // phi + eps + i 0
	}
}

__device__ __host__ __forceinline__ size_t columnMajorIndex(const size_t row, const size_t col, const size_t numRows) 
{
	return col * numRows + row;
}
/// <summary>
/// Constructs a Jacobian matrix from perturbed right-hand side vectors using central finite differences, for a system of particles. Designed for execution on a CUDA GPU.
/// </summary>
/// <param name="pos">Pointer to the array of complex values representing the positive perturbation results.</param>
/// <param name="neg">Pointer to the array of complex values representing the negative perturbation results.</param>
/// <param name="C">Pointer to the output array where the computed Jacobian matrix will be stored (in column-major order).</param>
/// <param name="particles">The number of particles in the system.</param>
/// <param name="eps">The perturbation magnitude used for finite difference calculations.</param>
/// <returns>This function does not return a value; it writes the computed Jacobian matrix to the array pointed to by C.</returns>
__global__ void createJacobianMatrixFromPerturbedRhs(const std_complex* __restrict__ pos, const std_complex* __restrict__ neg, double* __restrict__ C, size_t particles, double eps)
{
	int tId = blockIdx.x * blockDim.x + threadIdx.x;
	// create a stride loop to cover all elements in the matrix
	int totalElementsComplex = 6 * particles * particles; // 6N x 2N matrix with complex numbers 
	size_t k;
	size_t p;

	

	std_complex temp;
	for (int i = tId; i < totalElementsComplex; i += blockDim.x * gridDim.x)
	{
		temp = pos[i] - neg[i];
		k = i % (2 * particles); 
		p = i / (2 * particles); 
		
		if (k < particles) {
			// this is a velocity
			C[columnMajorIndex(k, p, 3 * particles)] = temp.real() / (2.0 * eps); // dVx / dx or dVy / dy
			C[columnMajorIndex(k + particles, p, 3 * particles)] = temp.imag() / (2.0 * eps); // dVy / dx or dVy / dy
		}
		else {
			// this is a potential
			C[columnMajorIndex(k + particles, p, 3 * particles)] = temp.real() / (2.0 * eps); // dPhi / dx or dPhi / dy
		}
	}
}


__global__ void createInitialState(const double* initialState, std_complex* complexState, size_t N) 
{
	int tId = blockIdx.x * blockDim.x + threadIdx.x;
	if (tId >= N) return;

	complexState[tId] = std_complex(initialState[tId], initialState[tId + N]); // x + i y
	complexState[tId + N] = std_complex(initialState[tId + 2 * N], 0.0); // phi + i 0
}

template <size_t N>
class JacobianCalculator
{
private:
	double eps = 1e-6;
	std::unique_ptr<BatchedAutonomousProblem<std_complex*, 2*N, 3*N>> autonomousProblem;
	
	std_complex* initialState;
	std_complex* devZPhiBatched; // this is an array holding 3*N * 2*N complex numbers. It's a basically 3*N different initial states for the autonomous problem, each one with a different perturbation in one of the state variables x, y, phi.
	std_complex* devNegZPhiBatched;
	std_complex* devPositiveRhsCalculated;
	std_complex* devNegativeRhsCalculated;
public:
	void setEpsilon(double eps);
	void calculateJacobian(const double* devState, double* devJacobian, cudaStream_t stream);
};

/// <summary>
/// Set the epsilon value used for numerical differentiation.
/// </summary>
/// <typeparam name="N"></typeparam>
/// <param name="eps"></param>
template<size_t N>
void JacobianCalculator<N>::setEpsilon(double eps)
{
	this->eps = eps;
}

template<size_t N>
void JacobianCalculator<N>::calculateJacobian(const double* devState, double* devJacobian, cudaStream_t stream)
{
	// for this operation we have to calculate how the autonomous problem changes when we change each state variable by eps
	// I am wondering wht's the best way to do this on the GPU so that it can be parallelized as much as possible.
	// first I want to translate the state in double* to a state in std_complex*.
	// The devState is (x, y, phi) where eache x, y, phi is a vector of size N. In the complex plane this is translated as (x + i y, phi + i 0) so it becomes a vector of size 2 N with complex numbers.
	if(initialState == nullptr)
	{
		checkCuda(cudaMallocAsync(&initialState, sizeof(std_complex) * 2 * N, stream));
	}
	if (devZPhiBatched == nullptr) 
	{
		checkCuda(cudaMallocAsync(&devZPhiBatched, sizeof(std_complex) * 6 * N * N, stream));
	}
	if (devPositiveRhsCalculated == nullptr) 
	{
		checkCuda(cudaMallocAsync(&devPositiveRhsCalculated, sizeof(std_complex) * 6 * N * N, stream));
	}

	// create the initial state in complex numbers using the custom kernel
	int blockSize = 256;
	int numBlocks = (N + blockSize - 1) / blockSize;
	createInitialState << <numBlocks, blockSize, 0, stream >> > (devState, initialState, N);

	createInitialBatchedZ<N> << <numBlocks, blockSize, 0, stream >> > (initialState, devZBatched, eps);
	createInitialBatchedZ<N> << <numBlocks, blockSize, 0, stream >> > (initialState, devNegZPhiBatched, -eps); // negative perturbation

	autonomousProblem->setStream(stream);
	autonomousProblem->runBatch(devZPhiBatched, devPositiveRhsCalculated); // calculate the autonomous problem for each perturbed initial state in devZBatched with a postiive perturbation F(Y + eps)
	autonomousProblem->runBatch(devNegZPhiBatched, devNegativeRhsCalculated); // calculates the autonomous problem for each perturbed initial state in devZBatched with a negative perturbation F(Y - eps)
	// now we have the results of the autonomous problem for each perturbed initial state in devRhsCalculated. We can now calculate the Jacobian using finite differences.
	// we will use central differences here.
	// J_ij = (f(x + eps) - f(x - eps)) / (2 * eps)
	createJacobianMatrixFromPerturbedRhs << <(6 * N * N + 255) / 256, 256, 0, stream >> > (devPositiveRhsCalculated, devNegativeRhsCalculated, devJacobian, N, this->eps);
}
