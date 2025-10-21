#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "AutonomousProblem.h"
#include <memory>
#include "constants.cuh"
#include "utilities.cuh"

__global__ void createInitialBatchedZ(const std_complex* __restrict__ initialState, std_complex* __restrict__ ZBatched, double eps, size_t N)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x; // tid from 0 to 2*N - 1 ( each thread calculates one perturbation, 2 elements (position and potential))
	int b = blockIdx.y; // batch id
	//printf("Thread ID: %d, Block ID: %d\n", tid, b);
	
	if (tid >= 2 * N) return; // we have 2*N threads per block, each thread corresponds to one element in the state (x + i y or phi + i 0)

	//ZBatched[tid + b * N] = 2;

	// b tells you if we are perturbing x, y or phi
	size_t coordinatePerturbed = b / N; // 0 = x, 1 = y, 2 = phi
	size_t particleIndex = b % N; // which particle we are perturbing
	bool positionOrPotential = (tid < N); // true = position, false = potential

	size_t base;
	if (positionOrPotential) {
		base = 0; // positions are in the first 3N*N elements
	}
	else {
		base = 3 * N * N; // potentials are in the last 3N*N elements
	}

	//if (coordinatePerturbed == 2) {
	//	printf("b: %d", b);
	//}

	//printf("tid: %d, b: %d, coordinatePerturbed: %d, particleIndex: %d, positionOrPotential: %d, base: %d\n", tid, b, coordinatePerturbed, particleIndex, positionOrPotential, base);
	// let's first make sure that copying works correctly
	//if (positionOrPotential) {
	//	ZBatched[base + b * N + tid] = initialState[tid]; // copies the position (x + i y) into the batched array using b as the index indicating where to write it, the output is like: (N postions for pertubation 1 (b = 0), N positions for perturbation 2 (b = 1), ..., N postitions for perturbation N (b = N-1), N phis for perturbation 1 (b = 0), ...)
	//}
	//else {
	//	ZBatched[base + b * N + (tid - N)] = initialState[tid]; // copies the potential (phi + i 0) into the batched array
	//}

	if (tid == particleIndex && coordinatePerturbed != 2)
	{
		// perform perturbation in whatever coordinate we are perturbing
		if (coordinatePerturbed == 0) {
			// we need to perturb x
			ZBatched[base + b * N + tid] = initialState[tid] + std_complex(eps, 0.0); // x + i y + eps in the real part
		}
		else if(coordinatePerturbed == 1) {
			// we need to perturb y
			ZBatched[base + b * N + tid] = initialState[tid] + std_complex(0.0, eps); // x + i (y + eps) in the imaginary part
		}
	}
	else if(tid == particleIndex + N && coordinatePerturbed == 2)
	{
			// we need to perturb phi
		if (!positionOrPotential) // only perturb phi if we are in the potential part
			ZBatched[base + b * N + tid - N] = initialState[tid] + std_complex(eps, 0.0); // phi + i 0 + eps in the real part
	}
	else 
	{
		// simply copy the initial state into the right position in the batched array
		if (positionOrPotential) {
			// this is a position, it goes in the first 3N^2 elements.
			ZBatched[b * N + tid] = initialState[tid]; // copies the position (x + i y) into the batched array using b as the index indicating where to write it, the output is like: (N postions for pertubation 1 (b = 0), N positions for pertubation 2 (b = 1), ..., N postitions for perturbation N (b = N-1), N phis for perturbation 1 (b = 0), ...)
		}
		else {
			// this is a potential, it goes in the last 3N^2 elements.
			ZBatched[base + b * N + tid - N ] = initialState[tid]; // copies the potential (phi + i 0) into the batched array
		}		
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
//
//template <size_t N>
//class JacobianCalculator
//{
//private:
//	double eps = 1e-6;
//	std::unique_ptr<BatchedAutonomousProblem<std_complex*, 2*N, 3*N>> autonomousProblem;
//	
//	std_complex* initialState;
//	std_complex* devZPhiBatched; // this is an array holding 3*N * 2*N complex numbers. It's a basically 3*N different initial states for the autonomous problem, each one with a different perturbation in one of the state variables x, y, phi.
//	std_complex* devNegZPhiBatched;
//	std_complex* devPositiveRhsCalculated;
//	std_complex* devNegativeRhsCalculated;
//
//	dim3 matrix_threads;
//	dim3 matrix_blocks;// = dim3()
//public:
//	void setEpsilon(double eps);
//	void calculateJacobian(const double* devState, double* devJacobian, cudaStream_t stream);
//
//	JacobianCalculator() : matrix_threads(256, 1, 1), matrix_blocks((3 * N + 255) / 256, 3 * N, 1)
//	{
//	}
//};
//
///// <summary>
///// Set the epsilon value used for numerical differentiation.
///// </summary>
///// <typeparam name="N"></typeparam>
///// <param name="eps"></param>
//template<size_t N>
//void JacobianCalculator<N>::setEpsilon(double eps)
//{
//	this->eps = eps;
//}
//
//template<size_t N>
//void JacobianCalculator<N>::calculateJacobian(const double* devState, double* devJacobian, cudaStream_t stream)
//{
//	// for this operation we have to calculate how the autonomous problem changes when we change each state variable by eps
//	// I am wondering wht's the best way to do this on the GPU so that it can be parallelized as much as possible.
//	// first I want to translate the state in double* to a state in std_complex*.
//	// The devState is (x, y, phi) where eache x, y, phi is a vector of size N. In the complex plane this is translated as (x + i y, phi + i 0) so it becomes a vector of size 2 N with complex numbers.
//	if(initialState == nullptr)
//	{
//		checkCuda(cudaMallocAsync(&initialState, sizeof(std_complex) * 2 * N, stream));
//	}
//	if (devZPhiBatched == nullptr) 
//	{
//		checkCuda(cudaMallocAsync(&devZPhiBatched, sizeof(std_complex) * 6 * N * N, stream));
//	}
//	if (devPositiveRhsCalculated == nullptr) 
//	{
//		checkCuda(cudaMallocAsync(&devPositiveRhsCalculated, sizeof(std_complex) * 6 * N * N, stream));
//	}
//
//	//// create the initial state in complex numbers using the custom kernel
//	//int blockSize = 256;
//	//int numBlocks = (N + blockSize - 1) / blockSize;
//	//createInitialState << <this->matrixThreads, this->matrixBlocks, 0, stream >> > (devState, initialState, N);
//
//	//createInitialBatchedZ<N> << <numBlocks, blockSize, 0, stream >> > (initialState, devZBatched, eps);
//	//createInitialBatchedZ<N> << <numBlocks, blockSize, 0, stream >> > (initialState, devNegZPhiBatched, -eps); // negative perturbation
//
//	//autonomousProblem->setStream(stream);
//	//autonomousProblem->runBatch(devZPhiBatched, devPositiveRhsCalculated); // calculate the autonomous problem for each perturbed initial state in devZBatched with a postiive perturbation F(Y + eps)
//	//autonomousProblem->runBatch(devNegZPhiBatched, devNegativeRhsCalculated); // calculates the autonomous problem for each perturbed initial state in devZBatched with a negative perturbation F(Y - eps)
//	//// now we have the results of the autonomous problem for each perturbed initial state in devRhsCalculated. We can now calculate the Jacobian using finite differences.
//	//// we will use central differences here.
//	//// J_ij = (f(x + eps) - f(x - eps)) / (2 * eps)
//	//createJacobianMatrixFromPerturbedRhs << <(6 * N * N + 255) / 256, 256, 0, stream >> > (devPositiveRhsCalculated, devNegativeRhsCalculated, devJacobian, N, this->eps);
//}
