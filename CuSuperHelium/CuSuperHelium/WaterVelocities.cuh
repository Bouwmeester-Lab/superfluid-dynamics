#pragma once
#ifndef WATERVELOCITIES_H
#define WATERVELOCITIES_H


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utilities.cuh"
#include "cufft.h"
#include "constants.cuh"
#include "cuDoubleComplexOperators.cuh"
#include <cuda/std/complex>

/// <summary>
/// Forms the matrix and vector needed for obtaining the velocities in the water model.
/// </summary>
/// <param name="ZPhi"></param>
/// <param name="ZPhiPrime"></param>
/// <param name="Zpp"></param>
/// <param name="N"></param>
/// <param name="out1">V1 matrix</param>
/// <param name="out2">Diagonal entries of V2</param>
/// <param name="lower"></param>
__global__ void createVelocityMatrices(const std_complex* Z, const std_complex* Zp, const std_complex* Zpp, int N, std_complex* out1, std_complex* out2, bool lower = true, size_t batchSize = 1);
/// <summary>
/// Calculates the veloctities in the water model from the V1 and V2 matrices calculated by createVelocityMatrices.
/// </summary>
/// <param name="a">The vorticities strengths</param>
/// <param name="aprime">The derivative of a</param>
/// <param name="V1">An NxN matrix allowing for the calculation of the velocities.</param>
/// <param name="V2">An N vector representing the diagonal entries of V2 (purely diagonal matrix)</param>
/// <param name="N">Size of the system.</param>
///void calculateVelocities(double* a, double* aprime, cufftDoubleComplex* V1, cufftDoubleComplex* V2, int N);

__global__ void calculateDiagonalVectorMultiplication(std_complex* diag, std_complex* vec, std_complex* out, int N);


__global__ void createVelocityMatrices(const std_complex* Z, const std_complex* Zp, const std_complex* Zpp, int N, std_complex* out1, std_complex* out2, bool lower, size_t batchSize)
{
	int j = blockIdx.y * blockDim.y + threadIdx.y; // row
	int k = blockIdx.x * blockDim.x + threadIdx.x; // col
	int b = blockIdx.z; // batch index
	int indx = k + j * N + b * N * N; // column major index

	if (indx >= batchSize * N * N) return; // out of bounds check

	if (k < N && j < N) {
		

		if (k == j)
		{
			// we are in the diagonal:
			
			out1[indx] = multiply_by_i(- 1.0 /(4.0 * CUDART_PI) * Zpp[k + b * N] / (cuda::std::pow(Zp[k + b * N], 2.0)));
			if (lower)
			{
				out1[indx] += 1.0 /(2.0 * Zp[k + b*N]);
			}
			else
			{
				out1[indx] -= 0.5 / Zp[k + b *N];
			}
			out2[k + b*N] = multiply_by_i(1.0 / (2.0*CUDART_PI * Zp[k + b*N]));
		}
		else
		{
			out1[indx] = multiply_by_i(-1.0/(4.0*CUDART_PI) * cotangent_green_function(Z[k + b * N], Z[j + b*N]));
		}
	}
}

__global__ void createHeliumVelocityMatrices(const std_complex* const Z, const std_complex* const Zp, const std_complex* const Zpp, double h, int N, std_complex* const out1, std_complex* const out2, bool lower, size_t batchSize)
{
	int j = blockIdx.y * blockDim.y + threadIdx.y; // row
	int k = blockIdx.x * blockDim.x + threadIdx.x; // col
	int b = blockIdx.z; // batch index

	if (b >= batchSize) return; // out of bounds check

	if (k < N && j < N) {
		int indx = k + j * N + b * N * N; // column major index

		if (k == j)
		{
			// we are in the diagonal:

			out1[indx] = multiply_by_i(-1.0 / (4.0 * CUDART_PI) * Zpp[k + b*N] / (cuda::std::pow(Zp[k + b*N], 2.0)));
			out1[indx] += multiply_by_i(1.0 / (4.0 * CUDART_PI) * cot(std_complex(0, Z[k + b*N].imag() + h)));
			if (lower)
			{
				out1[indx] += 1.0 / (2.0 * Zp[k]);
			}
			else
			{
				out1[indx] -= 0.5 / Zp[k];
			}
			out2[k + b*N] = multiply_by_i(1.0 / (2.0 * CUDART_PI * Zp[k + b*N]));
		}
		else
		{
			out1[indx] = multiply_by_i(-1.0 / (4.0 * CUDART_PI) * cotangent_green_function(Z[k+b*N], Z[j + b*N]));
			out1[indx] += multiply_by_i(1.0 / (4.0 * CUDART_PI) * cot(0.5 * (Z[k + b*N] - cuda::std::conj(Z[j + b*N])) + std_complex(0, h)));
		}
	}
}

__global__ void calculateDiagonalVectorMultiplication(std_complex* diag, std_complex* vec, std_complex* out, int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		out[i] = diag[i] * vec[i];
	}
}


//void createVelocityMatrices(double* vorticities, double aprime, cufftDoubleComplex* ZPhi, cufftDoubleComplex* ZPhiPrime, cufftDoubleComplex* Zpp, double* u1, double* v1, double* u2, double* v2);

template<int N, size_t batchSize>
class VelocityCalculator {
private:
	cublasHandle_t handle; ///< cuBLAS handle for managing BLAS operations
	const int threads = 256;
	const int blocks = (batchSize * N + threads - 1) / threads;
	
	const dim3 matrix_threads;// (16, 16);     // 256 threads per block in 2D
	const dim3 matrix_blocks; // ((N + 15) / 16, (N + 15) / 16);

	// variables for the batched version
	cuDoubleComplex** devPtrV1Array = nullptr; // device pointer array for V1 matrices
	cuDoubleComplex** devPtrAArray = nullptr;  // device pointer array for a vectors
	cuDoubleComplex** devPtrVelArray = nullptr; // device pointer array for velocity vectors

public:
	VelocityCalculator();
	~VelocityCalculator();
	void calculateVelocities(const std_complex* Z,
		const std_complex* Zp,
		const std_complex* Zpp,
		std_complex* a,
		std_complex* aprime,
		std_complex* V1,
		std_complex* V2,
		std_complex* velocities,
		bool lower = true);
};

template<int N, size_t batchSize>
VelocityCalculator<N, batchSize>::VelocityCalculator() : matrix_threads(16, 16, 1), matrix_blocks((N + 15) / 16, (N + 15) / 16, batchSize)
{
	cublasCreate(&handle);

	if (batchSize != 1) {
		checkCuda(cudaMalloc((void**)&devPtrV1Array, batchSize * sizeof(cuDoubleComplex*)));
		checkCuda(cudaMalloc((void**)&devPtrAArray, batchSize * sizeof(cuDoubleComplex*)));
		checkCuda(cudaMalloc((void**)&devPtrVelArray, batchSize * sizeof(cuDoubleComplex*)));
	}

}

template<int N, size_t batchSize>
VelocityCalculator<N, batchSize>::~VelocityCalculator()
{
	auto error = cudaGetLastError();
	if (error != cudaSuccess) {
		std::cerr << "CUDA error in ~VelocityCalculator before destroy handle: " << cudaGetErrorString(error) << std::endl;
	}

	if (handle) {
		
		checkCublas(cublasDestroy(handle));
	}
	 error = cudaGetLastError();
	if (error != cudaSuccess) {
		std::cerr << "CUDA error in ~VelocityCalculator before free of pointers of pointers: " << cudaGetErrorString(error) << std::endl;
	}
	if (devPtrV1Array) {
		//std::cout << "Freeing devPtrV1Array in ~VelocityCalculator" << devPtrV1Array << std::endl;
		checkCuda(cudaFree(devPtrV1Array));
		devPtrV1Array = nullptr;
		checkCudaErrors("Freeing devPtrV1Array in ~VelocityCalculator");
	}
		

	if (devPtrAArray) {
		checkCuda(cudaFree(devPtrAArray));
		devPtrAArray = nullptr;
	}
	checkCudaErrors("Freeing devPtrAArray in ~VelocityCalculator");

	if (devPtrVelArray) {
		checkCuda(cudaFree(devPtrVelArray));
		devPtrVelArray = nullptr;
	}
		
	checkCudaErrors("Freeing devPtrVelArray in ~VelocityCalculator");

	error = cudaGetLastError();
	if (error != cudaSuccess) {
		std::cerr << "CUDA error in ~VelocityCalculator: " << cudaGetErrorString(error) << std::endl;
	}
}

template<int N, size_t batchSize>
void VelocityCalculator<N, batchSize>::calculateVelocities(const std_complex* Z,
	const std_complex* Zp,
	const std_complex* Zpp,
	std_complex* a,
	std_complex* aprime,
	std_complex* V1,
	std_complex* V2,
	std_complex* velocities,
	bool lower)
{
	// calculate v2*aprime
	calculateDiagonalVectorMultiplication << <blocks, threads >> > (V2, aprime, velocities, batchSize * N);


	// calculate V1 * a + v2*aprime
	const cufftDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
	const cufftDoubleComplex beta = make_cuDoubleComplex(1.0, 0.0);
	if(batchSize == 1)
		cublasZgemv(handle, CUBLAS_OP_N, N, N, &alpha, reinterpret_cast<cuDoubleComplex*>(V1), N, reinterpret_cast<cuDoubleComplex*>(a), 1, &beta, reinterpret_cast<cuDoubleComplex*>(velocities), 1);
	else {
		// batched version
		std::vector<cuDoubleComplex*> hAptr(batchSize), hxptr(batchSize), hyptr(batchSize);
		for (size_t i = 0; i < batchSize; i++) {
			hAptr[i] = reinterpret_cast<cuDoubleComplex*>(V1) + i * N * N;
			hxptr[i] = reinterpret_cast<cuDoubleComplex*>(a) + i * N;
			hyptr[i] = reinterpret_cast<cuDoubleComplex*>(velocities) + i * N;
		}
		// copy to device
		checkCuda(cudaMemcpy(devPtrV1Array, hAptr.data(), batchSize * sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice));
		checkCuda(cudaMemcpy(devPtrAArray, hxptr.data(), batchSize * sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice));
		checkCuda(cudaMemcpy(devPtrVelArray, hyptr.data(), batchSize * sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice));
		// perform batched matrix-vector multiplication
		cublasZgemvBatched(handle, CUBLAS_OP_N, N, N, &alpha, devPtrV1Array, N, devPtrAArray, 1, &beta, devPtrVelArray, 1, batchSize);
	}
	// calculate the conjugate of the velocities, since this df/dz = u - i v
	conjugate_vector<<<blocks, threads >>>(velocities, velocities, batchSize * N);
}
#endif // !WATERVELOCITIES_H