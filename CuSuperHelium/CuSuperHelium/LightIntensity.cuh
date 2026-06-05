
#pragma once
#ifndef LIGHT_INTENSITY_H
#define LIGHT_INTENSITY_H
#include <cub/cub.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include "cuda_runtime.h"

#include "OptomechanicalVariables.h"
#include "constants.cuh"
#include "ProblemProperties.hpp"
#include "ProblemProperties.hpp"



struct GaussianDistribution 
{
	double x0;
	double sigma;

	__host__ __device__ double operator()(double x) const
	{
		const double z = (x - x0) / sigma;
		return 1.0/(sqrt(2.0*PI_d * sigma * sigma)) * exp(-0.5 * z * z); // this uses the normalized version (area = 1.0) so that the integral converges.
	}
};

template <typename T>
__global__ void _compute_weights(double* weights, int N, const std_complex* Z, T shape) 
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N) {
		weights[idx] = shape(Z[idx].real());
	}
}

struct FrequencyShiftCombination
{
	const std_complex* Z;
	OptomechanicalVariables variables;
	const double* weights;
	const double dx;

	__host__ FrequencyShiftCombination(const std_complex* Z, OptomechanicalVariables variables, const double* weights, double dx) : Z(Z), variables(variables), weights(weights), dx(dx) {}

	__device__ double operator()(int idx) const
	{
		return - variables.G * weights[idx] * Z[idx].imag() * dx; // the weights represent the spatial profile of the optical mode,
		// and Z[idx].imag() is the height of the superfluid at that point. The product gives the contribution to the frequency shift from that point.
		// make sure the weights converge to 1 when summed up.
	}
};

class LightIntensity
{
private:
	double* weights; // device pointer to store the computed weights
	double* frequency_shift_result; // double stored in the GPU
	int N; // number of points in the spatial grid

	void* tempStorage = nullptr; // temporary storage for CUB reduction
	size_t tempStorageBytes = 0; // size of the temporary storage
	cudaStream_t stream = cudaStreamPerThread; // CUDA stream for asynchronous operations

	bool allocated = false; // flag to track if memory has been allocated
public:
	//__device__ __host__ LightIntensity(OptomechanicalVariables variables) : variables(variables) {}
	__host__ LightIntensity() {}
	
	~LightIntensity() 
	{
		free();
	}

	void inline guardAgainstNotAllocated() {
		if (!allocated) {
			throw std::runtime_error("Memory not allocated. Call allocate(N) before using this function.");
		}
	}

	void free() {
		cudaFree(weights);
		if(tempStorage != nullptr) {
			cudaFree(tempStorage);
		}
		if(frequency_shift_result != nullptr) {
			cudaFree(frequency_shift_result);
		}
	}

	void setStream(cudaStream_t stream) {
		this->stream = stream;
	}

	void allocate(int N)
	{
		this->N = N;
		cudaMalloc(&weights, N * sizeof(double));
		cudaMalloc(&frequency_shift_result, sizeof(double));
		allocated = true;
	}

	template <typename T>
	void compute_weights(const std_complex* Z, OptomechanicalVariables variables, T shape)
	{
		guardAgainstNotAllocated();
		const int threadsPerBlock = 256;
		const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
		_compute_weights<T><<<blocksPerGrid, threadsPerBlock>>>(weights, N, Z, shape);
	}

	static __device__ __host__ double compute_x_profile(double x, double x0, double sigma)
	{
		return exp(-pow(x - x0, 2) / (2 * pow(sigma, 2)));
	}

	static __device__ __host__ double compute_intensity(double frequency_shift, double x, OptomechanicalVariables variables)
	{
		double delta_f = variables.detuning + frequency_shift;

		return 0.25 * cuda::std::pow(variables.gamma, 2.0) * variables.max_intensity / ( cuda::std::pow(delta_f, 2.0) + cuda::std::pow(variables.gamma / 2, 2.0));
	}

	__host__ double* get_dev_frequency_shift() {
		return frequency_shift_result;
	}

	double* compute_frequency_shift(const std_complex* Z, OptomechanicalVariables variables, ProblemProperties properties)
	{
		guardAgainstNotAllocated();
		thrust::counting_iterator<int> countingIterator(0);

		// create a transform iterator that applies the FrequencyShiftCombination functor
		FrequencyShiftCombination frequencyShiftCombination(Z, variables, weights, 2.0*PI_d / static_cast<double>(N));

		auto transformIterator = thrust::make_transform_iterator(countingIterator, frequencyShiftCombination);

		// get temporary storage size
		cub::DeviceReduce::Sum(this->tempStorage, this->tempStorageBytes, transformIterator, this->frequency_shift_result, N);


		// allocate temporary storage
		cudaMalloc(&this->tempStorage, this->tempStorageBytes);

		// perform the reduction
		cub::DeviceReduce::Sum(this->tempStorage, this->tempStorageBytes, transformIterator, this->frequency_shift_result, N);

		cudaFreeAsync(this->tempStorage, stream);

		this->tempStorage = nullptr; // reset tempStorage pointer after freeing

		return frequency_shift_result;
	}

	static __device__ __host__ inline double get_current_intensity_drive_strength(OptomechanicalVariables variables, double sigma, ProblemProperties properties)
	{
		return hbar_d / (properties.base_energy * properties.base_time * properties.rho) * variables.G / cuda::std::pow(sigma, 2.0); // G is in a.u. of frequency per length. So hbar gets converted adimensionalized this way.
	}
};

#endif // LIGHT_INTENSITY_H