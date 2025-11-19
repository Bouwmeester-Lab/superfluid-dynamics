#pragma once
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <cuda/std/array>

template <typename T, size_t N>
void appendToVector(thrust::device_vector<cuda::std::array<T, N>>& vec, double* devState, cudaStream_t stream = cudaStreamPerThread) 
{
	vec.resize(vec.size() + 1);
	// get the raw pointer to the newly added element
	cuda::std::array<T, N>* rawPtr = thrust::raw_pointer_cast(&vec[(vec.size() - 1)]);

	cudaMemcpyAsync(rawPtr, devState, N * sizeof(T), cudaMemcpyDeviceToDevice, stream);
}
