#pragma once
#ifndef TRAJECTORY_LOGGER_H
#define TRAJECTORY_LOGGER_H

#include <thrust/device_vector.h>
#include "utilities.cuh"

template <typename T, size_t N>
class TrajectoryLogger
{
private:
	thrust::device_vector<double> devTimes;
	thrust::device_vector<cuda::std::array<T, N>> devYs;
	bool enabled = true;
	cudaStream_t stream = cudaStreamPerThread;
public:
	virtual int copyTimesToHost(double** times, size_t* count);
	virtual int copyStatesToHost(T** hostStates, size_t* countHost);

	void logTrajectory(double currentTime, T* state);

	bool isEnabled() const { return enabled; }
	void setStream(cudaStream_t s) { stream = s; }
};

template <typename T, size_t N>
int TrajectoryLogger<T, N>::copyTimesToHost(double** times, size_t* count)
{
	if(isEnabled())
	{
		// Copy data from device to host
		double* t = static_cast<double*>(std::malloc(devTimes.size() * sizeof(double)));
		if (t == nullptr) 
		{
			return -1; // Allocation failed
		}
		checkCuda(cudaMemcpyAsync(t, thrust::raw_pointer_cast(devTimes.data()), devTimes.size() * sizeof(double), cudaMemcpyDeviceToHost, this->stream), __func__, __FILE__, __LINE__);

		*times = t;
		*count = devTimes.size();
		return 0; // Success
	}
	return -1;
}

template<typename T, size_t N>
int TrajectoryLogger<T, N>::copyStatesToHost(T** hostStates, size_t* countHost)
{
	if(isEnabled())
	{
		// Copy data from device to host
		T* h = static_cast<T*>(std::malloc(devYs.size() * N * sizeof(T)));
		if (h == nullptr) 
		{
			return -1; // Allocation failed
		}
		checkCuda(cudaMemcpyAsync(h, thrust::raw_pointer_cast(devYs.data()), devYs.size() * N * sizeof(T), cudaMemcpyDeviceToHost, this->stream), __func__, __FILE__, __LINE__);
		*hostStates = h;
		*countHost = devYs.size();
		return 0; // Success
	}
	return -1;
}

template<typename T, size_t N>
void TrajectoryLogger<T, N>::logTrajectory(double currentTime, T* state)
{
	if(isEnabled())
	{
		devTimes.push_back(currentTime);
		appendToVector<T, N>(this->devYs, state, this->stream);
	}
}


#endif