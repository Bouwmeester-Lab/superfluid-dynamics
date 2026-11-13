#pragma once
#ifndef TRAJECTORY_LOGGER_H
#define TRAJECTORY_LOGGER_H

#include <cstdlib>
#include <stdexcept>
#include <thrust/device_vector.h>
#include "Core/CudaChecks.cuh"
#include "IO/StepObserver.h"

/// <summary>
/// Records the trajectory of an integration on the device: one time and one state per accepted step,
/// and optionally the initial state. The state size is taken from the first call.
/// </summary>
template <typename T>
class TrajectoryLogger final : public StepObserver<T>
{
public:
	explicit TrajectoryLogger(bool recordInitialState = false) : recordInitialState(recordInitialState) {}

	void onStart(double t0, const T* y0, size_t n, cudaStream_t stream) override
	{
		if (recordInitialState)
		{
			record(t0, y0, n, stream);
		}
	}

	void onAcceptedStep(double t, const T* y, size_t n, cudaStream_t stream) override
	{
		record(t, y, n, stream);
	}

	size_t count() const { return devTimes.size(); }

	/// Copies the recorded times into a malloc'ed host array the caller frees. Returns 0, or -1 on allocation failure.
	int copyTimesToHost(double** times, size_t* countHost)
	{
		double* t = static_cast<double*>(std::malloc(devTimes.size() * sizeof(double)));
		if (t == nullptr && !devTimes.empty())
		{
			return -1;
		}
		CUSH_CHECK(cudaMemcpyAsync(t, thrust::raw_pointer_cast(devTimes.data()), devTimes.size() * sizeof(double), cudaMemcpyDeviceToHost, stream));
		CUSH_CHECK(cudaStreamSynchronize(stream));
		*times = t;
		*countHost = devTimes.size();
		return 0;
	}

	/// Copies the recorded states (count() x stateSize, contiguous) into a malloc'ed host array the caller frees.
	int copyStatesToHost(T** hostStates, size_t* countHost)
	{
		T* h = static_cast<T*>(std::malloc(devYs.size() * sizeof(T)));
		if (h == nullptr && !devYs.empty())
		{
			return -1;
		}
		CUSH_CHECK(cudaMemcpyAsync(h, thrust::raw_pointer_cast(devYs.data()), devYs.size() * sizeof(T), cudaMemcpyDeviceToHost, stream));
		CUSH_CHECK(cudaStreamSynchronize(stream));
		*hostStates = h;
		*countHost = devTimes.size();
		return 0;
	}

private:
	bool recordInitialState;
	size_t stateSize = 0;
	cudaStream_t stream = cudaStreamPerThread;
	thrust::device_vector<double> devTimes;
	thrust::device_vector<T> devYs;

	void record(double t, const T* y, size_t n, cudaStream_t s)
	{
		if (stateSize == 0)
		{
			stateSize = n;
		}
		else if (n != stateSize)
		{
			throw std::invalid_argument("TrajectoryLogger: state size changed during the integration");
		}
		stream = s;
		devTimes.push_back(t);
		const size_t offset = devYs.size();
		devYs.resize(offset + n);
		CUSH_CHECK(cudaMemcpyAsync(thrust::raw_pointer_cast(devYs.data()) + offset, y, n * sizeof(T), cudaMemcpyDeviceToDevice, s));
	}
};

#endif
