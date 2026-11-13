#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>
#include "Core/CudaChecks.cuh"

template <typename T>
class DataLogger
{
public:
	/// n: number of elements of type T in each logged state
	explicit DataLogger(size_t n);
	virtual ~DataLogger();

	void setSize(size_t size, bool allocateTimes = true)
	{
		this->size = size;
		data.clear();
		data.reserve(size); // , std::vector<T>(N));
		if (allocateTimes) {
			times.reserve(size); // , 0.0);
		}
		fixedSize = true;
	}
	void setStep(size_t steps)
	{
		this->steps = steps;
	}
	void setReadyToCopy(T* devPointer, cudaStream_t stream = cudaStreamPerThread, double time = -1, bool logTimes = false) 
	{
		// return;

		CUSH_CHECK(cudaEventRecord(readyToCopy, stream));
		copyScheduled = true;

		CUSH_CHECK(cudaStreamWaitEvent(copyStream, readyToCopy, 0));
		if (logTimes) {
			times.push_back(time);
		}
		//if (currentIndex == data.capacity()) {
			data.emplace_back(N); // Add a new vector to hold the data if the size we reached capacity
		//}
		
		CUSH_CHECK(cudaMemcpyAsync(data.at(currentIndex++).data(), devPointer, N * sizeof(T), cudaMemcpyDeviceToHost, copyStream));
		
		CUSH_CHECK(cudaEventRecord(copyDone, copyStream));
	}

	bool isCopyScheduled() const
	{
		return copyScheduled;
	}

	void waitForCopy(cudaStream_t stream = cudaStreamPerThread)
	{
		if (copyScheduled) {
			CUSH_CHECK(cudaStreamWaitEvent(stream, copyDone, 0));
			copyScheduled = false;
		}
	}

	bool shouldCopy(const size_t step) const
	{
		return step % steps == 0;// && currentIndex < size;
	}

	std::vector<T> getData(const size_t index) const
	{
		if (index >= size) {
			throw std::out_of_range("Index out of range in DataLogger");
		}
		return data.at(index);
	}

	/// <summary>
	/// Gets a COPY of all current data.
	/// </summary>
	/// <returns></returns>
	std::vector<std::vector<T>> getAllData()
	{
		return data;
	}

	std::vector<double> getTimes() {
		return times;
	}

private:
	const size_t N; ///< Elements per logged state
	cudaEvent_t readyToCopy, copyDone;
	cudaStream_t copyStream;
	size_t size;
	bool fixedSize = false;
	std::vector<std::vector<T>> data;
	std::vector<double> times;

	bool copyScheduled = false;
	size_t currentIndex = 0;
	size_t steps = 1;
};

template <typename T>
DataLogger<T>::DataLogger(size_t n) : N(n)
{
	CUSH_CHECK(cudaEventCreate(&readyToCopy));
	CUSH_CHECK(cudaEventCreate(&copyDone));
	CUSH_CHECK(cudaStreamCreate(&copyStream));
	setSize(10);
}
template <typename T>
DataLogger<T>::~DataLogger()
{
	CUSH_CHECK_NOTHROW(cudaEventDestroy(readyToCopy));
	CUSH_CHECK_NOTHROW(cudaEventDestroy(copyDone));
	CUSH_CHECK_NOTHROW(cudaStreamDestroy(copyStream));
	//for(auto& vec : data) {
	//	vec.clear();
	//}
}