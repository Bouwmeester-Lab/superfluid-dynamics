#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>

template <typename T, size_t N>
class DataLogger
{
public:
	DataLogger();
	~DataLogger();

	void setSize(size_t size, bool allocateTimes = true)
	{
		this->size = size;
		data.resize(size, std::vector<T>(N));
		if (allocateTimes) {
			times.resize(size, 0.0);
		}
	}
	void setStep(size_t steps)
	{
		this->steps = steps;
	}
	void setReadyToCopy(T* devPointer, cudaStream_t stream = cudaStreamPerThread, double time = -1, bool logTimes = false) 
	{
		cudaEventRecord(readyToCopy, stream);
		copyScheduled = true;

		cudaStreamWaitEvent(copyStream, readyToCopy, 0);
		if (logTimes) {
			times[currentIndex] = time;
		}
		cudaMemcpyAsync(data.at(currentIndex++).data(), devPointer, N * sizeof(T), cudaMemcpyDeviceToHost, copyStream);
		
		cudaEventRecord(copyDone, copyStream);
	}

	bool isCopyScheduled() const
	{
		return copyScheduled;
	}

	void waitForCopy(cudaStream_t stream = cudaStreamPerThread)
	{
		if (copyScheduled) {
			cudaStreamWaitEvent(stream, copyDone, 0);
			copyScheduled = false;
		}
	}

	bool shouldCopy(const size_t step) const
	{
		return step % steps == 0 && currentIndex < size;
	}

	std::vector<T> getData(const size_t index) const
	{
		if (index >= size) {
			throw std::out_of_range("Index out of range in DataLogger");
		}
		return data.at(index);
	}

	std::vector<std::vector<T>>& getAllData()
	{
		return data;
	}

	std::vector<double>& getTimes() {
		return times;
	}

private:
	cudaEvent_t readyToCopy, copyDone;
	cudaStream_t copyStream;
	size_t size;

	std::vector<std::vector<T>> data;
	std::vector<double> times;

	bool copyScheduled = false;
	size_t currentIndex = 0;
	size_t steps = 1;
};

template <typename T, size_t N>
DataLogger<T, N>::DataLogger()
{
	cudaEventCreate(&readyToCopy);
	cudaEventCreate(&copyDone);
	cudaStreamCreate(&copyStream);
	setSize(10);
}
template <typename T, size_t N>
DataLogger<T, N>::~DataLogger()
{
	cudaEventDestroy(readyToCopy);
	cudaEventDestroy(copyDone);
	cudaStreamDestroy(copyStream);
}