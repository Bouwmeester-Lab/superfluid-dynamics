#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

template <typename T>
class PingPongBuffer
{
public:
	PingPongBuffer();
	~PingPongBuffer();
	T* getBuffer() { return ping ? pingBuffer : pongBuffer; }
	void flipFlag() { ping = !ping; }
	void allocatePingBuffer(size_t size)
	{
		if (pingBuffer) cudaFree(pingBuffer);
		cudaMalloc(&pingBuffer, size * sizeof(T));
	}
	void allocatePongBuffer(size_t size)
	{
		if (pongBuffer) cudaFree(pongBuffer);
		cudaMalloc(&pongBuffer, size * sizeof(T));
	}
	void setPingBuffer(T* buffer)
	{
		if (pingBuffer) cudaFree(pingBuffer);
		pingBuffer = buffer;
	}
	void setPongBuffer(T* buffer)
	{
		if (pongBuffer) cudaFree(pongBuffer);
		pongBuffer = buffer;
	}
private:
	bool ping = 0; // true for ping, false for pong
	T* pingBuffer = nullptr;
	T* pongBuffer = nullptr;
};

template <typename T>
PingPongBuffer<T>::PingPongBuffer()
{
}

template <typename T>
PingPongBuffer<T>::~PingPongBuffer()
{
	if (pingBuffer) cudaFree(pingBuffer);
	if (pongBuffer) cudaFree(pongBuffer);
}
