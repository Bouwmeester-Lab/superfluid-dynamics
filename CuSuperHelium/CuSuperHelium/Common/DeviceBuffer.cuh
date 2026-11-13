#pragma once
#ifndef DEVICE_BUFFER_H
#define DEVICE_BUFFER_H

#include "cuda_runtime.h"
#include <cstddef>
#include <utility>

#include "CudaChecks.cuh"

/// <summary>
/// Owns a device array of `count` elements of T, allocated with cudaMalloc and freed when the buffer goes out of
/// scope, including when an exception unwinds the stack. Move-only. Allocation failures throw std::runtime_error
/// (CHECK_CUDA). The memory is not initialized.
/// </summary>
template <typename T>
class DeviceBuffer
{
public:
	DeviceBuffer() = default;

	explicit DeviceBuffer(size_t count) : count_(count)
	{
		if (count_ > 0)
		{
			CHECK_CUDA(cudaMalloc(&ptr_, bytes()));
		}
	}

	~DeviceBuffer() { reset(); }

	DeviceBuffer(const DeviceBuffer&) = delete;
	DeviceBuffer& operator=(const DeviceBuffer&) = delete;

	DeviceBuffer(DeviceBuffer&& other) noexcept
		: ptr_(std::exchange(other.ptr_, nullptr)), count_(std::exchange(other.count_, 0)) {}

	DeviceBuffer& operator=(DeviceBuffer&& other) noexcept
	{
		if (this != &other)
		{
			reset();
			ptr_ = std::exchange(other.ptr_, nullptr);
			count_ = std::exchange(other.count_, 0);
		}
		return *this;
	}

	T* get() const noexcept { return ptr_; }
	size_t size() const noexcept { return count_; }
	size_t bytes() const noexcept { return count_ * sizeof(T); }

	/// <summary>
	/// Frees the memory now. Errors from cudaFree are ignored, since this also runs from the destructor.
	/// </summary>
	void reset() noexcept
	{
		if (ptr_ != nullptr)
		{
			cudaFree(ptr_);
			ptr_ = nullptr;
		}
		count_ = 0;
	}

private:
	T* ptr_ = nullptr;
	size_t count_ = 0;
};

#endif // DEVICE_BUFFER_H
