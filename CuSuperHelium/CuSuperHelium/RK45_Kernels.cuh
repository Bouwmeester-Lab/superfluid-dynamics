#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "Reductions.cuh"
#ifdef __INTELLISENSE__ // for Visual Studio IntelliSense https://stackoverflow.com/questions/77769389/intellisense-in-visual-studio-cannot-find-cuda-cooperative-groups-namespace
#define __CUDACC__
#endif // __INTELLISENSE__
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#ifdef __INTELLISENSE__
#undef __CUDACC__
#endif // __INTELLISENSE__
namespace cg = cooperative_groups;

struct RK45Coefficients {
	// Coefficients for the RK45 method
	// a_ij coefficients
	static constexpr double a21 = 1.0 / 4.0;
	static constexpr double a31 = 3.0 / 32.0;
	static constexpr double a32 = 9.0 / 32.0;
	static constexpr double a41 = 1932.0 / 2197.0;
	static constexpr double a42 = -7200.0 / 2197.0;
	static constexpr double a43 = 7296.0 / 2197.0;
	static constexpr double a51 = 439.0 / 216.0;
	static constexpr double a52 = -8.0;
	static constexpr double a53 = 3680.0 / 513.0;
	static constexpr double a54 = -845.0 / 4104.0;
	static constexpr double a61 = -8.0 / 27.0;
	static constexpr double a62 = 2.0;
	static constexpr double a63 = -3544.0 / 2565.0;
	static constexpr double a64 = 1859.0 / 4104.0;
	static constexpr double a65 = -11.0 / 40.0;
	// b_i coefficients for the 5th order solution
	static constexpr double b1 = 16.0 / 135.0;
	static constexpr double b2 = 0.0;
	static constexpr double b3 = 6656.0 / 12825.0;
	static constexpr double b4 = 28561.0 / 56430.0;
	static constexpr double b5 = -9.0 / 50.0;
	static constexpr double b6 = 2.0 / 55.0;
	// b*_i coefficients for the 4th order solution
	static constexpr double b1s = 25.0 / 216.0;
	static constexpr double b2s = 0.0;
	static constexpr double b3s = 1408.0 / 2565.0;
	static constexpr double b4s = 2197.0 / 4104.0;
	static constexpr double b5s = -1.0 / 5.0;
	static constexpr double b6s = 0.0;
	// difference bi - b*i:
	static constexpr double d1 = b1 - b1s;
	static constexpr double d2 = b2 - b2s;
	static constexpr double d3 = b3 - b3s;
	static constexpr double d4 = b4 - b4s;
	static constexpr double d5 = b5 - b5s;
	static constexpr double d6 = b6 - b6s;

	// c_i coefficients (nodes)
	static constexpr double c1 = 0;
};



template <class T, size_t BLOCK_SIZE>
__global__ void rk45_error_and_y5(
	const T* __restrict__ y,
	const T* __restrict__ k1, const T* __restrict__ k2, const T* __restrict__ k3,
	const T* __restrict__ k4, const T* __restrict__ k5, const T* __restrict__ k6,
	T* __restrict__ y5_out,                // write y5 here (your yTemp)
	double h,
	double atol, double rtol,
	// reduction outputs
	double* __restrict__ sumsq,
	unsigned int N)
{
	auto block = cg::this_thread_block();
	double local_sumsq = 0.0;
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
		i < N; i += blockDim.x * gridDim.x)
	{
		// y5
		T y5 = y[i]
			+ (T)(h * RK45Coefficients::b1) * k1[i] + (T)(h * RK45Coefficients::b2) * k2[i] + (T)(h * RK45Coefficients::b3) * k3[i]
			+ (T)(h * RK45Coefficients::b4) * k4[i] + (T)(h * RK45Coefficients::b5) * k5[i] + (T)(h * RK45Coefficients::b6) * k6[i];

		y5_out[i] = y5;

		// e = h * sum d_j k_j
		T e = (T)(h * RK45Coefficients::d1) * k1[i] + (T)(h *(RK45Coefficients::d2)) * k2[i] + (T)(h * RK45Coefficients::d3) * k3[i]
			+ (T)(h * RK45Coefficients::d4) * k4[i] + (T)(h * RK45Coefficients::d5) * k5[i] + (T)(h * RK45Coefficients::d6) * k6[i];

		double sc = atol + rtol * fmax(mag(y[i]), mag(y5));
		//printf("%i: k1[i] %.5e k2[i] %.5e, k3 %.5e k4 %.5e k5 %.5e k6 %.5e \n", i, k1[i], k2[i], k3[i], k4[i], k5[i], k6[i]);
		//printf("%i: N: %d, e %.5e sc %.5e abs y5 %.5e z %.5e\n",  i, N, mag(e), sc, mag(y5), z);
		// guard against sc=0 if both are exactly zero
		// this is the scale for the relative error
		sc = fmax(sc, 1e-300);

		double z = mag(e) / sc;
		local_sumsq += z * z;
	}

	double block_sum = block_reduce_sum<BLOCK_SIZE, cg::plus<double>>(local_sumsq);
	//printf("%i: block_sum: %.5f global: %.5e \n", id, block_sum, *sumsq);
	if (block.thread_rank() == 0) {
		if (sumsq) atomicAdd(sumsq, block_sum);
	}
}