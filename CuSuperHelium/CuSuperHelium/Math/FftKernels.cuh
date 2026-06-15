#pragma once
#ifndef FFT_KERNELS_H
#define FFT_KERNELS_H

#include "../constants.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cufft.h>

/// <summary>
/// Multiplies the coefficients of two complex vectors element-wise
/// </summary>
/// <param name="a"></param>
/// <param name="b"></param>
/// <param name="result"></param>
/// <param name="n"></param>
/// <returns></returns>
__global__ void first_derivative_multiplication(
    const cufftDoubleComplex* a,
    cufftDoubleComplex* result,
    const int n,
    const int batchSize = 1
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n * batchSize) return; // out of bounds

    int  i = tid % n; // index within the vector
    double x, y;
    if (i < n / 2)
    {
        x = a[tid].x; // it's important to do this copy since if you don't you will be using the new value in the expression instead of the old one.
        y = a[tid].y;
        result[tid].x = -i * y / static_cast<double>(n);
        result[tid].y = i * x / static_cast<double>(n);
    }
    else if (i == n / 2)
    {
        x = a[tid].x;
        result[tid].x = -PI_d * i * a[tid].y / static_cast<double>(n);
        result[tid].y = PI_d * i * x / static_cast<double>(n); // -PI_d * a[i].x / n; // we want to treat the Nyquist frequency as exp(i*pi*j) which means
        // that the inverse fft of the fft of exp(i*pi*j) should give i pi * exp(i * pi *j). This happens when the coeff[n/2] = -pi.
        // Usually this coefficient should be -pi *n but cuFFT will NOT normalize by n, so when we do normalize manually by dividing by n, we get -pi.
    }
    else if (i == n / 2 + 1)
    {
        x = a[tid].x;
        y = a[tid].y;
        // this is the Nyquist frequency, we want to set it to zero since we don't want to have any imaginary part in the result
        result[tid].x = 0;// -PI_d * i * y / static_cast<double>(n);
        result[tid].y = 0; //-PI_d * i * x / static_cast<double>(n);// PI_d* x / n; // this is the same as setting the imaginary part to zero
    }
    else if (i < n) {
        x = a[tid].x;
        y = a[tid].y;
        result[tid].x = -(i - n) * y / static_cast<double>(n); // - because you multiply by i (imaginary i) => z = x+iy => iz = ix - y
        result[tid].y = (i - n) * x / static_cast<double>(n); // https://math.mit.edu/~stevenj/fft-deriv.pdf
    }
}

__global__ void multiply_element_wise(const cufftDoubleComplex* a, cufftDoubleComplex* b, cufftDoubleComplex* result, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        result[i] = cuCmul(a[i], b[i]);
    }
}

/// <summary>
/// Filters the coefficients using a tanh filter. This is used to remove high frequency noise from the signal. It assumes that the coefficients are in the order of the frequencies of [0, N/2, -N/2, ..., -1].
/// </summary>
/// <param name="k"></param>
/// <param name="N"></param>
/// <param name="eps"></param>
/// <param name="d"></param>
/// <returns></returns>
__device__ __host__ inline double filterTanh(int k, int N, double eps, double d)
{
    return 0.5 + 0.5 * tanh((2.0 * PI_d / N * k - eps) / d);
}

/// <summary>
/// Calculates the coefficients needed to run an ifft and get the second derivative from the original function: d2/dt f(t)
/// </summary>
/// <param name="coeffsFft"></param>
/// <param name="result"></param>
/// <param name="n"></param>
/// <returns></returns>
__global__ void second_derivative_fft(const cufftDoubleComplex* coeffsFft, cufftDoubleComplex* result, const int n, const int batchSize = 1)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n * batchSize) return; // out of bounds

    int i = tid % n; // index within the vector
    if (i < n / 2)
    {
        result[tid].x = -i * i * coeffsFft[tid].x / (n);
        result[tid].y = -i * i * coeffsFft[tid].y / (n);
    }
    else if (i == n / 2)
    {
        result[tid].x = -i * i * coeffsFft[tid].x / (n); // real part, this is the same idea as in the first derivative. We want the second derivative of exp(i*pi*j) to be -pi^2 * exp(i*pi*j), this will happen only if this coefficient is this value
        result[tid].y = -i * i * coeffsFft[tid].y / (n);// setting this to zero seems fine? although I wonder if it's better to leave the same value as the theoretical value: N/2 * N/2 * coeffsFft[i].y / N ?
    }
    else if (i < n)
    {
        result[tid].x = -(i - n) * (i - n) * coeffsFft[tid].x / (n); //https://math.mit.edu/~stevenj/fft-deriv.pdf
        result[tid].y = -(i - n) * (i - n) * coeffsFft[tid].y / (n);
    }
}

__global__ void force_real_only(cufftDoubleComplex* a, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        a[i].y = 0; // set the imaginary part to 0
    }
}

__global__ void set_mode_to_imaginary(cufftDoubleComplex* a, double img, int n)
{
    a[n].x = 0; // set the real part of the Nyquist frequency to 0
    a[n].y = img; // set the imaginary part of the Nyquist frequency to 1
}

#endif
