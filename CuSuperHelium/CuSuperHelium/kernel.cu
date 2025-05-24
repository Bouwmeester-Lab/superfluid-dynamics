
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "Derivatives.hpp"
#include <cufft.h>
#include "array"
#include "constants.cuh"

cudaError_t setDevice();
cudaError_t fftDerivative();

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
    const int arraySize = 8;
    const int a[arraySize] = { 1, 2, 3, 4, 5, 6, 7 };
    const int b[arraySize] = { 10, 20, 30, 40, 50, 60, 70 };

    const double X0[arraySize] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    const double Y0[arraySize] = { 0, 1, 2, 3, 4, 5, 6, 7 };

    int c[arraySize] = { 0 };

    // test fft derivative
    cudaError_t cudaStatus = fftDerivative();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

cudaError_t fftDerivative()
{
    cudaError_t cudaStatus;
    cudaStatus = setDevice();
    if (cudaStatus != cudaSuccess) {
        return cudaStatus;
    }
    

    const int N = 16;

    std::array<cufftDoubleComplex, N> y;
    std::array<cufftDoubleComplex, N> yp;
    cufftDoubleComplex* devY;
    cufftDoubleComplex* devYp;

    for (int i = 0; i < N; i++) 
    {
        y[i].x = sin(2.0 * 2.0 * i * PI_d / (1.0*N));
        y[i].y = 0;
    }

    FftDerivative<N, 1> derivativeFft;

    cudaStatus = derivativeFft.initialize();

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "failed to initialize the fft derivatives! ");
        return cudaStatus;
    }
    
    cudaStatus = cudaMalloc(&devY, sizeof(cufftDoubleComplex) * N);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "failed to allocate devY to gpu! ");
        return cudaStatus;
    }
    // copy y to the gpu
    cudaStatus = cudaMemcpy(devY, y.data(), sizeof(cufftDoubleComplex) * N, cudaMemcpyHostToDevice);
    cudaStatus = cudaMalloc(&devYp, sizeof(cufftDoubleComplex) * N);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "failed to copy to gpu! ");
        return cudaStatus;
    }

    derivativeFft.exec(devY, devYp);

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        return cudaStatus;
    }

    cudaStatus = cudaMemcpy(yp.data(), devYp, N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return cudaStatus;
    }
    printf("end");
    printf("d/dj y {%d, %d, %d, ... } = {%d,%d,%d, ...}\n",
        y[0].x, y[1].x, y[2].x, yp[0].x, yp[1].x, yp[2].x);
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t setDevice()
{
    
    cudaError_t cudaStatus;
    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return cudaStatus;
    }

    return cudaStatus;
    
    
//    // cudaDeviceSynchronize waits for the kernel to finish, and returns
//    // any errors encountered during the launch.
//    cudaStatus = cudaDeviceSynchronize();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
//        goto Error;
//    }
//
//    // Copy output vector from GPU buffer to host memory.
//    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//Error:
//    cudaFree(dev_c);
//    cudaFree(dev_a);
//    cudaFree(dev_b);
//    
//    return cudaStatus;
}
