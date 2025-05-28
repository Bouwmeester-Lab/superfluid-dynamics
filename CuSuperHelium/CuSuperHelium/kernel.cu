
#include "ProblemProperties.hpp"
#include "array"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <cufft.h>

#include "constants.cuh"
#include <complex>

#include "TimeStepManager.cuh"

//#include "math.h"
//#include "complex.h"

#define j_complex std::complex<double>(0, 1)
cudaError_t setDevice();
cudaError_t fftDerivative();


__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

double X(double j, double h, double omega, double t) {
    return j - h * std::exp(j_complex * (j - omega * t)).imag();
}

double Y(double j, double h, double omega, double t) {
	return h * std::exp(j_complex * (j - omega * t)).real();
}

double Phi(double j, double h, double omega, double t, double rho) {
    return h * ((1 + rho) * omega * std::exp(j_complex * (j - omega * t))).imag();
}

int runTimeStep() 
{
	ProblemProperties problemProperties;
    problemProperties.rho = 0;
	problemProperties.kappa = 0;
    problemProperties.U = 0;
    const int N = 32;
	TimeStepManager<N> timeStepManager(problemProperties);

	std::array<double, N> j;
    std::array<cufftDoubleComplex, N> Z0;
	std::array<cufftDoubleComplex, N> PhiArr;

	std::array<cufftDoubleComplex, N> VelocitiesLower;

	for (int i = 0; i < N; i++) {
		j[i] = 2 * PI_d * i / (1.0 * N);
		Z0[i].x = X(j[i], 0.2, 0.1, 0);
		Z0[i].y = Y(j[i], 0.2, 0.1, 0);
        PhiArr[i].x = Phi(j[i], 0.2, 0.1, 0, problemProperties.rho);
        PhiArr[i].y = 0; // Phi is real.
	}
    
	// Initialize the time step manager with the initial conditions.

    timeStepManager.initialize_device(Z0.data(), PhiArr.data());
    timeStepManager.runTimeStep();
	cudaDeviceSynchronize();

	cudaMemcpy(VelocitiesLower.data(), timeStepManager.devVelocitiesLower, N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);

	printf("VelocitiesLower: ");
	for (int i = 0; i < N; i++) {
		printf("{%f, %f} ", VelocitiesLower[i].x, -1*VelocitiesLower[i].y);
	}
	printf("\n");

    return 0;
}

int main()
{

    runTimeStep();
    //const int arraySize = 8;
    //const int a[arraySize] = { 1, 2, 3, 4, 5, 6, 7 };
    //const int b[arraySize] = { 10, 20, 30, 40, 50, 60, 70 };

    //const double X0[arraySize] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    //const double Y0[arraySize] = { 0, 1, 2, 3, 4, 5, 6, 7 };

    //int c[arraySize] = { 0 };

    //// test fft derivative
    //cudaError_t cudaStatus = fftDerivative();
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "addWithCuda failed!");
    //    return 1;
    //}

    //

    //// cudaDeviceReset must be called before exiting in order for profiling and
    //// tracing tools such as Nsight and Visual Profiler to show complete traces.
    //cudaStatus = cudaDeviceReset();
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaDeviceReset failed!");
    //    return 1;
    //}

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
	return cudaStatus;
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
