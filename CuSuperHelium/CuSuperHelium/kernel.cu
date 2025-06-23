
//#define DEBUG_DERIVATIVES
//#define DEBUG_RUNGE_KUTTA
//#define DEBUG_VELOCITIES
#include "ProblemProperties.hpp"
#include "array"
#include "vector"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <cufft.h>

#include "constants.cuh"
#include <complex>

#include "TimeStepManager.cuh"
#include "SimpleEuler.cuh"
#include "RungeKunta.cuh"

#include <format>
//#include "math.h"
//#include "complex.h"
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

#define j_complex std::complex<double>(0, 1)
cudaError_t setDevice();
cudaError_t fftDerivative();


__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

double X(double j, double h, double omega, double t) {
    return j - h * std::sin((j - omega * t));
}

double Y(double j, double h, double omega, double t) {
	return h * std::cos((j - omega * t));
}

double Phi(double j, double h, double omega, double t, double rho) {
    return h * ((1 + rho) * omega * std::sin((j - omega * t)));
}

int runTimeStep() 
{

    cudaError_t cudaStatus;
    cudaStatus = setDevice();
    if (cudaStatus != cudaSuccess) {
        return cudaStatus;
    }

	ProblemProperties problemProperties;
    problemProperties.rho = 0;
	problemProperties.kappa = 0;
    problemProperties.U = 0;

    const int N = 16;
    const int steps = 10;
	double stepSize = 5e-2;
	WaterBoundaryIntegralCalculator<N> timeStepManager(problemProperties);

	std::array<double, N> j;
    std::vector<double> x0;
	std::vector<double> y0;

    std::array<cufftDoubleComplex, N> Z0;
	std::array<cufftDoubleComplex, N> PhiArr;
    std::vector<double> phiPrime;
    std::vector<double> phi0;

	std::array<cufftDoubleComplex, N> VelocitiesLower;
    std::array<cufftDoubleComplex, N> VelocitiesUpper;

    std::array<cufftDoubleComplex, N> ZVect;
    std::array<cufftDoubleComplex, N> PhiVect;



    std::vector<double> x;
	std::vector<double> y;

	x0.resize(N, 0);
	y0.resize(N, 0);
	phi0.resize(N, 0);
    x.resize(N, 0);
	y.resize(N, 0);
	phiPrime.resize(N, 0);
    double h = 0.1;
    double omega = 1;
    double t0 = 0;
	for (int i = 0; i < N; i++) {
		j[i] = 2.0 * PI_d * i / (1.0 * N);
		Z0[i].x = X(j[i], h, omega, t0);
		x0[i] = Z0[i].x;

		Z0[i].y = Y(j[i], h, omega, t0);
		y0[i] = Z0[i].y;

        PhiArr[i].x = Phi(j[i], h, omega, t0, problemProperties.rho);
		phi0[i] = PhiArr[i].x;

        PhiArr[i].y = 0; // Phi is real.
	}
    plt::figure();
    plt::title("Interface And Potential");
    plt::plot(x0, y0, {{"label", "Interface"}});
	plt::plot(x0, phi0, {{"label", "Potential"}});
    plt::legend();
    //plt::show();
    
	// Initialize the time step manager with the initial conditions.
	cuDoubleComplex* devZ = nullptr;
	cuDoubleComplex* devPhi = nullptr;

    timeStepManager.initialize_device(Z0.data(), PhiArr.data(), devZ, devPhi);
    /*timeStepManager.runTimeStep();
    cudaDeviceSynchronize();
	cudaMemcpy(phiPrime.data(), timeStepManager.devPhiPrime, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(VelocitiesLower.data(), timeStepManager.devVelocitiesLower, N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
    printf("\velocities after 1: ");
    for (int i = 0; i < N; i++) {
        printf("{%f, %f} ", VelocitiesLower[i].x, VelocitiesLower[i].y);
        x[i] = ZVect[i].x;
        y[i] = ZVect[i].y;
    }
    plt::figure();
    plt::plot(x0, phi0);
    plt::plot(x0, phiPrime);
    plt::show();*/
    // create Euler stepper
	RungeKuntaStepper<N> rungeKunta(timeStepManager, stepSize);
	// Euler<N> euler(timeStepManager, stepSize);
	/*euler.setDevZ(devZ);
	euler.setDevPhi(devPhi);*/
    rungeKunta.initialize(devZ);
	rungeKunta.setTimeStep(stepSize);
	for (int i = 0; i < steps; i++) {
        // Perform a time step
		rungeKunta.step();
	}
	

    // timeStepManager.runTimeStep();
	cudaDeviceSynchronize();

	cudaMemcpy(ZVect.data(), devZ, N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
    cudaMemcpy(PhiVect.data(), devPhi, N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
    cudaMemcpy(VelocitiesLower.data(), timeStepManager.devVelocitiesLower, N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
    //cudaMemcpy(PhiVect.data(), devPhi, N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);

	printf("\velocities after 1: ");
	double t = steps * stepSize;
	std::vector<double> x_fin(N, 0);
	std::vector<double> y_fin(N, 0);

	for (int i = 0; i < N; i++) {
		printf("{%f, %f} ", VelocitiesLower[i].x, VelocitiesLower[i].y);
        x[i] = ZVect[i].x;
        y[i] = ZVect[i].y;

		x_fin[i] = X(j[i], h, omega, t);
		y_fin[i] = Y(j[i], h, omega, t);
	}
	printf("\n");
    printf("\nPhi: ");
    for (int i = 0; i < N; i++) {
        printf("{%f, %f} ", PhiVect[i].x, -1 * PhiVect[i].y);
    }
    plt::figure();
    auto title = std::format("Interface And Potential at t={:.4f}", steps * stepSize);
	plt::title(title);

    //plt::plot(x_fin, y_fin, {{"label", "Interface at t=" + std::to_string(t)}});
    // Plot the initial position and the result of the Euler method

	plt::plot(x0, y0, {{"label", "Initial Position"}});
    plt::scatter(x, y);
    plt::legend();
    plt::show();
    

    return 0;
}

int main()
{
    //Py_SetPythonHome(L"C:/ProgramData/anaconda3");

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
