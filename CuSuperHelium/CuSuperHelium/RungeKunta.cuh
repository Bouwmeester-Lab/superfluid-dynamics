#pragma once
#ifndef RUNGE_KUNTA_H
#define RUNGE_KUNTA_H

#include "ProblemProperties.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "TimeStepManager.cuh"
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

template <int N>
class RungeKuntaStepper
{
public:
	RungeKuntaStepper(TimeStepManager<N>& timeStepManager, double tstep = 1e-2);
	~RungeKuntaStepper();
	
	void setTimeStep(double tstep) { timeStep = make_cuDoubleComplex(tstep, 0); halfTimeStep = make_cuDoubleComplex(tstep * 0.5, 0); sixthTimeStep = make_cuDoubleComplex(tstep / 6.0, 0); }
	void initialize(cufftDoubleComplex* devZPhi0);
	void step();
private:
	const int threads = 256;
	const int blocks = (2 * N + threads - 1) / threads;

	cufftDoubleComplex* k1;
	cufftDoubleComplex* k2;
	cufftDoubleComplex* k3;
	cufftDoubleComplex* k4;

	cufftDoubleComplex* devZPhi0;
	cufftDoubleComplex* devZPhi1;
	cufftDoubleComplex* devZPhi2;
	cufftDoubleComplex* devZPhi3;
	//cufftDoubleComplex* devZphi4;

	TimeStepManager<N>& timeStepManager; ///< Instance of the TimeStepManager to handle time-stepping operations
	cublasHandle_t handle; ///< CUBLAS handle for matrix operations

	cuDoubleComplex timeStep;
	cuDoubleComplex halfTimeStep;
	cuDoubleComplex sixthTimeStep;
	
	void copyk(const int i);
};
template <int N>
RungeKuntaStepper<N>::RungeKuntaStepper(TimeStepManager<N>& timeStepManager, double tstep) : timeStepManager(timeStepManager)
{
	cublasCreate(&handle);
	cudaMalloc(&k1, 2 * N * sizeof(cufftDoubleComplex)); // allocate memory for k1
	cudaMalloc(&k2, 2 * N * sizeof(cufftDoubleComplex)); // allocate memory for k2
	cudaMalloc(&k3, 2 * N * sizeof(cufftDoubleComplex)); // allocate memory for k3
	cudaMalloc(&k4, 2 * N * sizeof(cufftDoubleComplex)); // allocate memory for k4

	cudaMalloc(&devZPhi1, 2 * N * sizeof(cufftDoubleComplex)); // allocate memory for devZphi1
	cudaMalloc(&devZPhi2, 2 * N * sizeof(cufftDoubleComplex)); // allocate memory for devZphi2
	cudaMalloc(&devZPhi3, 2 * N * sizeof(cufftDoubleComplex)); // allocate memory for devZphi3

	setTimeStep(tstep);
}

template <int N>
RungeKuntaStepper<N>::~RungeKuntaStepper()
{
}

template<int N>
void RungeKuntaStepper<N>::step()
{
	timeStepManager.setZPhi(devZPhi0);
	timeStepManager.runTimeStep();

	copyk(1); // copy k1 from the timeStepManager

	
	// x_n+1 = x_n + h * v_n -> x_n is a N dimensional vector, h is the time step, v_n is the velocity at time n
	cublasZaxpy(handle, 2 * N, &halfTimeStep, k1, 1, devZPhi1, 1); // here devZ get's overwritten with the new positions

	timeStepManager.setZPhi(devZPhi1);
	timeStepManager.runTimeStep();

	copyk(2); // copy k2 from the timeStepManager
	cublasZaxpy(handle, 2 * N, &halfTimeStep, k2, 1, devZPhi2, 1); // here devZ get's overwritten with the new positions

	timeStepManager.setZPhi(devZPhi2);
	timeStepManager.runTimeStep();

	copyk(3); // copy k3 from the timeStepManager
	cublasZaxpy(handle, 2 * N, &timeStep, k3, 1, devZPhi3, 1); // here devZ get's overwritten with the new positions

	timeStepManager.setZPhi(devZPhi3);
	timeStepManager.runTimeStep();

	copyk(4); // copy k4 from the timeStepManager
	
	add_k_vectors << <blocks, threads >> > (k1, k2, k3, k4, k1, 2 * N); // add the four vectors together

	//cudaDeviceSynchronize(); // synchronize the device to ensure all operations are completed

	//std::vector<cufftDoubleComplex> k1_host(2 * N);

	//cudaMemcpy(k1_host.data(), k1, 2 * N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost); // copy k1 to host for debugging
	//printf("%f h/6\n", cuCreal(sixthTimeStep)); // print the time step for debugging
	//for(int i = 0; i < 2*N; i++) {
	//	printf("k[%d] = (%f, %f)\n", i, cuCreal(k1_host[i]), cuCimag(k1_host[i]));
	//}
	//std::cin.get(); // wait for user input to continue
	cublasZaxpy(handle, N, &sixthTimeStep , k1, 1, devZPhi0, 1); // here devZ get's overwritten with the new positions

	//cudaDeviceSynchronize(); // synchronize the device to ensure all operations are completed

	//std::vector<cufftDoubleComplex> devZPhi0_host(2 * N);
	//std::vector<double> x(N);
	//std::vector<double> y(N);
	//std::vector<double> Phi(N);
	//std::vector<double> dx(N);
	//std::vector<double> dy(N);
	//std::vector<double> dPhi(N);

	//cudaMemcpy(devZPhi0_host.data(), devZPhi0, 2 * N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost); // copy devZPhi0 to host for debugging

	//for(int i = 0; i < N; i++) {
	//	x[i] = cuCreal(devZPhi0_host[i]);
	//	y[i] = cuCimag(devZPhi0_host[i]);
	//	Phi[i] = cuCreal(devZPhi0_host[i + N]); // assuming the second half of the array contains Phi values

	//	dx[i] = cuCreal(k1_host[i]);
	//	dy[i] = cuCimag(k1_host[i]);
	//	dPhi[i] = cuCreal(k1_host[i + N]); // assuming the second half of k1 contains Phi values
	//}

	//plt::figure();
	//plt::plot(x, y, "ro-"); // plot the positions
	//plt::plot(x, Phi, "b-"); // plot the Phi values
	//plt::xlabel("x");
	//plt::ylabel("y");
	//plt::title("Positions after Runge-Kutta Step");

	//plt::figure();
	//plt::plot(x, dx); // plot the x component of k1
	//plt::plot(x, dy); // plot the y component of k1
	//plt::plot(x, dPhi); // plot the Phi component of k1

	cudaMemcpy(devZPhi1, devZPhi0, 2 * N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice); // copy initial state to devZPhi1
	cudaMemcpy(devZPhi2, devZPhi0, 2 * N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice); // copy initial state to devZPhi2
	cudaMemcpy(devZPhi3, devZPhi0, 2 * N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice); // copy initial state to devZPhi3

	cudaDeviceSynchronize(); // synchronize the device to ensure all operations are completed
}

template<int N>
void RungeKuntaStepper<N>::initialize(cufftDoubleComplex* devZPhi0)
{
	this->devZPhi0 = devZPhi0; // store the initial state

	cudaMemcpy(devZPhi1, devZPhi0, 2 * N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice); // copy initial state to devZPhi1
	cudaMemcpy(devZPhi2, devZPhi0, 2 * N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice); // copy initial state to devZPhi2
	cudaMemcpy(devZPhi3, devZPhi0, 2 * N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice); // copy initial state to devZPhi3
	//cudaMemcpy(devZPhi4, devZPhi0, 2 * N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice); // copy initial state to devZPhi4
}

template<int N>
void RungeKuntaStepper<N>::copyk(const int i)
{
	if (i == 1) {
		cudaMemcpy(k1, timeStepManager.devVelocitiesLower, N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice);
		cudaMemcpy(k1 + N, timeStepManager.devRhsPhi, N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice);
	}
	else if (i == 2) {
		cudaMemcpy(k2, timeStepManager.devVelocitiesLower, N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice);
		cudaMemcpy(k2 + N, timeStepManager.devRhsPhi, N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice);
	}
	else if (i == 3) {
		cudaMemcpy(k3, timeStepManager.devVelocitiesLower, N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice);
		cudaMemcpy(k3 + N, timeStepManager.devRhsPhi, N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice);
	}
	else if (i == 4) {
		cudaMemcpy(k4, timeStepManager.devVelocitiesLower, N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice);
		cudaMemcpy(k4 + N, timeStepManager.devRhsPhi, N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice);
	}
}

#endif