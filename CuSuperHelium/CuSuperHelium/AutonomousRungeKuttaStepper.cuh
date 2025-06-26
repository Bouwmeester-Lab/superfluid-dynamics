#pragma once
#ifndef RUNGE_KUNTA_H
#define RUNGE_KUNTA_H

#include "ProblemProperties.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "AutonomousProblem.h"
#include "matplotlibcpp.h"
#include "cuDoubleComplexOperators.cuh"
#include "cublas_v2.h"
namespace plt = matplotlibcpp;

template <typename T, int N>
class AutonomousRungeKuttaStepperBase
{
public:
	AutonomousRungeKuttaStepperBase(AutonomousProblem<T, N>& autonomousProblem, double tstep = 1e-2);
	~AutonomousRungeKuttaStepperBase();
	
	void setTimeStep(double tstep) 
	{ 
		timeStep = CastTo<T>(tstep);
		halfTimeStep = CastTo<T>(tstep * 0.5);
		sixthTimeStep = CastTo<T>(tstep / 6.0);
	}

	void initialize(T* devY0);
	void runStep();
protected:
	const int threads = 256;
	const int blocks = (N + threads - 1) / threads;

	T* k1;
	T* k2;
	T* k3;
	T* k4;

	T* devY0;
	T* devY1;
	T* devY2;
	T* devY3;

	AutonomousProblem<T, N>& autonomousProblem; ///< Instance of the TimeStepManager to handle time-stepping operations
	cublasHandle_t handle; ///< CUBLAS handle for matrix operations

	T timeStep;
	T halfTimeStep;
	T sixthTimeStep;
	
	void copyk(const int i);
	virtual void step(const int i) = 0;
};

template <typename T, int N>
AutonomousRungeKuttaStepperBase<T, N>::AutonomousRungeKuttaStepperBase(AutonomousProblem<T, N>& autonomousProblem, double tstep) : autonomousProblem(autonomousProblem)
{
	cublasCreate(&handle);
	cudaMalloc(&k1, N * sizeof(T)); // allocate memory for k1
	cudaMalloc(&k2, N * sizeof(T)); // allocate memory for k2
	cudaMalloc(&k3, N * sizeof(T)); // allocate memory for k3
	cudaMalloc(&k4, N * sizeof(T)); // allocate memory for k4

	cudaMalloc(&devY1, N * sizeof(T)); // allocate memory for devZphi1
	cudaMalloc(&devY2, N * sizeof(T)); // allocate memory for devZphi2
	cudaMalloc(&devY3, N * sizeof(T)); // allocate memory for devZphi3

	setTimeStep(tstep);
}

template <typename T,int N>
AutonomousRungeKuttaStepperBase<T, N>::~AutonomousRungeKuttaStepperBase()
{
}

template <typename T, int N>
void AutonomousRungeKuttaStepperBase<T, N>::runStep()
{
	autonomousProblem.run(devY0);

	copyk(1); // copy k1 from the autonomousProblem
#ifdef DEBUG_RUNGE_KUTTA
	cudaDeviceSynchronize(); // synchronize the device to ensure all operations are completed
	std::vector<cufftDoubleComplex> k1_host(N);
	std::vector<cufftDoubleComplex> y_host(N);

	std::vector<double> x(N);
	std::vector<double> y(N);
	std::vector<double> Phi(N);
	std::vector<double> dx(N);
	std::vector<double> dy(N);
	std::vector<double> dPhi(N);

	cudaMemcpy(k1_host.data(), k1, N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost); // copy k1 to host for debugging
	cudaMemcpy(y_host.data(), devY0, N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost); // copy devY0 to host for debugging

	for (int i = 0; i < N; i++) {
		x[i] = cuCreal(y_host[i]);
		y[i] = cuCimag(y_host[i]);
		// Phi[i] = cuCreal(y_host[i + N]); // assuming the second half of the array contains Phi values

		dx[i] = x[i] + halfTimeStep.x * cuCreal(k1_host[i]);
		dy[i] = y[i] + halfTimeStep.x * cuCimag(k1_host[i]);
		// dPhi[i] = Phi[i] + halfTimeStep.x * cuCreal(k1_host[i + N]); // assuming the second half of k1 contains Phi values
	}

	plt::figure();
	plt::plot(x, y, { {"label", "initial pos"} }); // plot the positions
	//plt::plot(x, Phi, { {"label", "initial phi"} }); // plot the Phi values
	plt::plot(dx, dy, { {"label", "perturbed pos"} });
	//plt::plot(x, dy, { {"label", "k1 y + dy"} });
	// plt::plot(dx, dPhi, { {"label", "k1 phi + phi"} }); // plot the Phi component of k1

	/*plt::xlabel("x");
	plt::ylabel("y");*/
	plt::legend();
	plt::title("Positions after Runge-Kutta Step");

#endif
	// x_n+1 = x_n + h*0.5 * v_n -> x_n is a N dimensional vector, h is the time step, v_n is the velocity at time n
	step(1); // step 1 of the Runge-Kutta method

#ifdef DEBUG_RUNGE_KUTTA
	cudaDeviceSynchronize(); // synchronize the device to ensure all operations are completed
	cudaMemcpy(y_host.data(), devY1, N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost); // copy devY1 to host for debugging

	for (int i = 0; i < N; i++) {
		x[i] = cuCreal(y_host[i]);
		y[i] = cuCimag(y_host[i]);
		//Phi[i] = cuCreal(y_host[i + N]); // assuming the second half of the array contains Phi values
		//dx[i] = x[i] + halfTimeStep.x * cuCreal(k1_host[i]);
		//dy[i] = y[i] + halfTimeStep.x * cuCimag(k1_host[i]);
		//dPhi[i] = Phi[i] + halfTimeStep.x * cuCreal(k1_host[i + N]); // assuming the second half of k1 contains Phi values
	}
	plt::figure();
	plt::plot(x, y, { {"label", "pertubed calculated positions"} }); // plot the positions
	//plt::plot(x, Phi, { {"label", "pertubed calculated phi"} }); // plot the Phi values

	
#endif

	autonomousProblem.run(devY1);

	copyk(2); // copy k2 from the autonomousProblem

	step(2); // step 2 of the Runge-Kutta method

#ifdef DEBUG_RUNGE_KUTTA
	cudaDeviceSynchronize(); // synchronize the device to ensure all operations are completed
	cudaMemcpy(y_host.data(), devY2,  N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost); // copy devY1 to host for debugging

	for (int i = 0; i < N; i++) {
		x[i] = cuCreal(y_host[i]);
		y[i] = cuCimag(y_host[i]);
		//Phi[i] = cuCreal(y_host[i + N]); // assuming the second half of the array contains Phi values
		//dx[i] = x[i] + halfTimeStep.x * cuCreal(k1_host[i]);
		//dy[i] = y[i] + halfTimeStep.x * cuCimag(k1_host[i]);
		//dPhi[i] = Phi[i] + halfTimeStep.x * cuCreal(k1_host[i + N]); // assuming the second half of k1 contains Phi values
	}

	plt::plot(x, y, { {"label", "3rd pertubed calculated positions"} }); // plot the positions
	//plt::plot(x, Phi, { {"label", "3rd pertubed calculated phi"} }); // plot the Phi values

	//plt::legend();
#endif

	autonomousProblem.run(devY2);

	copyk(3); // copy k3 from the autonomousProblem
	
	step(3);

	autonomousProblem.run(devY3);

	copyk(4); // copy k4 from the autonomousProblem

#ifdef DEBUG_RUNGE_KUTTA
	cudaDeviceSynchronize(); // synchronize the device to ensure all operations are completed
	cudaMemcpy(y_host.data(), devY3, N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost); // copy devY1 to host for debugging

	for (int i = 0; i < N; i++) {
		x[i] = cuCreal(y_host[i]);
		y[i] = cuCimag(y_host[i]);
		//Phi[i] = cuCreal(y_host[i + N]); // assuming the second half of the array contains Phi values
		//dx[i] = x[i] + halfTimeStep.x * cuCreal(k1_host[i]);
		//dy[i] = y[i] + halfTimeStep.x * cuCimag(k1_host[i]);
		//dPhi[i] = Phi[i] + halfTimeStep.x * cuCreal(k1_host[i + N]); // assuming the second half of k1 contains Phi values
	}

	plt::plot(x, y, { {"label", "4th pertubed calculated positions"} }); // plot the positions
	//plt::plot(x, Phi, { {"label", "4th pertubed calculated phi"} }); // plot the Phi values

	plt::legend();
#endif
	
	add_k_vectors << <blocks, threads >> > (k1, k2, k3, k4, k1, N); // add the four vectors together

	//cudaDeviceSynchronize(); // synchronize the device to ensure all operations are completed

	//std::vector<cufftDoubleComplex> k1_host(2 * N);

	//cudaMemcpy(k1_host.data(), k1, 2 * N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost); // copy k1 to host for debugging
	//printf("%f h/6\n", cuCreal(sixthTimeStep)); // print the time step for debugging
	//for(int i = 0; i < 2*N; i++) {
	//	printf("k[%d] = (%f, %f)\n", i, cuCreal(k1_host[i]), cuCimag(k1_host[i]));
	//}
	//std::cin.get(); // wait for user input to continue
#ifdef DEBUG_RUNGE_KUTTA
	cudaDeviceSynchronize(); // synchronize the device to ensure all operations are completed
	cudaMemcpy(y_host.data(), devY0, N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost); // copy devY1 to host for debugging
	cudaMemcpy(k1_host.data(), k1, N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; i++) {
		x[i] = cuCreal(y_host[i]); // +sixthTimeStep.x * k1_host[i].x;
		y[i] = cuCimag(y_host[i]);// +sixthTimeStep.x * k1_host[i].y;
		//Phi[i] = cuCreal(y_host[i + N]);// +sixthTimeStep.x * k1_host[i + N].x; // assuming the second half of the array contains Phi values
		//dx[i] = x[i] + halfTimeStep.x * cuCreal(k1_host[i]);
		//dy[i] = y[i] + halfTimeStep.x * cuCimag(k1_host[i]);
		//dPhi[i] = Phi[i] + halfTimeStep.x * cuCreal(k1_host[i + N]); // assuming the second half of k1 contains Phi values
	}
	plt::figure();
	plt::plot(x, y, { {"label", "Initial"}}); // plot the positions
	//plt::plot(x, Phi, { {"label", "Initial phi"} }); // plot the Phi values

#endif

	step(4); // step 4 of the Runge-Kutta method, which combines the results of the previous steps, it should overwrite devY0 with the final result

#ifdef DEBUG_RUNGE_KUTTA
	cudaDeviceSynchronize(); // synchronize the device to ensure all operations are completed
	cudaMemcpy(y_host.data(), devY0, N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost); // copy devY1 to host for debugging

	for (int i = 0; i < N; i++) {
		x[i] = cuCreal(y_host[i]);
		y[i] = cuCimag(y_host[i]);
		//Phi[i] = cuCreal(y_host[i + N]); // assuming the second half of the array contains Phi values
		dx[i] = sixthTimeStep.x * cuCreal(k1_host[i]);

		dy[i] = sixthTimeStep.x * cuCimag(k1_host[i]);
		//dPhi[i] = sixthTimeStep.x * cuCreal(k1_host[i + N]); // assuming the second half of k1 contains Phi values
	}

	plt::plot(x, y, { {"label", "Final"} }); // plot the positions
	//plt::plot(x, Phi, { {"label", "Final phi"} }); // plot the Phi values

	plt::legend();

	plt::figure();
	plt::plot(dx, { {"label", "dx"} }); // plot the x component of k1
	plt::plot(dy, { {"label", "dy"} }); // plot the y component of k1
	//plt::plot(dPhi, { {"label", "dPhi"} }); // plot the Phi component of k1
	plt::legend();
#endif
	//cudaDeviceSynchronize(); // synchronize the device to ensure all operations are completed
	initialize(devY0); // reinitialize the stepper with the initial state
}

template <typename T, int N>
void AutonomousRungeKuttaStepperBase<T, N>::initialize(T* devY0)
{
	this->devY0 = devY0; // store the initial state

	cudaMemcpy(devY1, devY0, N * sizeof(T), cudaMemcpyDeviceToDevice); // copy initial state to devY1
	cudaMemcpy(devY2, devY0, N * sizeof(T), cudaMemcpyDeviceToDevice); // copy initial state to devY2
	cudaMemcpy(devY3, devY0, N * sizeof(T), cudaMemcpyDeviceToDevice); // copy initial state to devY3
	//cudaMemcpy(devZPhi4, devY0, 2 * N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice); // copy initial state to devZPhi4
}

template<typename T, int N>
void AutonomousRungeKuttaStepperBase<T, N>::copyk(const int i)
{
	if (i == 1) {
		cudaMemcpy(k1, autonomousProblem.devTimeEvolutionRhs, N * sizeof(T), cudaMemcpyDeviceToDevice);
	}
	else if (i == 2) {
		cudaMemcpy(k2, autonomousProblem.devTimeEvolutionRhs, N * sizeof(T), cudaMemcpyDeviceToDevice);
	}
	else if (i == 3) {
		cudaMemcpy(k3, autonomousProblem.devTimeEvolutionRhs, N * sizeof(T), cudaMemcpyDeviceToDevice);
	}
	else if (i == 4) {
		cudaMemcpy(k4, autonomousProblem.devTimeEvolutionRhs, N * sizeof(T), cudaMemcpyDeviceToDevice);
	}
}


template <typename T, int N>
class AutonomousRungeKuttaStepper : public AutonomousRungeKuttaStepperBase<T, N>
{
public:
	using AutonomousRungeKuttaStepper<T, N>::AutonomousRungeKuttaStepper;
};

template <int N>
class AutonomousRungeKuttaStepper<std_complex, N> : public AutonomousRungeKuttaStepperBase<std_complex, N>
{
public:
	using AutonomousRungeKuttaStepperBase<std_complex, N>::AutonomousRungeKuttaStepperBase;
protected:
	virtual void step(const int i) override
	{
		cublasStatus_t result;
		if (i == 1)
		{
			result = cublasZaxpy(this->handle, N, reinterpret_cast<const cuDoubleComplex*>(&this->halfTimeStep), reinterpret_cast<cuDoubleComplex*>(this->k1), 1, reinterpret_cast<cuDoubleComplex*>(this->devY1), 1);
		}
		else if (i == 2)
		{
			result = cublasZaxpy(this->handle, N, reinterpret_cast<cuDoubleComplex*>(&this->halfTimeStep), reinterpret_cast<cuDoubleComplex*>(this->k2), 1, reinterpret_cast<cuDoubleComplex*>(this->devY2), 1);
		}
		else if (i == 3)
		{
			result = cublasZaxpy(this->handle, N, reinterpret_cast<cuDoubleComplex*>(&this->timeStep), reinterpret_cast<cuDoubleComplex*>(this->k3), 1, reinterpret_cast<cuDoubleComplex*>(this->devY3), 1);
		}
		else if (i == 4)
		{
			result = cublasZaxpy(this->handle, N, reinterpret_cast<cuDoubleComplex*>(&this->sixthTimeStep), reinterpret_cast<cuDoubleComplex*>(this->k1), 1, reinterpret_cast<cuDoubleComplex*>(this->devY0), 1); // k1 must contain the sum of k1, k2, k3, and k4
		}
		else
		{
			fprintf(stderr, "Invalid step index: %d\n", i);
			return;
		}


		if (result != CUBLAS_STATUS_SUCCESS) {
			fprintf(stderr, "cublasZaxpy failed with error code %d\n", result);
			return;
		}
	}
};


template <int N>
class AutonomousRungeKuttaStepper<cuDoubleComplex, N> : public AutonomousRungeKuttaStepperBase<cuDoubleComplex, N>
{
public:
	using AutonomousRungeKuttaStepperBase<cuDoubleComplex, N>::AutonomousRungeKuttaStepperBase;
protected:
	virtual void step(const int i) override
	{
		cublasStatus_t result;
		if (i == 1)
		{
			result = cublasZaxpy(this->handle, N, &this->halfTimeStep, this->k1, 1, this->devY2, 1);
		}
		else if (i == 2) 
		{
			result = cublasZaxpy(this->handle, N, &this->halfTimeStep, this->k2, 1, this->devY3, 1);
		}
		else if (i == 3) 
		{
			result = cublasZaxpy(this->handle, N, &this->timeStep, this->k3, 1, this->devY3, 1);
		}
		else if (i == 4)
		{
			result = cublasZaxpy(this->handle, N, &this->sixthTimeStep, this->k1, 1, this->devY0, 1); // k1 must contain the sum of k1, k2, k3, and k4
		}
		else 
		{
			fprintf(stderr, "Invalid step index: %d\n", i);
			return;
		}


		if (result != CUBLAS_STATUS_SUCCESS) {
			fprintf(stderr, "cublasZaxpy failed with error code %d\n", result);
			return;
		}
	}
};

template <int N>
class AutonomousRungeKuttaStepper<double, N> : public AutonomousRungeKuttaStepperBase<double, N>
{
public:
	using AutonomousRungeKuttaStepperBase<double, N>::AutonomousRungeKuttaStepperBase;
protected:
	virtual void step(const int i) override
	{
		cublasStatus_t result;
		if (i == 1)
		{
			result = cublasDaxpy(this->handle, N, &this->halfTimeStep, this->k1, 1, this->devY1, 1);
		}
		else if (i == 2)
		{
			result = cublasDaxpy(this->handle, N, &this->halfTimeStep, this->k2, 1, this->devY2, 1);
		}
		else if (i == 3)
		{
			result = cublasDaxpy(this->handle, N, &this->timeStep, this->k3, 1, this->devY3, 1);
		}
		else if (i == 4)
		{
			result = cublasDaxpy(this->handle, N, &this->sixthTimeStep, this->k1, 1, this->devY0, 1); // k1 must contain the sum of k1, k2, k3, and k4
		}
		else
		{
			fprintf(stderr, "Invalid step index: %d\n", i);
			return;
		}


		if (result != CUBLAS_STATUS_SUCCESS) {
			fprintf(stderr, "cublasZaxpy failed with error code %d\n", result);
			return;
		}
	}
};



#endif