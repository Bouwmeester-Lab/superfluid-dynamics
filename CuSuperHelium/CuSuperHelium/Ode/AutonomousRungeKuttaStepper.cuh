#pragma once
#ifndef RUNGE_KUNTA_H
#define RUNGE_KUNTA_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Ode/AutonomousProblem.h"
#include "Core/cuDoubleComplexOperators.cuh"
#include "cublas_v2.h"
#include "IO/DataLogger.cuh"
#include "IO/ValueLogger.h"
#include <stdexcept>
#include <initializer_list>
#include "Ode/OdeSolver.h"
#include "Ode/RK4Options.h"
#include "IO/TrajectoryLogger.cuh"
#include "indicators/progress_bar.hpp"
#include <memory>


template <typename T, int N>
class AutonomousRungeKuttaStepperBase : public OdeSolver
{
public:
	AutonomousRungeKuttaStepperBase(AutonomousProblem<T, N>& autonomousProblem, double tstep = 1e-2, std::shared_ptr<TrajectoryLogger<T, N>> logger = nullptr);
	virtual ~AutonomousRungeKuttaStepperBase();
	
	void setTimeStep(double tstep) 
	{ 
		timeStep = CastTo<T>(tstep);
		halfTimeStep = CastTo<T>(tstep * 0.5);
		sixthTimeStep = CastTo<T>(tstep / 6.0);
	}

	void initialize(T* devY0, bool onDevice = false);
	void runStep(int _step);
	void setOptions(const RK4Options& options) 
	{ 
		setTimeStep(options.initial_timestep); 
	}
	virtual OdeSolverResult runEvolution(double startTime, double endTime) override;
protected:
	indicators::ProgressBar bar{
		indicators::option::BarWidth{50},
		indicators::option::Start{"["},
		indicators::option::Fill{"="},
		indicators::option::Lead{">"},
		indicators::option::Remainder{" "},
		indicators::option::End{"]"},
		indicators::option::PostfixText{"Running integration for " + std::to_string(N) + " variables"},
		indicators::option::ForegroundColor{indicators::Color::green},
		indicators::option::ShowPercentage{true},
		indicators::option::FontStyles{std::vector<indicators::FontStyle>{indicators::FontStyle::bold}},
		indicators::option::ShowElapsedTime{true},
		indicators::option::ShowRemainingTime{true},
	};

	const int threads = 256;
	const int blocks = (N + threads - 1) / threads;
	bool allocatedY0 = false; ///< Flag to indicate if devY0 was allocated by the stepper
	//DataLogger<T, N>& logger; ///< Logger for data collection
	std::shared_ptr<TrajectoryLogger<T, N>> logger; ///< Logger for trajectory data collection


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

	double currentTime = 0.0;

	virtual void step(const int i) = 0;
	
};

template <typename T, int N>
AutonomousRungeKuttaStepperBase<T, N>::AutonomousRungeKuttaStepperBase(AutonomousProblem<T, N>& autonomousProblem, double tstep, std::shared_ptr<TrajectoryLogger<T, N>> logger) : autonomousProblem(autonomousProblem), logger(logger)
{
	CUSH_CHECK(cublasCreate(&handle));
	CUSH_CHECK(cudaMalloc(&k1, N * sizeof(T))); // allocate memory for k1
	CUSH_CHECK(cudaMalloc(&k2, N * sizeof(T))); // allocate memory for k2
	CUSH_CHECK(cudaMalloc(&k3, N * sizeof(T))); // allocate memory for k3
	CUSH_CHECK(cudaMalloc(&k4, N * sizeof(T))); // allocate memory for k4

	CUSH_CHECK(cudaMalloc(&devY1, N * sizeof(T))); // allocate memory for devZphi1
	CUSH_CHECK(cudaMalloc(&devY2, N * sizeof(T))); // allocate memory for devZphi2
	CUSH_CHECK(cudaMalloc(&devY3, N * sizeof(T))); // allocate memory for devZphi3

	setTimeStep(tstep);
}

template <typename T,int N>
AutonomousRungeKuttaStepperBase<T, N>::~AutonomousRungeKuttaStepperBase()
{
	if(allocatedY0)
		CUSH_CHECK_NOTHROW(cudaFree(devY0));
	CUSH_CHECK_NOTHROW(cudaFree(k1));
	CUSH_CHECK_NOTHROW(cudaFree(k2));
	CUSH_CHECK_NOTHROW(cudaFree(k3));
	CUSH_CHECK_NOTHROW(cudaFree(k4));

	CUSH_CHECK_NOTHROW(cudaFree(devY1));
	CUSH_CHECK_NOTHROW(cudaFree(devY2));
	CUSH_CHECK_NOTHROW(cudaFree(devY3));
	CUSH_CHECK_NOTHROW(cublasDestroy(handle));
}

template <typename T, int N>
void AutonomousRungeKuttaStepperBase<T, N>::runStep(int _step)
{
	autonomousProblem.run(devY0, k1); // run the autonomous problem with the initial state devY0 and store the result in k1

	// copyk(1); // copy k1 from the autonomousProblem
	// x_n+1 = x_n + h*0.5 * v_n -> x_n is a N dimensional vector, h is the time step, v_n is the velocity at time n
	step(1); // step 1 of the Runge-Kutta method


	autonomousProblem.run(devY1, k2);

	//copyk(2); // copy k2 from the autonomousProblem

	step(2); // step 2 of the Runge-Kutta method


	autonomousProblem.run(devY2, k3);

	//copyk(3); // copy k3 from the autonomousProblem
	
	step(3);

	autonomousProblem.run(devY3, k4);

	//copyk(4); // copy k4 from the autonomousProblem

	
	add_k_vectors << <blocks, threads >> > (k1, k2, k3, k4, k1, N);
	CUSH_CHECK_LAUNCH(); // add the four vectors together

	//cudaDeviceSynchronize(); // synchronize the device to ensure all operations are completed

	//std::vector<cufftDoubleComplex> k1_host(2 * N);

	//cudaMemcpy(k1_host.data(), k1, 2 * N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost); // copy k1 to host for debugging
	//printf("%f h/6\n", cuCreal(sixthTimeStep)); // print the time step for debugging
	//for(int i = 0; i < 2*N; i++) {
	//	printf("k[%d] = (%f, %f)\n", i, cuCreal(k1_host[i]), cuCimag(k1_host[i]));
	//}
	//std::cin.get(); // wait for user input to continue

	//logger.waitForCopy(); // wait for the logger to finish copying data before proceeding

	step(4); // step 4 of the Runge-Kutta method, which combines the results of the previous steps, it should overwrite devY0 with the final result

	//if(logger.shouldCopy(_step))
	//	logger.setReadyToCopy(devY0, cudaStreamPerThread, currentTime + CastFrom<T>(timeStep), true); //TODO: make this work with a custom stream.
	//cudaDeviceSynchronize(); // synchronize the device to ensure all operations are completed
	initialize(devY0, true); // reinitialize the stepper with the initial state which is already on the device
}

template <typename T, int N>
void AutonomousRungeKuttaStepperBase<T, N>::initialize(T* devY0, bool onDevice)
{
	if (onDevice) 
	{
		this->devY0 = devY0; // store the initial state
		CUSH_CHECK(cudaMemcpy(devY1, devY0, N * sizeof(T), cudaMemcpyDeviceToDevice)); // copy initial state to devY1
		CUSH_CHECK(cudaMemcpy(devY2, devY0, N * sizeof(T), cudaMemcpyDeviceToDevice)); // copy initial state to devY2
		CUSH_CHECK(cudaMemcpy(devY3, devY0, N * sizeof(T), cudaMemcpyDeviceToDevice)); // copy initial state to devY3
	}
	else {
		allocatedY0 = true;
		CUSH_CHECK(cudaMalloc(&this->devY0, N * sizeof(T))); // allocate memory for devZphi1
		CUSH_CHECK(cudaMemcpy(this->devY0, devY0, N * sizeof(T), cudaMemcpyHostToDevice)); // copy initial state to devY1
		CUSH_CHECK(cudaMemcpy(devY1, devY0, N * sizeof(T), cudaMemcpyHostToDevice)); // copy initial state to devY1
		CUSH_CHECK(cudaMemcpy(devY2, devY0, N * sizeof(T), cudaMemcpyHostToDevice)); // copy initial state to devY2
		CUSH_CHECK(cudaMemcpy(devY3, devY0, N * sizeof(T), cudaMemcpyHostToDevice)); // copy initial state to devY3
	}
	
	//cudaMemcpy(devZPhi4, devY0, 2 * N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice); // copy initial state to devZPhi4
}

template <typename T, int N>
class AutonomousRungeKuttaStepper : public AutonomousRungeKuttaStepperBase<T, N>
{
public:
	using AutonomousRungeKuttaStepperBase<T, N>::AutonomousRungeKuttaStepperBase;
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
			throw std::logic_error("Invalid Runge-Kutta stage index " + std::to_string(i));
		}


		CUSH_CHECK(result);
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
			throw std::logic_error("Invalid Runge-Kutta stage index " + std::to_string(i));
		}


		CUSH_CHECK(result);
	}
};

template<typename T, int N>
OdeSolverResult AutonomousRungeKuttaStepperBase<T, N>::runEvolution(double startTime, double endTime)
{
	currentTime = startTime;
	size_t steps = static_cast<size_t>((endTime - startTime) / CastFrom<T>(timeStep));
	bar.set_progress(0); // reset progress bar
	for(size_t step = 0; step < steps; step++)
	{
		runStep(step);
		currentTime = startTime + (step + 1) * CastFrom<T>(timeStep);
		// log:
		logger->logTrajectory(currentTime, devY0);
		
		if ((step * 100) / steps > bar.current()) 
		{
			bar.tick();
		}		
	}
	if (currentTime >= endTime)
		return OdeSolverResult::ReachedEndTime;
	return OdeSolverResult::UnknownError;
}

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
			throw std::logic_error("Invalid Runge-Kutta stage index " + std::to_string(i));
		}


		CUSH_CHECK(result);
	}
};



#endif