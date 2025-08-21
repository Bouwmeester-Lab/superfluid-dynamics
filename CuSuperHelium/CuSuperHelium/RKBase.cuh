#pragma once
/// Defines the base class for Runge-Kutta methods.
#include "ProblemProperties.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "AutonomousProblem.h"
#include "matplotlibcpp.h"
#include "cuDoubleComplexOperators.cuh"
#include "cublas_v2.h"
#include "DataLogger.cuh"
namespace plt = matplotlibcpp;

template <typename T, uint16_t order>
struct kcoefficients 
{
	T coeffs[order]; ///< Coefficients for the Runge-Kutta method
};

template <typename T, uint16_t order>
__global__ void add_k_vectors(T* k, T* result, kcoefficients<double, order> coeffs, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		#pragma unroll
		for(int j = 0; j < order; ++j) {
			result[i] += coeffs.coeffs[j] * k[i + j * n]; // accumulate the contributions from each k vector
		}
	}
}


template <uint16_t order>
__global__ void add_k_vectors(cuDoubleComplex* k, cuDoubleComplex* result, kcoefficients<double, order> coeffs, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
#pragma unroll
		for (int j = 0; j < order; ++j) {
			result[i].x += coeffs.coeffs[j] * k[i + j * n].x; // accumulate the contributions from each k vector
			result[i].y += coeffs.coeffs[j] * k[i + j * n].y; // accumulate the contributions from each k vector
		}
	}
}


template <typename T, int N, uint16_t order>
class AutonomousRungeKuttaStepperBase
{
public:
	AutonomousRungeKuttaStepperBase(AutonomousProblem<T, N>& autonomousProblem, DataLogger<T, N>& logger, double tstep = 1e-2);
	~AutonomousRungeKuttaStepperBase();

	void setTimeStep(double tstep)
	{
		timeStep = CastTo<T>(tstep);
		/*halfTimeStep = CastTo<T>(tstep * 0.5);
		sixthTimeStep = CastTo<T>(tstep / 6.0);*/
	}

	void initialize(T* devY0);
	void runStep(int _step);
protected:
	const int threads = 256;
	const int blocks = (N + threads - 1) / threads;

	DataLogger<T, N>& logger; ///< Logger for data collection

	T* k;

	T* devY;
	kcoefficients<double, order> kcoeffs; ///< Coefficients for the Runge-Kutta method

	AutonomousProblem<T, N>& autonomousProblem; ///< Instance of the TimeStepManager to handle time-stepping operations
	cublasHandle_t handle; ///< CUBLAS handle for matrix operations

	T timeStep;
	//T halfTimeStep;
	//T sixthTimeStep;

	virtual void step(const int i) = 0;

	inline T* getk(uint16_t i) const 
	{
		return *k[i * N];
	}
	inline T* getDevY(uint16_t i) const
	{
		return *devY[i * N];
	}
};

template <typename T, int N, uint16_t order>
AutonomousRungeKuttaStepperBase<T, N, order>::AutonomousRungeKuttaStepperBase(AutonomousProblem<T, N>& autonomousProblem, DataLogger<T, N>& logger, double tstep) : autonomousProblem(autonomousProblem), logger(logger)
{
	cublasCreate(&handle);
	cudaMalloc(&k,  N * sizeof(T) * order); // allocate memory for k

	cudaMalloc(&devY, N * sizeof(T) * order); // allocate memory for devY size N * order

	setTimeStep(tstep);
}

template <typename T, int N, uint16_t order>
AutonomousRungeKuttaStepperBase<T, N, order>::~AutonomousRungeKuttaStepperBase()
{
}

template <typename T, int N, uint16_t order>
void AutonomousRungeKuttaStepperBase<T, N, order>::runStep(int _step)
{
	#pragma unroll
	for(int i = 0; i < order; ++i) 
	{
		autonomousProblem.run(getDevY(i), getk(i)); // run the autonomous problem with the current state and store the result in k[i]
		if (i == order - 1) // if this is the last step add all k vectors together
		{
			add_k_vectors << <blocks, threads >> > (k, k, kcoeffs, N); // add the four vectors together
			logger.waitForCopy(); // wait for the logger to finish copying data before proceeding
		}
			
		step(i); // perform the Runge-Kutta step for the current order
	}

	if (logger.shouldCopy(_step))
		logger.setReadyToCopy(devY0);
	//cudaDeviceSynchronize(); // synchronize the device to ensure all operations are completed
	initialize(devY); // reinitialize the stepper with the initial state
}

template <typename T, int N, uint16_t order>
void AutonomousRungeKuttaStepperBase<T, N, order>::initialize(T* devY0)
{
	this->devY0 = devY0; // store the initial state
	#pragma unroll
	for(int i = 1; i < order; ++i) {
		cudaMemcpy(getDevY(i), devY0, N * sizeof(T), cudaMemcpyDeviceToDevice); // copy initial state to each devY
	}
	//cudaMemcpy(devZPhi4, devY0, 2 * N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice); // copy initial state to devZPhi4
}