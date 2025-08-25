#pragma once
#ifndef RUNGE_KUNTA_H
#define RUNGE_KUNTA_H

#include "RKBase.cuh"

template <typename T, int N>
class AutonomousRungeKuttaStepper : public AutonomousRungeKuttaStepperBase<T, N, 4>
{
public:
	using AutonomousRungeKuttaStepperBase<T, N, 4>::AutonomousRungeKuttaStepperBase;
};

template <int N>
class AutonomousRungeKuttaStepper<std_complex, N> : public AutonomousRungeKuttaStepperBase<std_complex, N, 4>
{
public:
	using AutonomousRungeKuttaStepperBase<std_complex, N, 4>::AutonomousRungeKuttaStepperBase;
protected:
	virtual void step(const int i) override
	{
		cublasStatus_t result;
		if (i < 3)
		{
			result = cublasZaxpy(this->handle, N, reinterpret_cast<const cuDoubleComplex*>(&this->halfTimeStep), reinterpret_cast<cuDoubleComplex*>(this->getk(i)), 1, reinterpret_cast<cuDoubleComplex*>(this->getDevY(i)), 1);
		}
		else if (i == 3)
		{
			result = cublasZaxpy(this->handle, N, reinterpret_cast<cuDoubleComplex*>(&this->sixthTimeStep), reinterpret_cast<cuDoubleComplex*>(this->getk(0), 1, reinterpret_cast<cuDoubleComplex*>(this->getDevY(0)), 1); // k1 must contain the sum of k1, k2, k3, and k4
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
class AutonomousRungeKuttaStepper<cuDoubleComplex, N> : public AutonomousRungeKuttaStepperBase<cuDoubleComplex, N, 4>
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