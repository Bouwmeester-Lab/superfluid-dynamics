#pragma once
#ifndef RUNGE_KUNTA_H
#define RUNGE_KUNTA_H

#include "ProblemProperties.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "TimeStepManager.cuh"

template <int N>
class RungeKuntaStepper
{
public:
	RungeKuntaStepper();
	~RungeKuntaStepper();

private:
	TimeStepManager<N> timeStepManager; ///< Instance of the TimeStepManager to handle time-stepping operations
	void secondStep();
};
template <int N>
RungeKuntaStepper<N>::RungeKuntaStepper()
{
}
template <int N>
RungeKuntaStepper<N>::~RungeKuntaStepper()
{
}

template<int N>
void RungeKuntaStepper<N>::secondStep()
{
	// we try to do the second step of the Runge-Kutta method
	// we need to do x1 = x0 + h/2 * velocitiesLower
}

#endif