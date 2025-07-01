#pragma once
#include "AutonomousProblem.h"
#include "ProblemProperties.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

template <int size>
class DoubleWaterBoundaryIntegralCalculator : public AutonomousProblem<double, 3*size>
{
public:
	DoubleWaterBoundaryIntegralCalculator(ProblemProperties& problemProperties);
	~DoubleWaterBoundaryIntegralCalculator();



private:

};

template <int size>
DoubleWaterBoundaryIntegralCalculator<size>::DoubleWaterBoundaryIntegralCalculator(ProblemProperties& problemProperties)
{
}

template <int size>
DoubleWaterBoundaryIntegralCalculator<size>::~DoubleWaterBoundaryIntegralCalculator()
{
}