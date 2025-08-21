#pragma once
/// This file contains the required code to implement the Runge-Kutta 45 method which is a
/// variable step size method for solving ordinary differential equations.
/// The method computes the 4th order RK and 5th order RK approximations and uses the difference
/// to determine the step size for the next iteration.

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "AutonomousProblem.h"
#include "cublas_v2.h"

