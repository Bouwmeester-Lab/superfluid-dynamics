#pragma once
#ifndef TIMESTEP_MANAGER_H
#define TIMESTEP_MANAGER_H


#include "ProblemProperties.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cufft.h"
#include "utilities.cuh"
#include "constants.cuh"
#include "Derivatives.cuh"
#include "createM.cuh"
#include "WaterVelocities.cuh"
#include "MatrixSolver.cuh"
#include "AutonomousProblem.h"
#include "Energies.cuh"
#include "DelayedIntensityTerm.cuh"
#include "BoundaryProblem.cuh"

#include "matplotlibcpp.h"

#include "BaseBoundaryIntegrator.cuh"
#include "TimedBoundaryIntegrator.cuh"

namespace plt = matplotlibcpp;







#endif // TIMESTEP_MANAGER_H
