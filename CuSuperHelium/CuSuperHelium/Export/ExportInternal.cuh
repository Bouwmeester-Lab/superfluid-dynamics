#pragma once
#ifndef EXPORT_INTERNAL_H
#define EXPORT_INTERNAL_H

#include "Export/ExportApi.cuh"

#include "Core/CudaChecks.cuh"
#include "Core/DeviceBuffer.cuh"
#include "Core/constants.cuh"
#include "Export/ExportTypes.cuh"
#include "IO/TrajectoryLogger.cuh"
#include "Ode/OdeSolver.h"
#include "Optomechanics/OptomechanicalVariables.h"
#include "Optomechanics/OptomechanicalNonDimensionalization.h"
#include "Physics/DimensionlessParameters.h"
#include "Physics/NonDimensionalization.h"
#include "Physics/PhysicalParameters.h"

#include <cstdlib>
#include <iostream>

/// SI inputs from the C ABI struct, including the caller's material.
PhysicalParameters toPhysicalParameters(const SimProperties& simProperties);
/// Non-dimensionalized solver parameters (with their scales) plus the solver options from the ABI struct.
DimensionlessParameters toDimensionlessParameters(const SimProperties& simProperties);
/// Times in seconds to dimensionless times.
RK4SolverOptions toDimensionless(RK4SolverOptions options, const Scales& scales);
void copyProperties(COptomechanicalVariables& c_optomechanicalVariables, OptomechanicalVariables& opto_variables);
void printOptomechanicalVariables(OptomechanicalVariables& variables);

/// <summary>
/// Copies an integration's output to malloc'ed host arrays the caller frees: the recorded trajectory when
/// a TrajectoryLogger was attached, otherwise only the final state (one state, no times).
/// </summary>
template <typename T>
int copyIntegrationResultToHost(const OdeSolver<T>& solver, size_t stateSize, TrajectoryLogger<T>* trajectory,
	T** statesOut, size_t* statesCount, double** timesOut, size_t* timesCount)
{
	CUSH_CHECK(cudaDeviceSynchronize());
	if (trajectory != nullptr)
	{
		if (trajectory->copyTimesToHost(timesOut, timesCount) != 0) return -1;
		return trajectory->copyStatesToHost(statesOut, statesCount);
	}
	*timesOut = nullptr;
	*timesCount = 0;
	T* s = static_cast<T*>(std::malloc(stateSize * sizeof(T)));
	if (s == nullptr) return -1;
	CUSH_CHECK(cudaMemcpy(s, solver.state(), stateSize * sizeof(T), cudaMemcpyDeviceToHost));
	*statesOut = s;
	*statesCount = 1;
	return 0;
}

inline int unsupportedN(size_t N)
{
	std::cerr << "Error: Unsupported N size " << N << std::endl;
	return -1;
}

inline int unsupportedBatchSize(int batchSize)
{
	std::cerr << "Error: Unsupported batch size " << batchSize << std::endl;
	return -1;
}

inline int freeExportMemory(double* statesOut, double* timesOut)
{
	std::free(statesOut);
	std::free(timesOut);
	return 0;
}

#endif
