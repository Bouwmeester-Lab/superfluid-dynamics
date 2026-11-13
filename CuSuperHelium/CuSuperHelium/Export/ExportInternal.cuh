#pragma once
#ifndef EXPORT_INTERNAL_H
#define EXPORT_INTERNAL_H

#include "Export/ExportApi.cuh"

#include "Core/CudaChecks.cuh"
#include "Core/DeviceBuffer.cuh"
#include "Core/constants.cuh"
#include "Export/ExportTypes.cuh"
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
