#pragma once
#ifndef EXPORT_INTERNAL_H
#define EXPORT_INTERNAL_H

#include "ExportApi.cuh"

#include "../Common/CudaChecks.cuh"
#include "../constants.cuh"
#include "../ExportTypes.cuh"
#include "../OptomechanicalVariables.h"
#include "../ProblemProperties.hpp"

#include <cstdlib>
#include <iostream>

void copyProperties(SimProperties& simProperties, ProblemProperties& properties);
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
