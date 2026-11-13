#pragma once
#if !defined(DIMENSIONLESS_PARAMETERS_H)
#define DIMENSIONLESS_PARAMETERS_H

#include "BoundaryIntegral/Periodic2D/InterfaceParameters.h"
#include "Physics/Scales.h"

/// <summary>
/// Dimensionless problem parameters: what the periodic-2D solver reads (InterfaceParameters) plus the SI scales
/// that produced them and a few App-only fields. Built from PhysicalParameters by
/// NonDimensionalization::nonDimensionalize. Passed by value into kernels.
/// </summary>
struct DimensionlessParameters : InterfaceParameters
{
public:
	double initial_amplitude = 1.0;

	// for plotting
	double y_min;
	double y_max;

	// SI scales of this problem (all 1 when the problem is defined directly in dimensionless units)
	Scales scales;
};

#endif // !DIMENSIONLESS_PARAMETERS_H
