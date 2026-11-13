#pragma once
#ifndef INTERFACE_PARAMETERS_H
#define INTERFACE_PARAMETERS_H

/// <summary>
/// The dimensionless values the periodic-2D boundary-integral solver reads. The solver defines what it needs here,
/// and Physics/DimensionlessParameters extends it, so this layer never depends on Physics. Passed by value into kernels.
/// </summary>
struct InterfaceParameters
{
	double rho = 1.0;   // density ratio rho_upper / rho_film
	double U = 0.0;     // background flow
	double kappa = 0.0; // surface tension, dimensionless
	double depth = 1.0; // film depth, dimensionless

	// solver options
	bool use_expansions = false;
	int expansion_order = 1;
	bool infinite_depth = false;
};

#endif // INTERFACE_PARAMETERS_H
