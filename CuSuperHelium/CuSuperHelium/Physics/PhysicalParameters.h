#pragma once
#ifndef PHYSICAL_PARAMETERS_H
#define PHYSICAL_PARAMETERS_H

/// <summary>
/// Problem inputs in SI units, as the caller states them. The material properties (Hamaker constant and film
/// density) have no defaults: they choose the material being simulated, so the caller must set them.
/// Turned into DimensionlessParameters by Physics/NonDimensionalization.h.
/// </summary>
struct PhysicalParameters
{
	double L;               // period length of the interface [m]
	double depth;           // film thickness h0 [m]
	double rho;             // density of the upper fluid [kg/m^3]; the solver uses rho / filmDensity
	double kappa;           // surface tension [N/m]
	double hamakerConstant; // van der Waals coefficient alpha [m^5/s^2]: the film feels an acceleration 3 alpha / h^4
	double filmDensity;     // density of the film fluid [kg/m^3]
};

#endif // PHYSICAL_PARAMETERS_H
