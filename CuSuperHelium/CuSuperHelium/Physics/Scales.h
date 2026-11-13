#pragma once
#ifndef SCALES_H
#define SCALES_H

/// <summary>
/// Characteristic SI scales that make the problem dimensionless, computed by Physics/NonDimensionalization.h.
/// A dimensionless quantity times its scale gives the SI value. The defaults of 1 mean "already dimensionless".
/// </summary>
struct Scales
{
	double length = 1.0;       // L / (2 pi) [m]
	double time = 1.0;         // sqrt(length / acceleration) [s]
	double acceleration = 1.0; // 3 alpha / h0^4 [m/s^2]
	double energy = 1.0;       // 3 rho_film alpha length^4 / h0^4 [J/m]
};

#endif // SCALES_H
