#pragma once
#ifndef NON_DIMENSIONALIZATION_H
#define NON_DIMENSIONALIZATION_H

// The one place where SI inputs become the dimensionless values the solvers use.
//
//   length       = L / (2 pi)
//   acceleration = 3 alpha / h0^4                    (van der Waals attraction of the film)
//   time         = sqrt(length / acceleration)
//   energy       = 3 rho_film alpha length^4 / h0^4
//
// The material (alpha = Hamaker constant, rho_film = film density) always comes from the caller.

#include "BoundaryIntegral/Axisymmetric/RadialModels.h"
#include "Core/constants.cuh"
#include "Physics/DimensionlessParameters.h"
#include "Physics/PhysicalParameters.h"
#include "Physics/Scales.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace NonDimensionalization
{
	namespace detail
	{
		inline void requirePositive(double value, const char* name)
		{
			if (!(value > 0.0))
			{
				throw std::invalid_argument(std::string("PhysicalParameters.") + name + " must be positive, got " + std::to_string(value));
			}
		}
	}

	/// <summary>
	/// Characteristic scales of the problem. Throws std::invalid_argument unless L, depth, the Hamaker constant
	/// and the film density are all positive.
	/// </summary>
	inline Scales computeScales(const PhysicalParameters& p)
	{
		detail::requirePositive(p.L, "L");
		detail::requirePositive(p.depth, "depth");
		detail::requirePositive(p.hamakerConstant, "hamakerConstant");
		detail::requirePositive(p.filmDensity, "filmDensity");

		Scales s;
		s.length = p.L / (2.0 * PI_d);
		s.acceleration = 3 * p.hamakerConstant / std::pow(p.depth, 4);
		s.time = std::sqrt(s.length / s.acceleration);
		s.energy = 3.0 * p.filmDensity * p.hamakerConstant * std::pow(s.length, 4) / std::pow(p.depth, 4);
		return s;
	}

	/// <summary>
	/// Dimensionless solver parameters (density ratio, surface tension, depth) and their scales. Solver options
	/// (expansions, infinite depth) keep their defaults; the caller sets them on the result.
	/// </summary>
	inline DimensionlessParameters nonDimensionalize(const PhysicalParameters& p)
	{
		DimensionlessParameters d;
		d.scales = computeScales(p);
		const Scales& s = d.scales;

		const double surfaceTensionScale = p.filmDensity * s.length * s.length * s.length / (s.time * s.time);
		d.kappa = p.kappa / surfaceTensionScale;
		d.depth = p.depth / s.length;
		d.rho = p.rho / p.filmDensity;
		return d;
	}

	inline double toDimensionlessTime(double seconds, const Scales& s) { return seconds / s.time; }
	inline double toDimensionlessLength(double meters, const Scales& s) { return meters / s.length; }
	inline double toDimensionlessFrequency(double perSecond, const Scales& s) { return perSecond * s.time; }

	/// <summary>
	/// Geometry of the axisymmetric problem, with lengths in units of the domain size L (so R = 1). Needs no
	/// material properties.
	/// </summary>
	inline RadialProperties radialGeometry(double L, double depth)
	{
		detail::requirePositive(L, "L");
		return RadialProperties{ .R = 1.0, .depth = depth / L };
	}
}

#endif // NON_DIMENSIONALIZATION_H
