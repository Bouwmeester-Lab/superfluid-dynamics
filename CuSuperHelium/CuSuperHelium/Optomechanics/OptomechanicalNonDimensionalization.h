#pragma once
#ifndef OPTOMECHANICAL_NON_DIMENSIONALIZATION_H
#define OPTOMECHANICAL_NON_DIMENSIONALIZATION_H

// Optomechanical counterpart of Physics/NonDimensionalization.h. It lives in Optomechanics/ because Physics
// may not depend on this layer; it only applies the Scales that Physics computed.

#include "Optomechanics/OptomechanicalVariables.h"
#include "Physics/Scales.h"

namespace NonDimensionalization
{
	inline OptomechanicalVariables nonDimensionalize(OptomechanicalVariables v, const Scales& s)
	{
		v.gamma *= s.time;              // frequency
		v.detuning *= s.time;           // frequency
		v.G *= s.time * s.length;       // coupling strength (Hz / m)
		v.Tau /= s.time;                // time delay
		v.location_x0_mode /= s.length; // length
		v.sigma_optical_mode /= s.length;
		v.sigma_thermal_mode /= s.length;
		v.ramp_rate *= s.time;          // intensity per second
		v.omega_drive *= s.time;        // frequency
		// TODO: max_intensity and Beta are passed through unchanged, as before.
		return v;
	}
}

#endif // OPTOMECHANICAL_NON_DIMENSIONALIZATION_H
