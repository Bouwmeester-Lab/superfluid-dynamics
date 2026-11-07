#pragma once
#ifndef RADIAL_MODELS_H
#define RADIAL_MODELS_H

#include <cstddef>

struct RadialProperties {
	double R = 1.0; // Radius of the domain
	double depth = 1.0; // Depth of the domain
};

/// <summary>
/// Device pointers describing the free surface z = eta(r) at the N surface collocation points, as the time stepper sees it.
/// The boundary-integral kernels do not read this directly: it is turned into BoundaryNodes (Radial/BoundaryNodes.cuh).
/// </summary>
struct RadialSurfaceState {
	const double* r;              // radial coordinates of the collocation points
	const double* eta;            // surface height eta(r)
	const double* etaPrime;       // d eta / dr
	const double* etaPrimePrime;  // d^2 eta / dr^2
	const double* phi;            // surface potential
	const double* phiPrime;       // d phi / dr along the surface
	size_t n;                     // number of surface collocation points
};

#endif // !RADIAL_MODELS_H
