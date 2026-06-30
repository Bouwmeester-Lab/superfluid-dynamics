#pragma once
#ifndef RADIAL_MODELS_H
#define RADIAL_MODELS_H


struct RadialProperties {
	double R = 1.0; // Radius of the domain
	double depth = 1.0; // Depth of the domain
};

struct RadialPointers {
	const double* dev_r; // Device pointer to store the radial coordinates of the collocation points.
	const double* dev_z; // Device pointer to store the z coordinates of the collocation points -> height

	// derivatives
	double* dev_z_prime; // Device pointer to store the derivative with respect to r (used as parameter in the curve) of the z coordinates of the collocation points -> height' (deta/drho)
	double* dev_z_pp; 

	double* devPhi; // Device pointer to the potential
	double* devPhiPrime; // Device pointer to the derivative of the potential with respect to r (used as parameter in the curve) -> potential' (dPhi/drho)
};

#endif // !RADIAL_MODELS_H