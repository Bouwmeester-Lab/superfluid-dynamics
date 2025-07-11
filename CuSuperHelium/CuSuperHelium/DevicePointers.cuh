#pragma once
#include "constants.cuh"

struct DevicePointers
{
	std_complex* Z;
	std_complex* Zp;
	std_complex* Zpp;
	std_complex* Phi;
	std_complex* LowerVelocities;
};