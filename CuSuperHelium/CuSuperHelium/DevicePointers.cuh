#pragma once
#include "constants.cuh"

struct DevicePointers
{
	const std_complex* Z;
	const std_complex* Zp;
	const std_complex* Zpp;
	const std_complex* Phi;
	const std_complex* LowerVelocities;
};