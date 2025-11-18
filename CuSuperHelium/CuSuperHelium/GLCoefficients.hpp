#pragma once

constexpr double sqrt3 = 1.7320508075688772935;

struct GL_Coefficients
{
	const double A11 = 0.25;
	const double A12 = 0.25 - sqrt3 / 6.0;
	const double A21 = 0.25 + sqrt3 / 6.0;
	const double A22 = 0.25;

	const double b1 = 0.5;
	const double b2 = 0.5;

	const double c1 = 0.5 - sqrt3 / 6.0;
	const double c2 = 0.5 + sqrt3 / 6.0;
};