#pragma once
#ifndef PROBLEM_PROPERTIES_H
#define PROBLEM_PROPERTIES_H

struct ProblemProperties
{
public:
	double L = 1.0;

	double rho = 1.0;
	double U = 0.0;
	double kappa = 0.0;
	double depth = 1.0;
	double initial_amplitude = 1.0;

	// for plotting
	double y_min;
	double y_max;

	// for using expansions
	bool use_expansions = false;
	int expansion_order = 1;

	// for ignoring depth
	bool infinite_depth = false;

	// base units
	double base_length = 1.0;
	double base_time = 1.0;
	double base_energy = 1.0;
	double base_acceleration = 1.0;
};

#endif // !PROBLEM_PROPERTIES_H
