#pragma once
#ifndef PROBLEM_PROPERTIES_H
#define PROBLEM_PROPERTIES_H

struct ProblemProperties
{
public:
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
};

#endif // !PROBLEM_PROPERTIES_H
