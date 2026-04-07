#pragma once
#ifndef PROBLEM_PROPERTIES_H
#define PROBLEM_PROPERTIES_H

struct ProblemProperties
{
public:
	double rho;
	double U;
	double kappa;
	double depth;
	double initial_amplitude;

	// for plotting
	double y_min;
	double y_max;

	// for using expansions
	bool use_expansions;
	int expansion_order;

	// for ignoring depth
	bool infinite_depth = false;
};

#endif // !PROBLEM_PROPERTIES_H
