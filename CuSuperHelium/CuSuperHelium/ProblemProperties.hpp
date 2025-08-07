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

	// for solver tolerances
	size_t maxIterations = 500;
	double tolerance = 1e-13;
};

#endif // !PROBLEM_PROPERTIES_H
