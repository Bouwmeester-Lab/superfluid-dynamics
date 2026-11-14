#pragma once
#ifndef COMPLEX_AS_REAL_PROBLEM_H
#define COMPLEX_AS_REAL_PROBLEM_H

#include "Core/constants.cuh"
#include "Ode/OdeProblem.h"

/// <summary>
/// Views an OdeProblem over n complex values as one over 2n doubles, so problems written for std_complex states
/// run with real-valued solvers (Gauss-Legendre). std_complex is laid out as {re, im}, so the state is reinterpreted
/// in place: the real problem's y is (re y0, im y0, re y1, im y1, ...). No copy is made.
/// </summary>
class ComplexAsRealProblem final : public OdeProblem<double>
{
public:
	explicit ComplexAsRealProblem(OdeProblem<std_complex>& complexProblem) : complexProblem(complexProblem) {}

	size_t size() const override { return 2 * complexProblem.size(); }

	void rhs(const double* y, double t, double* dydt) override
	{
		static_assert(sizeof(std_complex) == 2 * sizeof(double), "std_complex must be two packed doubles");
		complexProblem.rhs(reinterpret_cast<const std_complex*>(y), t, reinterpret_cast<std_complex*>(dydt));
	}

	void start(double t0) override { complexProblem.start(t0); }
	void setStream(cudaStream_t stream) override { complexProblem.setStream(stream); }

private:
	OdeProblem<std_complex>& complexProblem;
};

#endif // !COMPLEX_AS_REAL_PROBLEM_H
