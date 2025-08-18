#pragma once
class Solver
{
public:
	virtual void solve(double* A, double* b, double* x) = 0;
};