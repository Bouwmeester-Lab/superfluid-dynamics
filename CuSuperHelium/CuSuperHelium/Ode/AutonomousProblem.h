#pragma once
#ifndef AUTONOMOUS_PROBLEM_H
#define AUTONOMOUS_PROBLEM_H

#include <cstddef>
#include "Ode/OdeProblem.h"

/// <summary>
/// An OdeProblem whose right-hand side does not depend on time. The state size is set at construction.
/// Subclasses implement run(y, dydt); rhs() forwards to it and ignores t.
/// </summary>
template <typename T>
class AutonomousProblem : public OdeProblem<T>
{
public:
	explicit AutonomousProblem(size_t stateSize) : stateSize(stateSize) {}
	virtual ~AutonomousProblem() = default;

	size_t size() const override { return stateSize; }

	void rhs(const T* y, double, T* dydt) override
	{
		run(y, dydt);
	}

	virtual void run(const T* initialState, T* rhs) = 0;

private:
	size_t stateSize;
};

#endif // !AUTONOMOUS_PROBLEM_H
