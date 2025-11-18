#pragma once
enum class OdeSolverResult {
	ReachedEndTime,
	StiffnessDetected
};

class OdeSolver
{
public:
	OdeSolver();
	~OdeSolver();
	virtual OdeSolverResult runEvolution(double startTime, double endTime) = 0;
private:

};

OdeSolver::OdeSolver()
{
}

OdeSolver::~OdeSolver()
{
}