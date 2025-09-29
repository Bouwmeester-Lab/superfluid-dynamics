#include "Export.cuh"
//#include "SimulationFunctions.cuh"

int dispertionTest256(double wavelength, double simulationTime, double rho, double kappa, double depth, int steps)
{
	ProblemProperties properties;
	properties.rho = rho;
	properties.kappa = kappa;
	properties.depth = depth;

	return dispersionTest<256>(wavelength, simulationTime, properties, steps);
}