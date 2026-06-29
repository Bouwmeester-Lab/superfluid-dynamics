#include "ExportInternal.cuh"

#include "../AutonomousProblem.h"
#include "../BaseBoundaryIntegrator.cuh"
#include "../GaussLegendre.cuh"
#include "../HeliumBoundaryProblem.cuh"
#include "../JacobianCalculator.cuh"
#include "../RealBoundaryIntegralCalculator.cuh"
#include "../RK4_Time_Dependent.cuh"

#include <memory>

template <size_t N>
int integrateSimulationGL2_N(double* initialState, double** statesOut, size_t* statesCount,
	double** timesOut, size_t* timesCount,
	SimProperties* simProperties, GaussLegendreOptions* glCOptions)
{
	try {
		ProblemProperties properties;
		copyProperties(*simProperties, properties);
		std::cout << "Properties copied." << std::endl;
		std::cout << "Using expansions: " << properties.use_expansions << " with order " << properties.expansion_order << std::endl;
		
		// adimensionalize properties
		properties = adimensionalizeProperties(properties);
		std::cout << "depth: " << properties.depth << ", kappa: " << properties.kappa << ", rho: " << properties.rho << std::endl;
		// create the options for GL
		GaussLegendre2Options glOptions = createOptionsFromCOptions(*glCOptions);

		// calculators for the f(y)
		HeliumBoundaryProblem<N, 1> heliumProblem(properties);
		BaseBoundaryIntegralCalculator<N, 1> calculator(properties, heliumProblem);
		RealBoundaryItegralCalculator<N> realCalculator(calculator);
		// calculators for the jacobian
		HeliumBoundaryProblem<N, 3 * N> heliumJacProblem(properties);
		std::unique_ptr<AutonomousProblem<std_complex, 6 * N * N>> jacBoundaryIntegralCalculatorPtr = std::make_unique<BaseBoundaryIntegralCalculator<N, 3 * N>>(properties, heliumJacProblem);
		JacobianCalculator<N> jacobianCalculator(std::move(jacBoundaryIntegralCalculatorPtr));

		GaussLegendre2<N> integrator(realCalculator, jacobianCalculator, glOptions);
		integrator.setStream(cudaStreamPerThread);

		integrator.initialize(initialState, false);
		integrator.runEvolution(glCOptions->t0, glCOptions->t1);

		// copy results to output
		cudaStreamSynchronize(cudaStreamPerThread);
		integrator.copyTimesToHost(timesOut, timesCount);
		integrator.copyStatesToHost(statesOut, statesCount);
	}
	catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return -1;
	}
	return 0;
}

int integrateSimulationGL2(double* initialState, double** statesOut, size_t* statesCount,
	double** timesOut, size_t* timesCount,
	SimProperties* simProperties, GaussLegendreOptions* glCOptions, size_t N) 
{
	switch (N)
	{
		case 32:
			return integrateSimulationGL2_N<32>(initialState, statesOut, statesCount, timesOut, timesCount, simProperties, glCOptions);
		case 64:
			return integrateSimulationGL2_N<64>(initialState, statesOut, statesCount, timesOut, timesCount, simProperties, glCOptions);
		case 128:
			return integrateSimulationGL2_N<128>(initialState, statesOut, statesCount, timesOut, timesCount, simProperties, glCOptions);
		case 256:
			return integrateSimulationGL2_N<256>(initialState, statesOut, statesCount, timesOut, timesCount, simProperties, glCOptions);
		case 512:
			return integrateSimulationGL2_N<512>(initialState, statesOut, statesCount, timesOut, timesCount, simProperties, glCOptions);
		case 1024:
			return integrateSimulationGL2_N<1024>(initialState, statesOut, statesCount, timesOut, timesCount, simProperties, glCOptions);
	default:
		std::cerr << "Error: Unsupported N size " << N << std::endl;
		std::cerr << "Supported N sizes are: 32, 64, 128 , 256, 512, 1024" << std::endl;
		return -1;
	}
}

int integrateSimulationGL2_freeMemory(double* statesOut, double* timesOut)
{
	std::free(statesOut);
	std::free(timesOut);
	return 0;
}

template <size_t N>
int integrateSimulationRK4_N(double* initialState, double** statesOut, size_t* statesCount, double** timesOut, size_t* timesCount, SimProperties* simProperties, RK4Options* rkOptions)
{
	try {
		ProblemProperties properties;
		copyProperties(*simProperties, properties);
		// adimensionalize properties
		properties = adimensionalizeProperties(properties);
	}
	catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return -1;
	}
}

int integrateSimulationRK4(double* initialState, double** statesOut, size_t* statesCount, double** timesOut, size_t* timesCount, SimProperties* simProperties, RK4SolverOptions* rkOptions, size_t N)
{
	return 0;
}

int integrateSimulationRK4_freeMemory(double* statesOut, double* timesOut)
{
	std::free(statesOut);
	std::free(timesOut);
	return 0;
}
