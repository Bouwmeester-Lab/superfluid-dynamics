#include "Export/ExportInternal.cuh"

#include "Ode/AutonomousProblem.h"
#include "BoundaryIntegral/Periodic2D/BaseBoundaryIntegrator.cuh"
#include "Ode/GaussLegendre.cuh"
#include "Physics/HeliumBoundaryProblem.cuh"
#include "Ode/JacobianCalculator.cuh"
#include "BoundaryIntegral/Periodic2D/RealBoundaryIntegralCalculator.cuh"
#include "Ode/RungeKutta4.cuh"

#include <memory>

/// The C ABI options to the solver's own options. Whether a trajectory is returned is the export's choice.
static GaussLegendre2Options createOptionsFromCOptions(const GaussLegendreOptions& cOptions)
{
	GaussLegendre2Options options;
	options.stepSize = cOptions.stepSize;
	options.newtonTolerance = cOptions.newtonTolerance;
	options.maxNewtonIterations = cOptions.maxNewtonIterations;
	options.allowSimplifiedFallback = cOptions.allowSimplifiedFallback;
	options.armijo_c = cOptions.armijo_c;
	options.backtrack = cOptions.backtrack;
	options.minAlpha = cOptions.minAlpha;
	options.maxStepsHalves = cOptions.maxStepsHalves;
	return options;
}

template <size_t N>
int integrateSimulationGL2_N(double* initialState, double** statesOut, size_t* statesCount,
	double** timesOut, size_t* timesCount,
	SimProperties* simProperties, GaussLegendreOptions* glCOptions)
{
	try {
		DimensionlessParameters properties = toDimensionlessParameters(*simProperties);
		std::cout << "Properties copied." << std::endl;
		std::cout << "Using expansions: " << properties.use_expansions << " with order " << properties.expansion_order << std::endl;
		
		std::cout << "depth: " << properties.depth << ", kappa: " << properties.kappa << ", rho: " << properties.rho << std::endl;
		// create the options for GL
		GaussLegendre2Options glOptions = createOptionsFromCOptions(*glCOptions);

		// calculators for the f(y)
		HeliumBoundaryProblem<N, 1> heliumProblem(properties);
		BaseBoundaryIntegralCalculator<N, 1> calculator(properties, heliumProblem);
		RealBoundaryIntegralCalculator<N> realCalculator(calculator);
		// calculators for the jacobian
		HeliumBoundaryProblem<N, 3 * N> heliumJacProblem(properties);
		std::unique_ptr<AutonomousProblem<std_complex, 6 * N * N>> jacBoundaryIntegralCalculatorPtr = std::make_unique<BaseBoundaryIntegralCalculator<N, 3 * N>>(properties, heliumJacProblem);
		JacobianCalculator<N> jacobianCalculator(std::move(jacBoundaryIntegralCalculatorPtr));

		GaussLegendre2<N> integrator(realCalculator, jacobianCalculator, glOptions);
		integrator.setStream(cudaStreamPerThread);

		// Gauss-Legendre has always returned the initial state as the first trajectory entry.
		auto trajectory = std::make_shared<TrajectoryLogger<double>>(true);
		if (glCOptions->returnTrajectory)
		{
			integrator.addObserver(trajectory);
		}

		integrator.initialize(initialState, false);
		integrator.runEvolution(glCOptions->t0, glCOptions->t1);

		// copy results to output
		if (copyIntegrationResultToHost<double>(integrator, 3 * N, glCOptions->returnTrajectory ? trajectory.get() : nullptr, statesOut, statesCount, timesOut, timesCount) != 0)
		{
			std::cerr << "Failed to copy the integration result to host" << std::endl;
			return -1;
		}
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
