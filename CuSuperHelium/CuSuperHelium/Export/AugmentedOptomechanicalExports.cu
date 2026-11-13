#include "Export/ExportInternal.cuh"

#include "Optomechanics/AugmentedBoundaryIntegrator.cuh"
#include "Optomechanics/AugmentedIntensityBoundaryIntegrator.cuh"
#include "Ode/RungeKutta4.cuh"
#include "IO/ProgressBarObserver.h"
#include "BoundaryIntegral/Periodic2D/BaseBoundaryIntegrator.cuh"
#include "Optomechanics/DelayedIntensityIntegrator.cuh"
#include "Optomechanics/HeliumDrivenAutonomousProblem.cuh"
#include "Optomechanics/HeliumRampedIntensityAutonomousProblem.cuh"
#include "Optomechanics/LightIntensity.cuh"
#include "IO/TrajectoryLogger.cuh"
#include "Optomechanics/HeliumSineAutonomousProblem.cuh"
#include "Optomechanics/GeneralAugmentedBoundaryIntegrator.cuh"

#include <memory>
#include <vector>

static int integrateAugmentedOptomechanicalSimulationRK4_N(const size_t N, double* initialState, double** statesOut, size_t* statesCount, double** timesOut, size_t* timesCount, SimProperties* simProperties, RK4SolverOptions* rkOptions, COptomechanicalVariables* optomechanicalVariables) 
{
	try {
		DimensionlessParameters properties = toDimensionlessParameters(*simProperties);
		OptomechanicalVariables optoVars;

		copyProperties(*optomechanicalVariables, optoVars);

		std::cout << "Optomechanical properties before adimensionalization: " << std::endl;
		printOptomechanicalVariables(optoVars);

		optoVars = NonDimensionalization::nonDimensionalize(optoVars, properties.scales);

		std::cout << "Optomechanical properties after adimensionalization: " << std::endl;
		printOptomechanicalVariables(optoVars);

		// transfoms SI units to adimensional units for the RK4 options
		auto rk4SolverOptions = toDimensionless(*rkOptions, properties.scales);

		RK4Options rk4_options;

		rk4_options.initial_timestep = rk4SolverOptions.timeStep;
		rk4_options.returnTrajectory = rk4SolverOptions.returnTrajectory;
		const double t0 = rk4SolverOptions.t0;
		const double t1 = rk4SolverOptions.t1;

		std::shared_ptr<LightIntensity> lightIntensity = std::make_shared<LightIntensity>();

		if (optoVars.drive_type == DriveType::Ramped)
		{
			std::vector<std_complex> cplx_initialState(3 * N + 1);

			for (size_t i = 0; i < N; i++)
			{
				cplx_initialState[i] = std_complex(initialState[i], initialState[i + N]); // Z
				cplx_initialState[i + N] = std_complex(initialState[i + 2 * N], 0); // phi
				cplx_initialState[i + 2 * N] = std_complex(initialState[i + 3 * N], 0); // delayed intensity
			}
			cplx_initialState[3 * N] = std_complex(initialState[4 * N], 0); // scalar ramped intensity

			auto logger = std::make_shared<TrajectoryLogger<std_complex>>();
			HeliumRampedIntensityAutonomousProblem heliumProblem(properties, optoVars, lightIntensity, static_cast<int>(N));
			std::unique_ptr<BaseBoundaryIntegralCalculator> boundaryIntegralCalculator = std::make_unique<BaseBoundaryIntegralCalculator>(properties, heliumProblem);
			AugmentedIntensityBoundaryIntegrator integrator(
				std::move(boundaryIntegralCalculator),
				std::make_unique<DelayedRampedIntensityIntegrator>(optoVars, properties, lightIntensity, static_cast<int>(N)));

			RungeKutta4<std_complex> stepper(integrator, rk4_options.initial_timestep);
			stepper.addObserver(logger);
			stepper.addObserver(std::make_shared<ProgressBarObserver<std_complex>>(t0, t1, "Running integration for " + std::to_string(3 * N + 1) + " variables"));

			stepper.setOptions(rk4_options);
			stepper.initialize(cplx_initialState.data(), false);

			std::cout << "Starting ramped augmented RK4 evolution from t = " << t0 << " to t = " << t1 << " with time step " << rk4_options.initial_timestep << std::endl;
			stepper.runEvolution(t0, t1);

			std::cout << "Ramped augmented RK4 evolution completed. Copying results to host." << std::endl;
			CUSH_CHECK(cudaStreamSynchronize(cudaStreamPerThread));

			logger->copyTimesToHost(timesOut, timesCount);
			std::cout << (*timesCount) << " times copied to host" << "Copying states to host." << std::endl;
			std_complex* hostStates;

			if (logger->copyStatesToHost(&hostStates, statesCount) == -1)
			{
				std::cerr << "Failed to copy states to host" << std::endl;
				return -1;
			}
			std::cout << (*statesCount) << " complex states copied to host. Transforming to output format." << std::endl;

			double* states = (double*)std::malloc((4 * N + 1) * (*statesCount) * sizeof(double));
			size_t countStatesHost = *statesCount;

			for (size_t j = 0; j < countStatesHost; j++)
			{
				const size_t hostOffset = j * (3 * N + 1);
				const size_t outputOffset = j * (4 * N + 1);

				for (size_t i = 0; i < N; i++)
				{
					states[outputOffset + i] = hostStates[hostOffset + i].real();
					states[outputOffset + i + N] = hostStates[hostOffset + i].imag();
					states[outputOffset + i + 2 * N] = hostStates[hostOffset + N + i].real(); // phi
					states[outputOffset + i + 3 * N] = hostStates[hostOffset + 2 * N + i].real(); // delayed intensity
				}
				states[outputOffset + 4 * N] = hostStates[hostOffset + 3 * N].real(); // scalar ramped intensity
			}

			std::cout << countStatesHost << " ramped states transformed to output format. Setting output pointers." << std::endl;

			*statesOut = states;

			std::cout << "Freeing host states memory." << std::endl;

			std::free(hostStates);
		}
		else if (optoVars.drive_type == DriveType::Sine)
		{
			std::vector<std_complex> cplx_initialState(3 * N + 1);

			for (size_t i = 0; i < N; i++)
			{
				cplx_initialState[i] = std_complex(initialState[i], initialState[i + N]); // Z
				cplx_initialState[i + N] = std_complex(initialState[i + 2 * N], 0); // phi
				cplx_initialState[i + 2 * N] = std_complex(initialState[i + 3 * N], 0); // delayed intensity
			}
			cplx_initialState[3 * N] = std_complex(initialState[4 * N], 0); // scalar ramped intensity

			auto logger = std::make_shared<TrajectoryLogger<std_complex>>();
			HeliumSineAutonomousProblem heliumProblem(properties, optoVars, lightIntensity, static_cast<int>(N));
			std::unique_ptr<BaseBoundaryIntegralCalculator> boundaryIntegralCalculator = std::make_unique<BaseBoundaryIntegralCalculator>(properties, heliumProblem);
			GeneralAugmentedBoundaryIntegrator integrator(
				std::move(boundaryIntegralCalculator),
				std::make_unique<DelayedSineIntensityIntegrator>(optoVars, properties, lightIntensity, static_cast<int>(N)));

			RungeKutta4<std_complex> stepper(integrator, rk4_options.initial_timestep);
			stepper.addObserver(logger);
			stepper.addObserver(std::make_shared<ProgressBarObserver<std_complex>>(t0, t1, "Running integration for " + std::to_string(3 * N + 1) + " variables"));

			stepper.setOptions(rk4_options);
			stepper.initialize(cplx_initialState.data(), false);

			std::cout << "Starting sine augmented RK4 evolution from t = " << t0 << " to t = " << t1 << " with time step " << rk4_options.initial_timestep << std::endl;
			stepper.runEvolution(t0, t1);

			std::cout << "Sine augmented RK4 evolution completed. Copying results to host." << std::endl;
			CUSH_CHECK(cudaStreamSynchronize(cudaStreamPerThread));

			logger->copyTimesToHost(timesOut, timesCount);
			std::cout << (*timesCount) << " times copied to host" << "Copying states to host." << std::endl;
			std_complex* hostStates;

			if (logger->copyStatesToHost(&hostStates, statesCount) == -1)
			{
				std::cerr << "Failed to copy states to host" << std::endl;
				return -1;
			}
			std::cout << (*statesCount) << " complex states copied to host. Transforming to output format." << std::endl;

			double* states = (double*)std::malloc((4 * N + 1) * (*statesCount) * sizeof(double));
			size_t countStatesHost = *statesCount;

			for (size_t j = 0; j < countStatesHost; j++)
			{
				const size_t hostOffset = j * (3 * N + 1);
				const size_t outputOffset = j * (4 * N + 1);

				for (size_t i = 0; i < N; i++)
				{
					states[outputOffset + i] = hostStates[hostOffset + i].real();
					states[outputOffset + i + N] = hostStates[hostOffset + i].imag();
					states[outputOffset + i + 2 * N] = hostStates[hostOffset + N + i].real(); // phi
					states[outputOffset + i + 3 * N] = hostStates[hostOffset + 2 * N + i].real(); // delayed intensity
				}
				states[outputOffset + 4 * N] = hostStates[hostOffset + 3 * N].real(); // scalar theta
			}

			std::cout << countStatesHost << " sine states transformed to output format. Setting output pointers." << std::endl;

			*statesOut = states;

			std::cout << "Freeing host states memory." << std::endl;

			std::free(hostStates);
		}
		else
		{
			std::vector<std_complex> cplx_initialState(3 * N);

			for (size_t i = 0; i < N; i++)
			{
				cplx_initialState[i] = std_complex(initialState[i], initialState[i + N]); // Z
				cplx_initialState[i + N] = std_complex(initialState[i + 2 * N], 0); // phi
				cplx_initialState[i + 2 * N] = std_complex(initialState[i + 3 * N], 0); // delayed intensity
			}

			auto logger = std::make_shared<TrajectoryLogger<std_complex>>();
			HeliumDrivenAutonomousProblem heliumProblem(properties, optoVars, lightIntensity, static_cast<int>(N));
			std::unique_ptr<BaseBoundaryIntegralCalculator> boundaryIntegralCalculator = std::make_unique<BaseBoundaryIntegralCalculator>(properties, heliumProblem);
			AugmentedBoundaryIntegrator integrator(std::move(boundaryIntegralCalculator), std::make_unique<DelayedIntensityIntegrator>(optoVars, properties, lightIntensity, static_cast<int>(N)));

			RungeKutta4<std_complex> stepper(integrator, rk4_options.initial_timestep);
			stepper.addObserver(logger);
			stepper.addObserver(std::make_shared<ProgressBarObserver<std_complex>>(t0, t1, "Running integration for " + std::to_string(3 * N) + " variables"));

			stepper.setOptions(rk4_options);
			stepper.initialize(cplx_initialState.data(), false);

			std::cout << "Starting augmented RK4 evolution from t = " << t0 << " to t = " << t1 << " with time step " << rk4_options.initial_timestep << std::endl;
			stepper.runEvolution(t0, t1);

			std::cout << "Augmented RK4 evolution completed. Copying results to host." << std::endl;
			CUSH_CHECK(cudaStreamSynchronize(cudaStreamPerThread));

			logger->copyTimesToHost(timesOut, timesCount);
			std::cout << (*timesCount) << " times copied to host" << "Copying states to host." << std::endl;
			std_complex* hostStates;

			if (logger->copyStatesToHost(&hostStates, statesCount) == -1)
			{
				std::cerr << "Failed to copy states to host" << std::endl;
				return -1;
			}
			std::cout << (*statesCount) << " complex states copied to host. Transforming to output format." << std::endl;
			// transform std::complex to double arrays for output
			double* states = (double*)std::malloc(4 * (*statesCount) * sizeof(double) * N);
			size_t countStatesHost = *statesCount;

			for (size_t j = 0; j < countStatesHost; j++)
			{
				for (size_t i = 0; i < N; i++)
				{
					states[j * 4 * N + i] = hostStates[j * 3 * N + i].real();
					states[j * 4 * N + i + N] = hostStates[j * 3 * N + i].imag();
					states[j * 4 * N + i + 2 * N] = hostStates[j * 3 * N + N + i].real(); // phi
					states[j * 4 * N + i + 3 * N] = hostStates[j * 3 * N + 2 * N + i].real(); // delayed intensity
				}
			}

			std::cout << countStatesHost << " states transformed to output format. Setting output pointers." << std::endl;

			*statesOut = states;
			// free the one created by the stepper

			std::cout << "Freeing host states memory." << std::endl;

			std::free(hostStates);
		}

	}
	catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return -1;
	}
	return 0;
}

int integrateAugmentedOptomechanicalSimulationRK4(double* initialState, double** statesOut, size_t* statesCount, double** timesOut, size_t* timesCount, SimProperties* simProperties, RK4SolverOptions* rkOptions, COptomechanicalVariables* optomechanicalVariables, size_t N)
{
	if (!validPeriodicSize(N)) return -1;
	return integrateAugmentedOptomechanicalSimulationRK4_N(N, initialState, statesOut, statesCount, timesOut, timesCount, simProperties, rkOptions, optomechanicalVariables);
}

int integrateAugmentedOptomechanicalSimulationRK4_freeMemory(double* statesOut, double* timesOut)
{
	std::free(statesOut);
	std::free(timesOut);
	return 0;
}

static int calculateRhsAugmentedOptomechanical_N(const size_t N, double* state, double* rhs, SimProperties* simProperties, COptomechanicalVariables* optomechanicalVariables)
{
	try {
		DimensionlessParameters properties = toDimensionlessParameters(*simProperties);
		OptomechanicalVariables optoVars;

		copyProperties(*optomechanicalVariables, optoVars);

		optoVars = NonDimensionalization::nonDimensionalize(optoVars, properties.scales);

		std::vector<std_complex> cplx_initialState(3 * N);

		for (size_t i = 0; i < N; i++)
		{
			cplx_initialState[i] = std_complex(state[i], state[i + N]); // Z
			cplx_initialState[i + N] = std_complex(state[i + 2 * N], 0); // phi
			cplx_initialState[i + 2 * N] = std_complex(state[i + 3 * N], 0); // delayed intensity
		}

		std::shared_ptr<LightIntensity> lightIntensity = std::make_shared<LightIntensity>();

		HeliumDrivenAutonomousProblem heliumProblem(properties, optoVars, lightIntensity, static_cast<int>(N));
		std::unique_ptr<BaseBoundaryIntegralCalculator> boundaryIntegralCalculator = std::make_unique<BaseBoundaryIntegralCalculator>(properties, heliumProblem);

		AugmentedBoundaryIntegrator integrator(std::move(boundaryIntegralCalculator), std::make_unique<DelayedIntensityIntegrator>(optoVars, properties, lightIntensity, static_cast<int>(N)));

		// copy the state to device

		CUSH_CHECK(setDevice());
		DeviceBuffer<std_complex> devState(3 * N);
		DeviceBuffer<std_complex> devRhs(3 * N);
		CUSH_CHECK(cudaMemcpy(devState.get(), cplx_initialState.data(), devState.bytes(), cudaMemcpyHostToDevice));

		integrator.run(devState.get(), devRhs.get());

		CUSH_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
		// copy rhs back to host
		std::vector<std_complex> hostRhs(3 * N);
		CUSH_CHECK(cudaMemcpy(hostRhs.data(), devRhs.get(), devRhs.bytes(), cudaMemcpyDeviceToHost));
		// transform to output format
		for (size_t i = 0; i < N; i++)
		{
			rhs[i] = hostRhs[i].real();
			rhs[i + N] = hostRhs[i].imag();
			rhs[i + 2 * N] = hostRhs[i + N].real(); // d/dt phi 
			rhs[i + 3 * N] = hostRhs[i + 2 * N].real(); // d/dt delayed intensity
		}
		CUSH_CHECK(cudaDeviceSynchronize());
	}
	catch(const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return -1;
	}

	return 0;
}

int calculateRhsAugmentedOptomechanical(double* state, double* rhs, SimProperties* simProperties, COptomechanicalVariables* optomechanicalVariables, size_t N)
{
	if (!validPeriodicSize(N)) return -1;
	return calculateRhsAugmentedOptomechanical_N(N, state, rhs, simProperties, optomechanicalVariables);
}
