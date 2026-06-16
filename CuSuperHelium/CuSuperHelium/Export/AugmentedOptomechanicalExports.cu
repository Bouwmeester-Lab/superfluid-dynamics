#include "ExportInternal.cuh"

#include "../AugmentedBoundaryIntegrator.cuh"
#include "../AugmentedIntensityBoundaryIntegrator.cuh"
#include "../AutonomousRungeKuttaStepper.cuh"
#include "../BaseBoundaryIntegrator.cuh"
#include "../DelayedIntensityIntegrator.cuh"
#include "../HeliumDrivenAutonomousProblem.cuh"
#include "../HeliumRampedIntensityAutonomousProblem.cuh"
#include "../LightIntensity.cuh"
#include "../TrajectoryLogger.cuh"
#include "../HeliumSineAutonomousProblem.cuh"
#include "../GeneralAugmentedBoundaryIntegrator.cuh"

#include <memory>
#include <vector>

template <size_t N>
int integrateAugmentedOptomechanicalSimulationRK4_N(double* initialState, double** statesOut, size_t* statesCount, double** timesOut, size_t* timesCount, SimProperties* simProperties, RK4SolverOptions* rkOptions, COptomechanicalVariables* optomechanicalVariables) 
{
	try {
		ProblemProperties properties;
		//OptomechanicalVariables optoVars;
		copyProperties(*simProperties, properties);
		OptomechanicalVariables optoVars;

		copyProperties(*optomechanicalVariables, optoVars);

		std::cout << "Optomechanical properties before adimensionalization: " << std::endl;
		printOptomechanicalVariables(optoVars);

		// adimensionalize properties
		properties = adimensionalizeProperties(properties);
		optoVars = adimensionalizeOptomechanicalVariables(optoVars, properties);

		std::cout << "Optomechanical properties after adimensionalization: " << std::endl;
		printOptomechanicalVariables(optoVars);

		// transfoms SI units to adimensional units for the RK4 options
		auto rk4SolverOptions = adimensionalizeRK4SolverOptions(*rkOptions, properties);

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

			std::shared_ptr<TrajectoryLogger<std_complex, 3 * N + 1>> logger = std::make_shared<TrajectoryLogger<std_complex, 3 * N + 1>>();
			HeliumRampedIntensityAutonomousProblem<N, 1> heliumProblem(properties, optoVars, lightIntensity);
			std::unique_ptr<BaseBoundaryIntegralCalculator<N, 1>> boundaryIntegralCalculator = std::make_unique<BaseBoundaryIntegralCalculator<N, 1>>(properties, heliumProblem);
			AugmentedIntensityBoundaryIntegrator<N, 1> integrator(
				std::move(boundaryIntegralCalculator),
				std::make_unique<DelayedRampedIntensityIntegrator<N, 1>>(optoVars, properties, lightIntensity));

			AutonomousRungeKuttaStepper<std_complex, 3 * N + 1> stepper(integrator, rk4_options.initial_timestep, logger);

			stepper.setOptions(rk4_options);
			stepper.initialize(cplx_initialState.data(), false);

			std::cout << "Starting ramped augmented RK4 evolution from t = " << t0 << " to t = " << t1 << " with time step " << rk4_options.initial_timestep << std::endl;
			stepper.runEvolution(t0, t1);

			std::cout << "Ramped augmented RK4 evolution completed. Copying results to host." << std::endl;
			cudaStreamSynchronize(cudaStreamPerThread);

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

			std::shared_ptr<TrajectoryLogger<std_complex, 3 * N + 1>> logger = std::make_shared<TrajectoryLogger<std_complex, 3 * N + 1>>();
			HeliumSineAutonomousProblem<N, 1> heliumProblem(properties, optoVars, lightIntensity);
			std::unique_ptr<BaseBoundaryIntegralCalculator<N, 1>> boundaryIntegralCalculator = std::make_unique<BaseBoundaryIntegralCalculator<N, 1>>(properties, heliumProblem);
			GeneralAugmentedBoundaryIntegrator<N, 1, 1> integrator(
				std::move(boundaryIntegralCalculator),
				std::make_unique<DelayedSineIntensityIntegrator<N, 1>>(optoVars, properties, lightIntensity));

			AutonomousRungeKuttaStepper<std_complex, 3 * N + 1> stepper(integrator, rk4_options.initial_timestep, logger);

			stepper.setOptions(rk4_options);
			stepper.initialize(cplx_initialState.data(), false);

			std::cout << "Starting sine augmented RK4 evolution from t = " << t0 << " to t = " << t1 << " with time step " << rk4_options.initial_timestep << std::endl;
			stepper.runEvolution(t0, t1);

			std::cout << "Sine augmented RK4 evolution completed. Copying results to host." << std::endl;
			cudaStreamSynchronize(cudaStreamPerThread);

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

			std::shared_ptr<TrajectoryLogger<std_complex, 3 * N>> logger = std::make_shared<TrajectoryLogger<std_complex, 3 * N>>();
			HeliumDrivenAutonomousProblem<N, 1> heliumProblem(properties, optoVars, lightIntensity);
			std::unique_ptr<BaseBoundaryIntegralCalculator<N, 1>> boundaryIntegralCalculator = std::make_unique<BaseBoundaryIntegralCalculator<N, 1>>(properties, heliumProblem);
			AugmentedBoundaryIntegrator<N, 1> integrator(std::move(boundaryIntegralCalculator), std::make_unique<DelayedIntensityIntegrator<N, 1>>(optoVars, properties, lightIntensity));

			AutonomousRungeKuttaStepper<std_complex, 3 * N> stepper(integrator, rk4_options.initial_timestep, logger);

			stepper.setOptions(rk4_options);
			stepper.initialize(cplx_initialState.data(), false);

			std::cout << "Starting augmented RK4 evolution from t = " << t0 << " to t = " << t1 << " with time step " << rk4_options.initial_timestep << std::endl;
			stepper.runEvolution(t0, t1);

			std::cout << "Augmented RK4 evolution completed. Copying results to host." << std::endl;
			cudaStreamSynchronize(cudaStreamPerThread);

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
	switch (N) {
		case 32:
			return integrateAugmentedOptomechanicalSimulationRK4_N<32>(initialState, statesOut, statesCount, timesOut, timesCount, simProperties, rkOptions, optomechanicalVariables);
		case 64:
			return integrateAugmentedOptomechanicalSimulationRK4_N<64>(initialState, statesOut, statesCount, timesOut, timesCount, simProperties, rkOptions, optomechanicalVariables);
		case 128:
			return integrateAugmentedOptomechanicalSimulationRK4_N<128>(initialState, statesOut, statesCount, timesOut, timesCount, simProperties, rkOptions, optomechanicalVariables);
		case 256:
			return integrateAugmentedOptomechanicalSimulationRK4_N<256>(initialState, statesOut, statesCount, timesOut, timesCount, simProperties, rkOptions, optomechanicalVariables);
		case 512:
			return integrateAugmentedOptomechanicalSimulationRK4_N<512>(initialState, statesOut, statesCount, timesOut, timesCount, simProperties, rkOptions, optomechanicalVariables);
		case 1024:
			return integrateAugmentedOptomechanicalSimulationRK4_N<1024>(initialState, statesOut, statesCount, timesOut, timesCount, simProperties, rkOptions, optomechanicalVariables);
		case 2048:
			return integrateAugmentedOptomechanicalSimulationRK4_N<2048>(initialState, statesOut, statesCount, timesOut, timesCount, simProperties, rkOptions, optomechanicalVariables);
		case 4096:
			return integrateAugmentedOptomechanicalSimulationRK4_N<4096>(initialState, statesOut, statesCount, timesOut, timesCount, simProperties, rkOptions, optomechanicalVariables);
		case 8192:
			return integrateAugmentedOptomechanicalSimulationRK4_N<8192>(initialState, statesOut, statesCount, timesOut, timesCount, simProperties, rkOptions, optomechanicalVariables);
	}
	return -1;
}

int integrateAugmentedOptomechanicalSimulationRK4_freeMemory(double* statesOut, double* timesOut)
{
	std::free(statesOut);
	std::free(timesOut);
	return 0;
}

template <size_t N>
int calculateRhsAugmentedOptomechanical_N(double* state, double* rhs, SimProperties* simProperties, COptomechanicalVariables* optomechanicalVariables)
{
	std_complex* devState;
	std_complex* devRhs;
	std_complex* hostRhs;
	bool error = false;
	try {
		ProblemProperties properties;
		//OptomechanicalVariables optoVars;
		copyProperties(*simProperties, properties);
		OptomechanicalVariables optoVars;

		copyProperties(*optomechanicalVariables, optoVars);

		// adimensionalize properties
		properties = adimensionalizeProperties(properties);
		optoVars = adimensionalizeOptomechanicalVariables(optoVars, properties);

		std::vector<std_complex> cplx_initialState(3 * N);

		for (size_t i = 0; i < N; i++)
		{
			cplx_initialState[i] = std_complex(state[i], state[i + N]); // Z
			cplx_initialState[i + N] = std_complex(state[i + 2 * N], 0); // phi
			cplx_initialState[i + 2 * N] = std_complex(state[i + 3 * N], 0); // delayed intensity
		}

		std::shared_ptr<TrajectoryLogger<std_complex, 3 * N>> logger = std::make_shared<TrajectoryLogger<std_complex, 3 * N>>();
		std::shared_ptr<LightIntensity> lightIntensity = std::make_shared<LightIntensity>();

		HeliumDrivenAutonomousProblem<N, 1> heliumProblem(properties, optoVars, lightIntensity);
		std::unique_ptr<BaseBoundaryIntegralCalculator<N, 1>> boundaryIntegralCalculator = std::make_unique<BaseBoundaryIntegralCalculator<N, 1>>(properties, heliumProblem);

		AugmentedBoundaryIntegrator<N, 1> integrator(std::move(boundaryIntegralCalculator), std::make_unique<DelayedIntensityIntegrator<N, 1>>(optoVars, properties, lightIntensity));

		// copy the state to device

		checkCuda(setDevice());
		checkCuda(cudaMalloc(&devState, sizeof(std_complex) * 3 * N));
		checkCuda(cudaMalloc(&devRhs, sizeof(std_complex) * 3 * N));
		checkCuda(cudaMemcpy(devState, cplx_initialState.data(), sizeof(std_complex) * 3 * N, cudaMemcpyHostToDevice));

		integrator.run(devState, devRhs);

		cudaStreamSynchronize(cudaStreamPerThread);
		// copy rhs back to host
		hostRhs = (std_complex*)std::malloc(sizeof(std_complex) * 3 * N);
		checkCuda(cudaMemcpy(hostRhs, devRhs, sizeof(std_complex) * 3 * N, cudaMemcpyDeviceToHost));
		// transform to output format
		for (size_t i = 0; i < N; i++)
		{
			rhs[i] = hostRhs[i].real();
			rhs[i + N] = hostRhs[i].imag();
			rhs[i + 2 * N] = hostRhs[i + N].real(); // d/dt phi 
			rhs[i + 3 * N] = hostRhs[i + 2 * N].real(); // d/dt delayed intensity
		}
		checkCuda(cudaDeviceSynchronize());
	}
	catch(const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		// return -1;
	}

	// free memory allocated, both on host and device
	if(devState != nullptr) {
		cudaFree(devState);
	}
	if(devRhs != nullptr) {
		cudaFree(devRhs);
	}
	if(hostRhs != nullptr) {
		std::free(hostRhs);
	}
	checkCuda(cudaDeviceSynchronize());
	
	return error;
}

int calculateRhsAugmentedOptomechanical(double* state, double* rhs, SimProperties* simProperties, COptomechanicalVariables* optomechanicalVariables, size_t N)
{
	switch(N) {
		case 32:
			return calculateRhsAugmentedOptomechanical_N<32>(state, rhs, simProperties, optomechanicalVariables);
		case 64:
			return calculateRhsAugmentedOptomechanical_N<64>(state, rhs, simProperties, optomechanicalVariables);
		case 128:
			return calculateRhsAugmentedOptomechanical_N<128>(state, rhs, simProperties, optomechanicalVariables);
		case 256:
			return calculateRhsAugmentedOptomechanical_N<256>(state, rhs, simProperties, optomechanicalVariables);
		case 512:
			return calculateRhsAugmentedOptomechanical_N<512>(state, rhs, simProperties, optomechanicalVariables);
		case 1024:
			return calculateRhsAugmentedOptomechanical_N<1024>(state, rhs, simProperties, optomechanicalVariables);
		case 2048:
			return calculateRhsAugmentedOptomechanical_N<2048>(state, rhs, simProperties, optomechanicalVariables);
		case 4096:
			return calculateRhsAugmentedOptomechanical_N<4096>(state, rhs, simProperties, optomechanicalVariables);
		case 8192:
			return calculateRhsAugmentedOptomechanical_N<8192>(state, rhs, simProperties, optomechanicalVariables);
	}
	return -1;
}
