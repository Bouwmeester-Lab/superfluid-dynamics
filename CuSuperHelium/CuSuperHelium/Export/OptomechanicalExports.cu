#include "ExportInternal.cuh"

#include "../HeliumWithDrivingBoundaryProblem.cuh"
#include "../LightIntensity.cuh"
#include "../RK4_Time_Dependent.cuh"
#include "../SimulationRunner.cuh"
#include "../TimedBoundaryIntegrator.cuh"
#include "../TrajectoryLogger.cuh"

#include <complex>
#include <memory>
#include <vector>

template <size_t N>
int integrateOptomechanicalSimulationRK4_N(double* initialState, double** statesOut, size_t* statesCount, double** timesOut, size_t* timesCount, SimProperties* simProperties, COptomechanicalVariables* optoVariables, RK4SolverOptions* rkOptions) 
{
	try 
	{
		ProblemProperties properties;
		//OptomechanicalVariables optoVars;
		copyProperties(*simProperties, properties);
		OptomechanicalVariables optoVars;

		copyProperties(*optoVariables, optoVars);

		// adimensionalize properties
		properties = adimensionalizeProperties(properties);
		optoVars = adimensionalizeOptomechanicalVariables(optoVars, properties);
		// transfoms SI units to adimensional units for the RK4 options
		auto rk4SolverOptions = adimensionalizeRK4SolverOptions(*rkOptions, properties);

		// print properties for debugging
		std::cout << "Adimensionalized optomechanical properties. " << std::endl;
		printOptomechanicalVariables(optoVars);

		RK4Options rk4_options;

		rk4_options.initial_timestep = rk4SolverOptions.timeStep;
		rk4_options.returnTrajectory = rk4SolverOptions.returnTrajectory;
		const double t0 = rk4SolverOptions.t0;
		const double t1 = rk4SolverOptions.t1;

		HeliumWithOptomechanicalDrivingProblem<N> heliumProblem(properties, optoVars, std::make_unique<LightIntensity>());
		TimedBoundaryIntegrator<N, 1> integrator(properties, heliumProblem);

		RungeKuttaStepper<std_complex, 2 * N> stepper(integrator);

		stepper.setOptions(rk4_options);

		std::vector<std::complex<double>> Z0(N);
		std::vector<double> Phi(N);

		for(size_t i = 0; i < N; i++)
		{
			Z0[i] = std::complex<double>(initialState[i], initialState[i + N]);
			Phi[i] = initialState[i + 2 * N];
		}

		ParticleData particleData(Z0, Phi);
		DeviceParticleData deviceParticleData;

		loadDataToDevice(particleData, deviceParticleData, N);
		cudaStreamSynchronize(cudaStreamPerThread);

		stepper.initialize(deviceParticleData.devZ, true);

		std::cout << "Starting RK4 evolution from t = " << t0 << " to t = " << t1 << " with time step " << rk4_options.initial_timestep << std::endl;

		stepper.runEvolution(t0, t1);

		std::cout << "RK4 evolution completed. Copying results to host." << std::endl;

		cudaStreamSynchronize(cudaStreamPerThread);

		stepper.copyTimesToHost(timesOut, timesCount);

		std::cout << "Times copied to host. Copying states to host." << std::endl;

		std_complex* hostStates;
		stepper.copyStatesToHost(&hostStates, statesCount);

		std::cout << "Complex states copied to host. Transforming to output format." << std::endl;

		// transform std::complex to double arrays for output
		// this must be freed by the caller
		 double* states = (double*)std::malloc(3 * (*statesCount) * sizeof(double) * N);
		 size_t countStatesHost = *statesCount;

		 for (size_t j = 0; j < countStatesHost; j++) 
		 {
			 for (size_t i = 0; i < N; i++)
			 {
				 states[j * 3 * N + i] = hostStates[j*2* N + i].real();
				 states[j * 3 * N + i + N] = hostStates[j * 2 * N +i].imag();
				 states[j * 3 * N + i + 2 * N] = hostStates[j * 2 * N + N + i].real();
			 }
		 }

		 std::cout << "States transformed to output format. Setting output pointers." << std::endl;

		 *statesOut = states;
		 // free the one created by the stepper

		 std::cout << "Freeing host states memory." << std::endl;

		 std::free(hostStates);
	}
	catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return -1;
	}

	return 0;
}

int integrateOptomechanicalSimulationRK4(double* initialState, double** statesOut, size_t* statesCount, double** timesOut, size_t* timesCount, SimProperties* simProperties, RK4SolverOptions* rkOptions, COptomechanicalVariables* optomechanicalVariables, size_t N)
{
	switch (N) {
		case 32:
			return integrateOptomechanicalSimulationRK4_N<32>(initialState, statesOut, statesCount, timesOut, timesCount, simProperties, optomechanicalVariables, rkOptions);
		case 64:
			return integrateOptomechanicalSimulationRK4_N<64>(initialState, statesOut, statesCount, timesOut, timesCount, simProperties, optomechanicalVariables, rkOptions);
		case 128:
			return integrateOptomechanicalSimulationRK4_N<128>(initialState, statesOut, statesCount, timesOut, timesCount, simProperties, optomechanicalVariables, rkOptions);
		case 256:
			return integrateOptomechanicalSimulationRK4_N<256>(initialState, statesOut, statesCount, timesOut, timesCount, simProperties, optomechanicalVariables, rkOptions);
		case 512:
			return integrateOptomechanicalSimulationRK4_N<512>(initialState, statesOut, statesCount, timesOut, timesCount, simProperties, optomechanicalVariables, rkOptions);
		case 1024:
			return integrateOptomechanicalSimulationRK4_N<1024>(initialState, statesOut, statesCount, timesOut, timesCount, simProperties, optomechanicalVariables, rkOptions);
		case 2048:
			return integrateOptomechanicalSimulationRK4_N<2048>(initialState, statesOut, statesCount, timesOut, timesCount, simProperties, optomechanicalVariables, rkOptions);
		case 4096:
			return integrateOptomechanicalSimulationRK4_N<4096>(initialState, statesOut, statesCount, timesOut, timesCount, simProperties, optomechanicalVariables, rkOptions);
		case 8192:
			return integrateOptomechanicalSimulationRK4_N<8192>(initialState, statesOut, statesCount, timesOut, timesCount, simProperties, optomechanicalVariables, rkOptions);
	}
	return 0;
}

int integrateOptomechanicalSimulationRK4_freeMemory(double* statesOut, double* timesOut)
{
	std::free(statesOut);
	std::free(timesOut);
	return 0;
}
