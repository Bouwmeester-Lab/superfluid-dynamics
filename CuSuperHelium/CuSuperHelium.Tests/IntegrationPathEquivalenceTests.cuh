#pragma once

#include <gtest/gtest.h>
#include <memory>
#include <vector>

#include "AugmentedBoundaryIntegrator.cuh"
#include "AutonomousRungeKuttaStepper.cuh"
#include "BaseBoundaryIntegrator.cuh"
#include "DelayedIntensityIntegrator.cuh"
#include "HeliumDrivenAutonomousProblem.cuh"
#include "HeliumWithDrivingBoundaryProblem.cuh"
#include "LightIntensity.cuh"
#include "RK4_Time_Dependent.cuh"
#include "TimedBoundaryIntegrator.cuh"
#include "TrajectoryLogger.cuh"
#include "utilities.cuh"
#include "matplotlibcpp.h"

TEST(IntegrationPaths, NullFluidTimedAndAugmentedDrivingAgree)
{
	constexpr int N = 32;
	constexpr double dt = 1e-1;
	constexpr double t0 = 0.0;
	constexpr double t1 = 100'000* dt;
	constexpr double tolerance = 1e-10;

	ProblemProperties timedProperties;
	timedProperties.base_energy = 1.0;
	timedProperties.base_time = 1.0;
	timedProperties.rho = 1.0;
	timedProperties.depth = 1.0;

	ProblemProperties augmentedProperties = timedProperties;

	OptomechanicalVariables timedVariables;
	timedVariables.detuning = 0.0;
	timedVariables.gamma = 100;
	timedVariables.G = 1.0;
	timedVariables.Tau = 1.0;
	timedVariables.max_intensity = 1000.0;
	timedVariables.initial_time = t0;
	timedVariables.location_x0_mode = PI_d / 2.0;
	timedVariables.sigma_optical_mode = 0.2;
	timedVariables.sigma_thermal_mode = 0.5;
	timedVariables.Beta = 1.0e6;
	timedVariables.DampingStrength = 0.0;

	OptomechanicalVariables augmentedVariables = timedVariables;

	std::vector<std_complex> timedInitialState(2 * N, std_complex(0.0, 0.0));
	std::vector<std_complex> augmentedInitialState(3 * N, std_complex(0.0, 0.0));

	double x;
	for(size_t i = 0; i < N; ++i)
	{
		/// Set the initial state for everything to zero except for x:
		x = 2.0 * PI_d / static_cast<double>(N) * i;
		timedInitialState[i] = std_complex(x, 0.0);
		augmentedInitialState[i] = std_complex(x, 0.0);
	}

	std_complex* devTimedState = nullptr;
	std_complex* devAugmentedState = nullptr;
	checkCuda(cudaMalloc(&devTimedState, timedInitialState.size() * sizeof(std_complex)));
	checkCuda(cudaMalloc(&devAugmentedState, augmentedInitialState.size() * sizeof(std_complex)));
	checkCuda(cudaMemcpy(devTimedState, timedInitialState.data(), timedInitialState.size() * sizeof(std_complex), cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(devAugmentedState, augmentedInitialState.data(), augmentedInitialState.size() * sizeof(std_complex), cudaMemcpyHostToDevice));

	std::unique_ptr<LightIntensity> lightIntensityTimed = std::make_unique<LightIntensity>();

	HeliumWithOptomechanicalDrivingProblem<N> timedBoundaryProblem(timedProperties, timedVariables, std::move(lightIntensityTimed));
	TimedBoundaryIntegrator<N, 1> timedIntegrator(timedProperties, timedBoundaryProblem);
	RungeKuttaStepper<std_complex, 2 * N> timedStepper(timedIntegrator, dt);
	timedStepper.initialize(devTimedState, true);

	std::shared_ptr<LightIntensity> lightIntensity = std::make_shared<LightIntensity>();
	HeliumDrivenAutonomousProblem<N, 1> augmentedBoundaryProblem(augmentedProperties, augmentedVariables, lightIntensity);
	std::unique_ptr<BaseBoundaryIntegralCalculator<N, 1>> augmentedBoundaryIntegralCalculator =
		std::make_unique<BaseBoundaryIntegralCalculator<N, 1>>(augmentedProperties, augmentedBoundaryProblem);
	AugmentedBoundaryIntegrator<N, 1> augmentedIntegrator(
		std::move(augmentedBoundaryIntegralCalculator),
		std::make_unique<DelayedIntensityIntegrator<N, 1>>(augmentedVariables, augmentedProperties, lightIntensity));
	auto logger = std::make_shared<TrajectoryLogger<std_complex, 3 * N>>();

	AutonomousRungeKuttaStepper<std_complex, 3 * N> augmentedStepper(augmentedIntegrator, dt, logger);
	augmentedStepper.initialize(devAugmentedState, true);

	EXPECT_EQ(timedStepper.runEvolution(t0, t1), OdeSolverResult::ReachedEndTime);
	EXPECT_EQ(augmentedStepper.runEvolution(t0, t1), OdeSolverResult::ReachedEndTime);
	checkCuda(cudaDeviceSynchronize());

	std::vector<std_complex> timedFinalState(2 * N);
	std::vector<std_complex> augmentedFinalState(3 * N);
	checkCuda(cudaMemcpy(timedFinalState.data(), devTimedState, timedFinalState.size() * sizeof(std_complex), cudaMemcpyDeviceToHost));
	checkCuda(cudaMemcpy(augmentedFinalState.data(), devAugmentedState, augmentedFinalState.size() * sizeof(std_complex), cudaMemcpyDeviceToHost));

	for (int i = 0; i < N; ++i)
	{
		EXPECT_NEAR(timedFinalState[i].real(), augmentedFinalState[i].real(), tolerance) << "Z real mismatch at index " << i;
		EXPECT_NEAR(timedFinalState[i].imag(), augmentedFinalState[i].imag(), tolerance) << "Z imag mismatch at index " << i;
		EXPECT_NEAR(timedFinalState[N + i].real(), augmentedFinalState[N + i].real(), tolerance) << "Phi real mismatch at index " << i;
		EXPECT_NEAR(timedFinalState[N + i].imag(), augmentedFinalState[N + i].imag(), tolerance) << "Phi imag mismatch at index " << i;
	}

	std::vector<double> x_timed(N);
	std::vector<double> y_timed(N);
	std::vector<double> x_augmented(N);
	std::vector<double> y_augmented(N);

	for(int i = 0; i < N; ++i)
	{
		x_timed[i] = timedFinalState[i].real();
		y_timed[i] = timedFinalState[i].imag();
		x_augmented[i] = augmentedFinalState[i].real();
		y_augmented[i] = augmentedFinalState[i].imag();
	}

	plt::figure();
	plt::plot(x_timed, y_timed, {{"label", "Timed"}});
	plt::plot(x_augmented, y_augmented, {{"label", "Augmented"}});
	plt::legend();

	checkCuda(cudaFree(devTimedState));
	checkCuda(cudaFree(devAugmentedState));
}
