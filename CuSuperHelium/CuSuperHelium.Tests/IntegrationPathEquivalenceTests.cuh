#pragma once

#include <gtest/gtest.h>
#include <memory>
#include <vector>

#include "Optomechanics/AugmentedBoundaryIntegrator.cuh"
#include "Ode/RungeKutta4.cuh"
#include "BoundaryIntegral/Periodic2D/BaseBoundaryIntegrator.cuh"
#include "Optomechanics/DelayedIntensityIntegrator.cuh"
#include "Optomechanics/HeliumDrivenAutonomousProblem.cuh"
#include "Optomechanics/LightIntensity.cuh"
#include "IO/TrajectoryLogger.cuh"
#include "Core/CudaChecks.cuh"
#include "matplotlibcpp.h"
#include "Optomechanics/AugmentedIntensityBoundaryIntegrator.cuh"
#include "Optomechanics/HeliumRampedIntensityAutonomousProblem.cuh"

// NullFluidTimedAndAugmentedDrivingAgree compared the timed optomechanical drive with the augmented one; the timed
// drive was retired in Phase 8 (its delayed intensity is now part of the state). See DelayedIntensityTests.cuh.

TEST(IntegrationPaths, RampedIntensityAugmentedTest) 
{
	constexpr int N = 32;
	constexpr double dt = 1e-1;
	constexpr double t0 = 0.0;
	constexpr double t1 = 1'000 * dt;
	constexpr double tolerance = 1e-10;

	DimensionlessParameters properties;
	properties.scales.energy = 1.0;
	properties.scales.time = 1.0;
	properties.rho = 1.0;
	properties.depth = 1.0;

	OptomechanicalVariables optomechanicalVariables;
	optomechanicalVariables.detuning = 0.0;
	optomechanicalVariables.gamma = 100;
	optomechanicalVariables.G = 1.0;
	optomechanicalVariables.Tau = 1.0;
	optomechanicalVariables.max_intensity = 1000.0;
	optomechanicalVariables.initial_time = t0;
	optomechanicalVariables.location_x0_mode = PI_d / 2.0;
	optomechanicalVariables.sigma_optical_mode = 0.2;
	optomechanicalVariables.sigma_thermal_mode = 0.5;
	optomechanicalVariables.Beta = 1.0e6;
	optomechanicalVariables.DampingStrength = 0.0;

	std::vector<std_complex> augmentedInitialState(3 * N + 1, std_complex(0.0, 0.0));

	double x;
	for (size_t i = 0; i < N; ++i)
	{
		/// Set the initial state for everything to zero except for x:
		x = 2.0 * PI_d / static_cast<double>(N) * i;
		augmentedInitialState[i] = std_complex(x, 0.0);
	}

	std_complex* devInitialState = nullptr;
	CUSH_CHECK(cudaMalloc(&devInitialState, augmentedInitialState.size() * sizeof(std_complex)));
	CUSH_CHECK(cudaMemcpy(devInitialState, augmentedInitialState.data(), augmentedInitialState.size() * sizeof(std_complex), cudaMemcpyHostToDevice));

	std::shared_ptr<LightIntensity> lightIntensity = std::make_shared<LightIntensity>();

	
	HeliumRampedIntensityAutonomousProblem heliumProblem(properties, optomechanicalVariables, lightIntensity, N);

	std::unique_ptr<BaseBoundaryIntegralCalculator> boundaryIntegralCalculator = std::make_unique<BaseBoundaryIntegralCalculator>(properties, heliumProblem);

	AugmentedIntensityBoundaryIntegrator augmentedIntegrator(
		std::move(boundaryIntegralCalculator),
		std::make_unique<DelayedRampedIntensityIntegrator>(optomechanicalVariables, properties, lightIntensity, N));

	auto logger = std::make_shared<TrajectoryLogger<std_complex>>();

	RungeKutta4<std_complex> augmentedStepper(augmentedIntegrator, dt);
	augmentedStepper.addObserver(logger);

	augmentedStepper.initialize(devInitialState, true);

	EXPECT_EQ(augmentedStepper.runEvolution(t0, t1), OdeSolverResult::ReachedEndTime);

	CUSH_CHECK(cudaDeviceSynchronize());

	std::vector<std_complex> augmentedFinalState(3 * N + 1);

	std_complex* hostStates;
	size_t countHost;
	logger->copyStatesToHost(&hostStates, &countHost);

	std::vector<double> x_vect(N, 0.0);
	std::vector<double> y_vect(N, 0.0);
	std::vector<double> ramped_intensity(countHost, 0.0);
	for (size_t i = 0; i < N; ++i)
	{
		x_vect[i] = hostStates[countHost - (3 * N + 1) + i].real();
		y_vect[i] = hostStates[countHost - (3 * N + 1) + i].imag();
		
	}

	for (size_t i = 0; i < countHost; ++i)
	{
		ramped_intensity[i] = hostStates[i * (3*N+1) + 3*N].real();
	}

	plt::figure();
	plt::title("Final State: ramped intensity interface");
	plt::plot(x_vect, y_vect, { {"label", "Augmented Final State"} });
	plt::legend();
	plt::figure();
	plt::title("Ramped Intensity over Time");
	plt::plot(ramped_intensity, { {"label", "Ramped Intensity"} });
	plt::legend();

	//CUSH_CHECK(cudaMemcpy(augmentedFinalState.data(), logger->, augmentedFinalState.size() * sizeof(std_complex), cudaMemcpyDeviceToHost));
	// delete the hostStates array
	std::free(hostStates);

}
