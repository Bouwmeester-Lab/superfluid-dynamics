#pragma once

#include <cmath>
#include <cstdlib>
#include <vector>

#include <gtest/gtest.h>

#include "constants.cuh"
#include "ExportTypes.cuh"

extern "C"
{
	int integrateAugmentedOptomechanicalSimulationRK4(double* initialState, double** statesOut, size_t* statesCount, double** timesOut, size_t* timesCount, SimProperties* simProperties, RK4SolverOptions* rkOptions, COptomechanicalVariables* optomechanicalVariables, size_t N);
	int integrateAugmentedOptomechanicalSimulationRK4_freeMemory(double* statesOut, double* timesOut);
}

namespace
{
constexpr size_t ExportTestN = 32;

SimProperties makeExportTestSimProperties()
{
	SimProperties properties{};
	properties.L = 2.0 * PI_d;
	properties.rho = 150.0;
	properties.kappa = 0.0;
	properties.depth = 1.0;
	properties.use_expansions = false;
	properties.expansion_order = 2;
	properties.infinite_depth = false;
	return properties;
}

RK4SolverOptions makeExportTestRk4Options()
{
	RK4SolverOptions options{};
	options.timeStep = 1.0e-3;
	options.t0 = 0.0;
	options.t1 = options.timeStep;
	options.returnTrajectory = true;
	return options;
}

COptomechanicalVariables makeExportTestOptomechanicalVariables(bool rampIntensity)
{
	COptomechanicalVariables variables{};
	variables.detuning = 0.0;
	variables.gamma = 100.0;
	variables.G = 1.0;
	variables.tau = 1.0;
	variables.max_intensity = 1000.0;
	variables.initial_time = 0.0;
	variables.location_x0_mode = PI_d / 2.0;
	variables.sigma_optical_mode = 0.2;
	variables.sigma_thermal_mode = 0.5;
	variables.beta = 1.0e6;
	variables.damping_strength = 0.0;
	if (rampIntensity) {
		variables.drive_type = CDRIVE_TYPE_Ramped;
	}
	else {
		variables.drive_type = CDRIVE_TYPE_Constant;
	}
	variables.ramp_rate = 0.25;
	return variables;
}

std::vector<double> makeAugmentedExportInitialState(size_t stride)
{
	std::vector<double> initialState(stride, 0.0);
	for (size_t i = 0; i < ExportTestN; ++i)
	{
		initialState[i] = 2.0 * PI_d * static_cast<double>(i) / static_cast<double>(ExportTestN);
		initialState[i + ExportTestN] = 0.0;
		initialState[i + 2 * ExportTestN] = 0.0;
		initialState[i + 3 * ExportTestN] = 0.0;
	}
	return initialState;
}
}

TEST(ExportFunctions, AugmentedOptomechanicalRK4UsesNonRampedStateLayout)
{
	std::vector<double> initialState = makeAugmentedExportInitialState(4 * ExportTestN);
	SimProperties simProperties = makeExportTestSimProperties();
	RK4SolverOptions rkOptions = makeExportTestRk4Options();
	COptomechanicalVariables optomechanicalVariables = makeExportTestOptomechanicalVariables(false);

	double* statesOut = nullptr;
	double* timesOut = nullptr;
	size_t statesCount = 0;
	size_t timesCount = 0;

	const int result = integrateAugmentedOptomechanicalSimulationRK4(
		initialState.data(),
		&statesOut,
		&statesCount,
		&timesOut,
		&timesCount,
		&simProperties,
		&rkOptions,
		&optomechanicalVariables,
		ExportTestN);

	ASSERT_EQ(result, 0);
	ASSERT_NE(statesOut, nullptr);
	ASSERT_NE(timesOut, nullptr);
	ASSERT_GT(statesCount, 0u);
	EXPECT_EQ(timesCount, statesCount);

	const size_t lastStateOffset = (statesCount - 1) * 4 * ExportTestN;
	for (size_t i = 0; i < 4 * ExportTestN; ++i)
	{
		EXPECT_TRUE(std::isfinite(statesOut[lastStateOffset + i])) << "non-ramped output index " << i;
	}

	integrateAugmentedOptomechanicalSimulationRK4_freeMemory(statesOut, timesOut);
}

TEST(ExportFunctions, AugmentedOptomechanicalRK4UsesRampedStateLayout)
{
	constexpr double initialRampIntensity = 2.0;
	std::vector<double> initialState = makeAugmentedExportInitialState(4 * ExportTestN + 1);
	initialState[4 * ExportTestN] = initialRampIntensity;

	SimProperties simProperties = makeExportTestSimProperties();
	RK4SolverOptions rkOptions = makeExportTestRk4Options();
	COptomechanicalVariables optomechanicalVariables = makeExportTestOptomechanicalVariables(true);

	double* statesOut = nullptr;
	double* timesOut = nullptr;
	size_t statesCount = 0;
	size_t timesCount = 0;

	const int result = integrateAugmentedOptomechanicalSimulationRK4(
		initialState.data(),
		&statesOut,
		&statesCount,
		&timesOut,
		&timesCount,
		&simProperties,
		&rkOptions,
		&optomechanicalVariables,
		ExportTestN);

	ASSERT_EQ(result, 0);
	ASSERT_NE(statesOut, nullptr);
	ASSERT_NE(timesOut, nullptr);
	ASSERT_GT(statesCount, 0u);
	EXPECT_EQ(timesCount, statesCount);

	const size_t lastStateOffset = (statesCount - 1) * (4 * ExportTestN + 1);
	for (size_t i = 0; i < 4 * ExportTestN + 1; ++i)
	{
		EXPECT_TRUE(std::isfinite(statesOut[lastStateOffset + i])) << "ramped output index " << i;
	}

	const double expectedRampIntensity = initialRampIntensity + optomechanicalVariables.ramp_rate * rkOptions.timeStep;
	EXPECT_NEAR(statesOut[lastStateOffset + 4 * ExportTestN], expectedRampIntensity, 1.0e-9);

	integrateAugmentedOptomechanicalSimulationRK4_freeMemory(statesOut, timesOut);
}
