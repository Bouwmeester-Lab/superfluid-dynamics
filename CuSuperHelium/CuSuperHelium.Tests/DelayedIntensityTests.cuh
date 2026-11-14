#pragma once
// Phase 8 gate: the delayed intensity D is part of the state, so any solver integrates it.
//
// With the film frozen (Z and Phi do not move), the intensity I(Z) is fixed and D obeys dD/dt = (I - D) / tau,
// whose solution is D(t) = I + (D0 - I) exp(-t / tau). The same problem is integrated with RK4, RK45 and
// Gauss-Legendre (through ComplexAsRealProblem and FiniteDifferenceJacobian) and compared with that solution.
// N = 300 puts the per-node kernels on two blocks, where the retired timed drive raced on its saved time.

#include <gtest/gtest.h>
#include <cmath>
#include <memory>
#include <vector>

#include "Core/DeviceBuffer.cuh"
#include "Ode/AutonomousProblem.h"
#include "Ode/ComplexAsRealProblem.h"
#include "Ode/GaussLegendre.cuh"
#include "Ode/JacobianProvider.h"
#include "Ode/RK45.cuh"
#include "Ode/RungeKutta4.cuh"
#include "Optomechanics/DelayedIntensityIntegrator.cuh"
#include "Optomechanics/LightIntensity.cuh"

namespace delayed_intensity_test
{
	constexpr int N = 300;
	constexpr double TAU = 1.0;
	constexpr double T_END = 1.0;

	/// State (Z, Phi, D), N complex values each. Only D evolves; the drive on Phi is dropped, so the film stays put.
	class FrozenFilmDelayedIntensity final : public AutonomousProblem<std_complex>
	{
	public:
		FrozenFilmDelayedIntensity(OptomechanicalVariables& variables, DimensionlessParameters& properties, std::shared_ptr<LightIntensity> intensity)
			: AutonomousProblem<std_complex>(3 * N), delayed(variables, properties, intensity, N) {}

		void run(const std_complex* y, std_complex* dydt) override
		{
			CUSH_CHECK(cudaMemsetAsync(dydt, 0, 3 * N * sizeof(std_complex), stream));
			delayed.run(y, dydt);
			CUSH_CHECK(cudaMemsetAsync(dydt + N, 0, N * sizeof(std_complex), stream)); // keep Phi frozen
		}

		void setStream(cudaStream_t s) override { stream = s; delayed.setStream(s); }

	private:
		DelayedIntensityIntegrator delayed;
		cudaStream_t stream = cudaStreamPerThread;
	};

	struct Fixture
	{
		OptomechanicalVariables variables;
		DimensionlessParameters properties;
		std::shared_ptr<LightIntensity> intensity = std::make_shared<LightIntensity>();
		std::vector<std_complex> initial = std::vector<std_complex>(3 * N);
		DeviceBuffer<std_complex> devInitial = DeviceBuffer<std_complex>(3 * N);
		std::vector<double> I = std::vector<double>(N);   // intensity felt by D at each node
		std::vector<double> D0 = std::vector<double>(N);

		Fixture()
		{
			variables.detuning = 0.2;
			variables.gamma = 0.5;
			variables.G = -0.02;
			variables.Tau = TAU;
			variables.max_intensity = 0.1;
			variables.location_x0_mode = PI_d;
			variables.sigma_optical_mode = 0.2;
			variables.sigma_thermal_mode = 0.5;
			variables.Beta = 1.0;
			variables.DampingStrength = 0.0;
			properties.rho = 1.0;
			properties.depth = 0.1;

			for (int j = 0; j < N; ++j)
			{
				const double x = 2.0 * PI_d * j / N;
				D0[j] = 0.05 * (1.0 + std::cos(3.0 * x)); // a known, non-uniform initial history
				initial[j] = std_complex(x, 0.01 * std::sin(x));
				initial[N + j] = std_complex(0.0, 0.0);
				initial[2 * N + j] = std_complex(D0[j], 0.0);
			}
			CUSH_CHECK(cudaMemcpy(devInitial.get(), initial.data(), devInitial.bytes(), cudaMemcpyHostToDevice));

			// The integrator reads the frequency shift computed from Z; Z is frozen, so once is enough.
			intensity->allocate(N);
			GaussianDistribution shape{ variables.location_x0_mode, variables.sigma_optical_mode };
			intensity->compute_weights<GaussianDistribution>(devInitial.get(), variables, shape);
			intensity->compute_frequency_shift(devInitial.get(), variables, properties);
			CUSH_CHECK(cudaDeviceSynchronize());
		}

		/// I from the problem itself: at D = 0, dD/dt = I / tau.
		void measureIntensity(FrozenFilmDelayedIntensity& problem)
		{
			std::vector<std_complex> zeroD = initial;
			for (int j = 0; j < N; ++j) zeroD[2 * N + j] = 0.0;
			DeviceBuffer<std_complex> y(3 * N), dydt(3 * N);
			CUSH_CHECK(cudaMemcpy(y.get(), zeroD.data(), y.bytes(), cudaMemcpyHostToDevice));
			problem.rhs(y.get(), 0.0, dydt.get());
			std::vector<std_complex> host(3 * N);
			CUSH_CHECK(cudaDeviceSynchronize());
			CUSH_CHECK(cudaMemcpy(host.data(), dydt.get(), dydt.bytes(), cudaMemcpyDeviceToHost));
			for (int j = 0; j < N; ++j) I[j] = TAU * host[2 * N + j].real();
		}

		/// Largest error of D(T_END) against the exact solution, relative to max |I|.
		double relativeError(const std::vector<std_complex>& finalState) const
		{
			double maxError = 0.0, scale = 0.0;
			for (int j = 0; j < N; ++j)
			{
				const double exact = I[j] + (D0[j] - I[j]) * std::exp(-T_END / TAU);
				maxError = std::max(maxError, std::abs(finalState[2 * N + j].real() - exact));
				scale = std::max(scale, std::abs(I[j]));
			}
			return maxError / scale;
		}
	};

	inline std::vector<std_complex> copyState(const void* devState)
	{
		std::vector<std_complex> host(3 * N);
		CUSH_CHECK(cudaDeviceSynchronize());
		CUSH_CHECK(cudaMemcpy(host.data(), devState, host.size() * sizeof(std_complex), cudaMemcpyDeviceToHost));
		return host;
	}
}

TEST(DelayedIntensity, IntensityIsNonTrivialOnSeveralBlocks)
{
	using namespace delayed_intensity_test;
	Fixture f;
	FrozenFilmDelayedIntensity problem(f.variables, f.properties, f.intensity);
	f.measureIntensity(problem);
	double minI = f.I[0], maxI = f.I[0];
	for (double v : f.I) { minI = std::min(minI, v); maxI = std::max(maxI, v); }
	EXPECT_GT(maxI, 0.0);
	EXPECT_GT(maxI - minI, 0.1 * maxI) << "the intensity profile should vary across the nodes";
	// nodes in the second block (j >= 256) see a non-zero intensity too
	EXPECT_GT(std::abs(f.I[N - 10]), 0.0);
}

TEST(DelayedIntensity, RungeKutta4MatchesExactRelaxation)
{
	using namespace delayed_intensity_test;
	Fixture f;
	FrozenFilmDelayedIntensity problem(f.variables, f.properties, f.intensity);
	f.measureIntensity(problem);

	RungeKutta4<std_complex> solver(problem, 0.05 * TAU);
	solver.initialize(f.devInitial.get(), true);
	ASSERT_EQ(solver.runEvolution(0.0, T_END), OdeSolverResult::ReachedEndTime);
	EXPECT_LT(f.relativeError(copyState(solver.state())), 1e-7);
}

TEST(DelayedIntensity, RK45MatchesExactRelaxation)
{
	using namespace delayed_intensity_test;
	Fixture f;
	FrozenFilmDelayedIntensity problem(f.variables, f.properties, f.intensity);
	f.measureIntensity(problem);

	RK45_std_complex solver(problem, 0.05 * TAU);
	solver.setTolerance(1e-11, 1e-11);
	solver.initialize(f.devInitial.get(), true);
	ASSERT_EQ(solver.runEvolution(0.0, T_END), OdeSolverResult::ReachedEndTime);
	EXPECT_LT(f.relativeError(copyState(solver.state())), 1e-7);
}

namespace delayed_intensity_test
{
	/// Gauss-Legendre with the generic finite-difference Jacobian, on the real view of the complex state.
	inline double gaussLegendreError(Fixture& f, FrozenFilmDelayedIntensity& problem, double h)
	{
		ComplexAsRealProblem realProblem(problem);
		FiniteDifferenceJacobian jacobian(realProblem, 1e-6);
		GaussLegendre2Options options;
		options.stepSize = h;
		GaussLegendre2 solver(realProblem, jacobian, options);
		solver.setStream(cudaStreamPerThread);
		solver.initialize(reinterpret_cast<const double*>(f.devInitial.get()), true);
		EXPECT_EQ(solver.runEvolution(0.0, T_END), OdeSolverResult::ReachedEndTime);
		return f.relativeError(copyState(solver.state()));
	}
}

// Gauss-Legendre (2 stages) is order 4: halving the step divides the error by about 16. Before Phase 8 its Newton
// iteration built the stages from the initial guess, which made it about order 3 (ratio 7).
TEST(DelayedIntensity, GaussLegendreWithFiniteDifferenceJacobianIsFourthOrderAndExact)
{
	using namespace delayed_intensity_test;
	Fixture f;
	FrozenFilmDelayedIntensity problem(f.variables, f.properties, f.intensity);
	f.measureIntensity(problem);

	const double coarse = gaussLegendreError(f, problem, 0.2 * TAU);
	const double fine = gaussLegendreError(f, problem, 0.1 * TAU);
	EXPECT_LT(fine, 1e-7);
	EXPECT_NEAR(coarse / fine, 16.0, 2.0);
}
