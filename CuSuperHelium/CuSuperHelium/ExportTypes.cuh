#pragma once
#include <complex>
#include <type_traits>

extern "C"
{
	struct __declspec(align(16)) c_double { double re; double im; };
	struct SimProperties {
		double L;
		double rho;
		double kappa;
		double depth;

		// for using expansions
		bool use_expansions;
		int expansion_order;

		// infinite_depth?
		bool infinite_depth = false;
	};
	struct GaussLegendreOptions
	{
		double t0;
		double t1;
		double stepSize;
		double newtonTolerance;
		size_t maxNewtonIterations;
		bool allowSimplifiedFallback = true;
		bool returnTrajectory = true;
		double armijo_c = 1e-4;
		double backtrack = 0.5;
		double minAlpha = 1e-6;
		size_t maxStepsHalves = 6;
	};
	struct RK4SolverOptions {
		double timeStep;
		double t0;
		double t1;
		bool returnTrajectory = true;
	};
	struct COptomechanicalVariables 
	{
		// initial detuning in the experiment
		double detuning = 0.0;
		// optical linewidth of the resonator
		double gamma = 1.0;

		// optomechanical coupling strength (Hz/m)
		double G = 1.0;

		// "delayed" strength of the optical effect on the superfluid.
		double tau = 1.0;

		// max intensity of the optical field
		double max_intensity;

		double initial_time = 0.0;

		double location_x0_mode = 0.0;
		double sigma_optical_mode = 1.0;

		double beta = 1.0;
		double damping_strength = 1.0;
	};
}


static_assert(std::is_trivially_copyable_v<c_double>);
static_assert(std::is_trivially_copyable_v<std::complex<double>>);

static_assert(sizeof(c_double) == sizeof(std::complex<double>),
	"Size mismatch: cannot bit_cast/memcpy elementwise");
static_assert(alignof(std_complex) == 16, "Your premise must hold");
static_assert(alignof(c_double) == 16, "Over-alignment failed");
