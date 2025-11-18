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

}


static_assert(std::is_trivially_copyable_v<c_double>);
static_assert(std::is_trivially_copyable_v<std::complex<double>>);

static_assert(sizeof(c_double) == sizeof(std::complex<double>),
	"Size mismatch: cannot bit_cast/memcpy elementwise");
static_assert(alignof(std_complex) == 16, "Your premise must hold");
static_assert(alignof(c_double) == 16, "Over-alignment failed");
