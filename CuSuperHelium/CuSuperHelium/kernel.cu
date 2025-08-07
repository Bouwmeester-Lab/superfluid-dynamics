
//#define DEBUG_DERIVATIVES
//#define DEBUG_RUNGE_KUTTA
//#define DEBUG_DERIVATIVES_3
//#define DEBUG_VELOCITIES
#include "ProblemProperties.hpp"
#include "array"
#include "vector"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <cufft.h>

#include "constants.cuh"
#include <complex>

#include "WaterBoundaryIntegralCalculator.cuh"
#include "SimpleEuler.cuh"
#include "AutonomousRungeKuttaStepper.cuh"
#include "ValueLogger.h"
#include "SolitonPeak.h"

#include <format>
//#include "math.h"
//#include "complex.h"
#include "matplotlibcpp.h"
#include "SimulationRunner.cuh"
namespace plt = matplotlibcpp;

#define j_complex std::complex<double>(0, 1)
cudaError_t setDevice();


__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

double X(double j, double h, double omega, double t) {
    return j - h * std::sin((j - omega * t));
}

double Y(double j, double h, double omega, double t) {
    return h * PeriodicFunctions::gaussianNarrow::gaussianNarrow_periodic(j);  // std::cos((j - omega * t));
}

double Phi(double j, double h, double omega, double t, double rho) {
    return 0.0*h * (1 + rho) * omega * std::sin(j);// PeriodicFunctions::bimodal::bimodal(j);// std::sech2_periodic((j - omega * t));
}

int main() 
{
	ProblemProperties problemProperties;
    problemProperties.rho = 0;
	problemProperties.kappa = 0;
    problemProperties.U = 0;
    
    int frames = 500;
    double omega = 1;
    double t0 = 0;
	double finalTime = 12e-3; // 15 ms
    
    double H0 = 15e-9; // 15 nm
    double g = 3 * 2.6e-24 / std::pow(H0, 4); //
	double L0 = 500e-6/(2.0*PI_d); // 6mm

    double _t0 = std::sqrt(L0 / g);

    problemProperties.depth = H0 / L0;
    double h = 0.1 * problemProperties.depth;

	problemProperties.initial_amplitude = h;
	problemProperties.y_min = -h - 0.0001 * problemProperties.depth; // -0.5 * H0
	problemProperties.y_max = h +  0.005 * problemProperties.depth; // 0.5 * H0
	printf("Simulating with depth (h_0) %.10e, h %.10e, omega %f, t0 %.10e, L0 %.10e\n", problemProperties.depth, h, omega, _t0, L0);
	printf("g %.10e, H0 %.10e, L0 %.10e\n", g, H0, L0);

    const int N = 1024;//512;
    
	const double stepSize = 0.015;
    const int steps = (finalTime / _t0) / stepSize;
	const int loggingSteps = steps / frames;

    printf("Simulating %i steps representing %.2e s", steps, steps * stepSize * _t0);
    

    std::array<std_complex, N> Z0;
	std::vector<double> X0(N, 0);
    std::vector<double> Y0(N, 0);
    
	std::array<std_complex, N> PhiArr;

    std::vector<double> Phireal(N, 0);
    double j;
	for (int i = 0; i < N; i++) {
		j = 2.0 * PI_d * i / (1.0 * N);
		Z0[i] = std_complex(X(j, h, omega, t0), Y(j, h, omega, t0));
        PhiArr[i] = std_complex(Phi(j, h, omega, t0, problemProperties.rho), 0.0);
		Phireal[i] = PhiArr[i].real();

		X0[i] = Z0[i].real();
		Y0[i] = Z0[i].imag();
	}

	plt::figure();
    plt::plot(X0, Phireal);
	plt::plot(X0, Y0);
    plt::show();

    ParticleData particleData;
	particleData.Z = Z0.data();
	particleData.Potential = PhiArr.data();

    
     return runSimulationHelium<N>(steps, stepSize, problemProperties, particleData, loggingSteps, true, true, _t0);
}