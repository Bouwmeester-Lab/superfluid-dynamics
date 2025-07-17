
//#define DEBUG_DERIVATIVES
//#define DEBUG_RUNGE_KUTTA
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
	return h * std::cos((j - omega * t));
}

double Phi(double j, double h, double omega, double t, double rho) {
    return h * ((1 + rho) * omega * std::sin((j - omega * t)));
}

int main() 
{
	ProblemProperties problemProperties;
    problemProperties.rho = 0;
	problemProperties.kappa = 0;
    problemProperties.U = 0;
    
    double omega = 1;
    double t0 = 0;
    
    double g = 3 * 2.6e-24 / std::pow(15e-9, 4); //
	double H0 = 15e-9; // 15 nm
	double L0 = 40e-6/(2.0*PI_d); // 40 um

    double _t0 = std::sqrt(L0 / g);


    problemProperties.depth = H0 / L0;
    double h = 0.01 * problemProperties.depth;

	printf("Simulating with depth %.10e, h %.10e, omega %f, t0 %.10e, L0 %.10e\n", problemProperties.depth, h, omega, _t0, L0);
	printf("g %.10e, H0 %.10e, L0 %.10e\n", g, H0, L0);

    const int N = 128;
    
	const double stepSize = PI_d/4000;
	const int steps = 100 * 3.1415 / stepSize;
	const int loggingSteps = steps / 1000;
    

    std::array<std_complex, N> Z0;
	std::array<std_complex, N> PhiArr;
    double j;
	for (int i = 0; i < N; i++) {
		j = 2.0 * PI_d * i / (1.0 * N);
		Z0[i] = std_complex(X(j, h, omega, t0), Y(j, h, omega, t0));
        PhiArr[i] = Phi(j, h, omega, t0, problemProperties.rho);
	}

    ParticleData particleData;
	particleData.Z = Z0.data();
	particleData.Potential = PhiArr.data();

    
     return runSimulationHelium<N>(steps, stepSize, problemProperties, particleData, loggingSteps);
}