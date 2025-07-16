
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
    problemProperties.depth = 0.11;
    double h = 0.1;
    double omega = 1;
    double t0 = 0;

    const int N = 256;
    
	const double stepSize = PI_d/4000;
	const int steps = 0.9 / stepSize;
	const int loggingSteps = steps / 10;
    

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