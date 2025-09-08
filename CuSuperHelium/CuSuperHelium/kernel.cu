
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
    return h * std::cos(j);// PeriodicFunctions::gaussian::gaussian_periodic(j);  // std::cos((j - omega * t));
}

double Phi(double j, double h, double omega, double t, double rho) {
    return h * (1 + rho) * omega * std::sin(j);// PeriodicFunctions::bimodal::bimodal(j);// std::sech2_periodic((j - omega * t));
}



void createRk45Solver() 
{
    
}

int dispersionTest(double wavelength)
{
    ProblemProperties problemProperties;
    problemProperties.rho = 0;
    problemProperties.kappa = 0;
    problemProperties.U = 0;

    SimulationOptions simOptions;
    simOptions.outputFilename = std::format("dispersion_test_{:.5e}.h5", wavelength);
	simOptions.videoFilename = std::format("dispersion_test_{:.5e}", wavelength);
    simOptions.createVideo = true;

    int frames = 250;
    double omega = 1;
    double t0 = 0;
    double finalTime = 3e-2; // 1 ms

    double H0 = 15e-7; // 15 nm
    double g = 3 * 2.6e-24 / std::pow(H0, 4); //
    double L0 = wavelength / (2.0 * PI_d); // 6mm

    double _t0 = std::sqrt(L0 / g);

    problemProperties.depth = H0 / L0;
    double h = 0.001 * problemProperties.depth;

	simOptions.attributes["wavelength"] = wavelength;
	simOptions.attributes["H0"] = H0;
	simOptions.attributes["g"] = g;

    problemProperties.initial_amplitude = h;
    problemProperties.y_min = -h - 0.0001 * problemProperties.depth; // -0.5 * H0
    problemProperties.y_max = h + 0.005 * problemProperties.depth; // 0.5 * H0
    printf("Simulating with depth (h_0) %.10e, h %.10e, omega %f, t0 %.10e, L0 %.10e\n", problemProperties.depth, h, omega, _t0, L0);
    printf("g %.10e, H0 %.10e, L0 %.10e\n", g, H0, L0);

    const int N = 128;//512;

    const double stepSize = 0.04;
    const int steps = (finalTime / _t0) / stepSize;
    const int loggingSteps = steps / frames;

    printf("Simulating %i steps representing %.2e s", steps, steps * stepSize * _t0);


    std::vector<std::complex<double>> Z0(N);
    std::vector<double> PhiVect(N);

    std::vector<double> X0(N, 0);
    std::vector<double> Y0(N, 0);


    RK4Options rk45_options;
    //rk45_options.atol = 1e-14;
    //rk45_options.rtol = 1e-10;
    //rk45_options.h_max = 5;
    //rk45_options.h_min = 1e-10; // smallest timestep
    rk45_options.initial_timestep = stepSize;

    // load data from Python H5 fileC:\Users\emore\Documents\repos\superfluid-calculations\simulations\data\sine_wave_128.h5
    // auto file_init = "C:\\Users\\emore\\Documents\\repos\\superfluid-calculations\\simulations\\data\\sine_wave_128.h5";
    // loadStateFile(file_init, Z0, Phi, N, h);

    // plot the data to verify that we have loaded the correct file
    double j;
    for (size_t n = 0; n < N; n++) {
        j = 2.0 * PI_d * n / (1.0 * N);
        X0[n] = X(j, h, omega, t0);
        Y0[n] = Y(j, h, omega, t0);
        PhiVect[n] = Phi(j, h, omega, t0, problemProperties.rho);

		Z0[n] = std::complex<double>(X0[n], Y0[n]);
    }

    /*plt::figure();
    plt::title("Initial Condition");
    plt::xlabel("x (0-2pi)");
    plt::ylabel("y (a.u.)");
    plt::plot(X0, Y0);
	plt::plot(X0, PhiVect);*/

    //plt::show();

    ParticleData particleData(Z0, PhiVect);
	DeviceParticleData deviceData;

    HeliumBoundaryProblem<N> boundaryProblem(problemProperties);
    return runSimulation<N, AutonomousRungeKuttaStepper<std_complex, 2*N>, RK4Options>(boundaryProblem, steps, problemProperties, particleData, rk45_options, simOptions, loggingSteps, false, false, _t0);
}

int modeSum() {
    ProblemProperties problemProperties;
    problemProperties.rho = 0;
    problemProperties.kappa = 0;
    problemProperties.U = 0;

    SimulationOptions simOptions;

    int frames = 30;
    double omega = 1;
    double t0 = 0;
    double finalTime = 15e-3; // 15 ms

    double H0 = 15e-9; // 15 nm
    double g = 3 * 2.6e-24 / std::pow(H0, 4); //
    double L0 = 500e-6 / (2.0 * PI_d); // 6mm

    double _t0 = std::sqrt(L0 / g);

    problemProperties.depth = H0 / L0;
    double h = 0.001 * problemProperties.depth;

    problemProperties.initial_amplitude = h;
    problemProperties.y_min = -h - 0.0001 * problemProperties.depth; // -0.5 * H0
    problemProperties.y_max = h + 0.005 * problemProperties.depth; // 0.5 * H0
    printf("Simulating with depth (h_0) %.10e, h %.10e, omega %f, t0 %.10e, L0 %.10e\n", problemProperties.depth, h, omega, _t0, L0);
    printf("g %.10e, H0 %.10e, L0 %.10e\n", g, H0, L0);


    const int N = 512;//512;
    
	  const double stepSize = 0.1;

    const int steps = (finalTime / _t0) / stepSize;
    const int loggingSteps = steps / frames;

    printf("Simulating %i steps representing %.2e s", steps, steps * stepSize * _t0);


    std::vector<std::complex<double>> Z0(N);
    std::vector<double> Phi(N);

    std::vector<double> X0(N, 0);
    std::vector<double> Y0(N, 0);

    RK45_Options rk45_options;
    rk45_options.atol = 1e-15;
    rk45_options.rtol = 1e-10;
    rk45_options.h_max = 2;
    //rk45_options.h_min = 1e-8 / _t0; // smallest timestep

    rk45_options.initial_timestep = stepSize;

    std::vector<double> Phireal(N, 0);
    // load data from Python H5 file
    auto file_init = "C:\\Users\\emore\\Documents\\repos\\superfluid-calculations\\simulations\\data\\mode_sum.h5";
    loadStateFile(file_init, Z0, Phi, N, h);

    // plot the data to verify that we have loaded the correct file
    for (size_t n = 0; n < N; n++) {
        X0[n] = Z0[n].real();
        Y0[n] = Z0[n].imag();
    }

    plt::figure();
    plt::title("Initial Condition");
    plt::xlabel("x (0-2pi)");
    plt::ylabel("y (a.u.)");
    plt::plot(X0, Y0);
    plt::show();

    ParticleData particleData(Z0, Phi);


    return runSimulationHelium<N, RK45_std_complex<2 * N>, RK45_Options>(steps, problemProperties, particleData, rk45_options, simOptions, loggingSteps, true, true, _t0);
}

int main() 
{
    double step = 1e-6;
    double start = 10e-6;
    double steps = 100;
    for(int i = 0; i < steps; i++) {
        double wavelength = start + i * step;
        printf("Running dispersion test for wavelength %.10e\n", wavelength);
        dispersionTest(wavelength);
	}
}