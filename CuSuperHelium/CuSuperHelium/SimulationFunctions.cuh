#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "ProblemProperties.hpp"
#include "WaterBoundaryIntegralCalculator.cuh"
#include "SimpleEuler.cuh"
#include "AutonomousRungeKuttaStepper.cuh"
#include "ValueLogger.h"
#include "matplotlibcpp.h"
#include "VideoMaking.h"
#include <highfive/H5File.hpp>
#include "RK45.cuh"
#include "SimulationOptions.h"
#include "SimulationRunner.cuh"

double X(double j, double h, double omega, double t) {
    return j - h * std::sin((j - omega * t));
}

double Y(double j, double h, double omega, double t) {
    return h * std::cos(j);// PeriodicFunctions::gaussian::gaussian_periodic(j);  // std::cos((j - omega * t));
}

double Phi(double j, double h, double omega, double t, double rho) {
    return h * (1 + rho) * omega * std::sin(j);// PeriodicFunctions::bimodal::bimodal(j);// std::sech2_periodic((j - omega * t));
}

template <size_t N>
int dispersionTest(double wavelength, double simulationTime, ProblemProperties problemProperties, int targetSteps = 15000);

template <size_t N>
int dispersionTest<N>(double wavelength, double simulationTime, ProblemProperties problemProperties, int targetSteps)
{

    problemProperties.rho = 0;
    problemProperties.U = 0;

    SimulationOptions simOptions;
    simOptions.outputFilename = std::format("dispersion_test_{:.5e}.h5", wavelength);
    simOptions.videoFilename = std::format("dispersion_test_{:.5e}", wavelength);
    simOptions.createVideo = false;
    simOptions.saveHDF5 = true;

    int frames = 4 * 1024;
    double omega = 1;
    double t0 = 0;
    double finalTime = simulationTime;

    const double H0 = problemProperties.depth; // m
    double g = 3 * 2.6e-24 / std::pow(H0, 4); //
    double L0 = wavelength / (2.0 * PI_d); // 6mm
    //double c;
    double _t0 = std::sqrt(L0 / g);

    double surfaceTensionFactor = 150 * L0 * L0 * L0 / (_t0 * _t0);
    problemProperties.kappa = problemProperties.kappa / surfaceTensionFactor; // non-dimensionalize kappa divide by 150 kg/m3 the density of helium to fully adimensionalize

    problemProperties.depth = 2.0 * CUDART_PI * H0 / wavelength;
    double test = problemProperties.depth / 3.0;

    double h = 0.0001 * problemProperties.depth;

    simOptions.attributes["wavelength"] = wavelength;
    simOptions.attributes["H0"] = H0;
    simOptions.attributes["g"] = g;

    problemProperties.initial_amplitude = h;
    problemProperties.y_min = -h - 0.0001 * problemProperties.depth; // -0.5 * H0
    problemProperties.y_max = h + 0.005 * problemProperties.depth; // 0.5 * H0
    printf("Simulating with depth (h_0) %.10e, h %.10e, omega %f, t0 %.10e, L0 %.10e\n", problemProperties.depth, h, omega, _t0, L0);
    printf("g %.10e, H0 %.10e, L0 %.10e\n", g, H0, L0);
    printf("1 a.u. of Force/Length is %.5e N/m\n", surfaceTensionFactor);

    const double stepSize = (finalTime / _t0) / targetSteps;
    const int steps = (finalTime / _t0) / stepSize;
    const int loggingSteps = steps / frames;

    printf("Simulating %i steps representing %.2e s with step size %.3e (a.u. of time)", steps, steps * stepSize * _t0, stepSize);


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
        PhiVect[n] = 0.0 * Phi(j, h, omega, t0, problemProperties.rho);

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
    auto res =  runSimulation<N, AutonomousRungeKuttaStepper<std_complex, 2 * N>, RK4Options>(boundaryProblem, steps, problemProperties, particleData, rk45_options, simOptions, loggingSteps, false, false, _t0);
    // _CrtDumpMemoryLeaks();
	return res;
}