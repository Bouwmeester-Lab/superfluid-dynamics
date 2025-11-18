
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
#include <stdfloat>
#include <format>
//#include "math.h"
//#include "complex.h"
#include "matplotlibcpp.h"
#include "SimulationRunner.cuh"
#include "SimulationFunctions.cuh"

namespace plt = matplotlibcpp;

#define j_complex std::complex<double>(0, 1)
cudaError_t setDevice();


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

    double H0 = 15e-7; // 15 nm
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
    ProblemProperties problemProperties;
    problemProperties.rho = 0;
    problemProperties.kappa = 3.5e-6; // in kg/s^2
    problemProperties.depth = 15e-8;

    double alpha = 2.6e-24;
    // c3 = np.sqrt(h0 * alpha * h0 * *-4)
    double c3 = std::sqrt(3 * alpha / std::pow(problemProperties.depth, 3));
    int periods = 5;
    //std::vector<double> wavelengths = { 6.283185307179586e-08, 6.495984809561782e-08, 6.723703834251217e-08, 6.967968409491701e-08, 7.230649758262222e-08, 7.513912326111669e-08, 7.820273558292189e-08, 8.152678922067472e-08, 8.514596911598505e-08, 8.910140533408705e-08, 9.344224302985025e-08, 9.822769482922264e-08, 1.035297579023909e-07, 1.0943686120613092e-07, 1.1605883688420891e-07, 1.2353381281912406e-07, 1.320379521073971e-07, 1.4179951276903346e-07, 1.5311964193967056e-07, 1.6640399443671964e-07, 1.8221237390820796e-07, 2.0133963967757788e-07, 2.2495354803482467e-07, 2.5484248099050066e-07, 2.938909256584e-07, 3.4707118839658667e-07, 4.2374970676327437e-07, 5.439175340543523e-07, 7.592182246175334e-07, 1.2566370614359173e-06 };
    //std::vector<double> simulation_time = { 2.067e-07, 2.137e-07, 2.211e-07, 2.292e-07, 2.378e-07, 2.471e-07, 2.572e-07, 2.681e-07, 2.800e-07, 2.931e-07, 3.073e-07, 3.231e-07, 3.405e-07, 3.599e-07, 3.817e-07, 4.063e-07, 4.343e-07, 4.664e-07, 5.036e-07, 5.473e-07, 5.993e-07, 6.622e-07, 7.399e-07, 8.382e-07, 9.666e-07, 1.142e-06, 1.394e-06, 1.789e-06, 2.497e-06, 4.133e-06 };

    double startWaveNumber = 1e2;
    double endWaveNumber = 70000;
    size_t steps = 20;

    double stepSize = (endWaveNumber - startWaveNumber) / (steps - 1);
    double waveNumber = startWaveNumber; // 1.2629161883e-07
    for (size_t i = 0; i < steps; ++i) {

        double wavelength = 2.0 * PI_d / waveNumber;
        double time = wavelength / c3 * periods;
        printf("Running dispersion test for wavelength %.10e\n", wavelength);
        dispersionTest<256>(wavelength, time, problemProperties, 10000);
        // next wave number
        waveNumber += stepSize;
    }
    //modeSum();
    return 0;
}