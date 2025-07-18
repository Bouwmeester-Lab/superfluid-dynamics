#pragma once

#include <cmath>
#include <vector>

namespace PeriodicFunctions {

    // Constants
    const double PI = 3.141592653589793;
    const double L = 2*PI;  // Half-period

    // Fourier coefficients for s = 2.5, centered at pi, over [0, 2pi]
    const double a0 = 0.127324;
    const std::vector<double> an = {
        -0.238634, 0.198205, -0.149202, 0.104368, -0.069272,
         0.044287, -0.027554, 0.016798, -0.010081, 0.005976,
        -0.003507, 0.002041, -0.00118, 0.000678, -0.000387,
         0.00022, -0.000125, 0.000071, -0.00004, 0.000022
    };
    const std::vector<double> bn = {
        0.0, -0.0, 0.0, -0.0, 0.0,
        -0.0, 0.0, -0.0, 0.0, -0.0,
         0.0, -0.0, -0.0, -0.0, 0.0,
        -0.0, -0.0, -0.0, 0.0, -0.0
    };

    // Evaluates the 2pi-periodic Fourier approximation of 1/cosh2(2.5(x - pi))
    double sech2_periodic(double x) {
        // Map x into [-pi, pi) for evaluation
        double x_mod = fmod(x + PI, 2 * PI);
        if (x_mod < 0) x_mod += 2 * PI;
        x_mod -= PI;


        // Evaluate Fourier series
        double sum = a0;
        for (size_t n = 0; n < an.size(); ++n) {
            double k = (n ) * x_mod;
            sum += an[n] * cos(k) + bn[n] * sin(k);
        }
        return sum;
    }
}