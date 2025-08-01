#pragma once

#include <cmath>
#include <vector>
#include <numbers>

namespace PeriodicFunctions 
{
    const double PI = std::numbers::pi;
    double PeriodicSeries(double x, const double a0, const std::vector<double>& an, const std::vector<double>& bn) {
        // Map x into [-pi, pi) for evaluation
        /*double x_mod = fmod(x + PI, 2 * PI);
        if (x_mod < 0) x_mod += 2 * PI;
        x_mod -= PI;*/
        double x_mod = x;

        // Evaluate Fourier series
        double sum = a0;
        for (size_t n = 1; n < an.size()+1; n++) {
            double k = (n)*x_mod;
            sum += an[n-1] * cos(k) + bn[n-1] * sin(k);
        }
        return sum;
    }
}


namespace PeriodicFunctions::sech2 {

    // Constants
   
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
		return PeriodicFunctions::PeriodicSeries(x, a0, an, bn);
    }
}

namespace PeriodicFunctions::gaussian 
{
    const double a0 = 1.5957691216057246e-01;
    const std::vector<double> an = {
        -2.9461611224265999e-01, 2.3175324220918495e-01, -1.5534884398657164e-01, 8.8736667743563202e-02, -4.3192773210551573e-02, 1.7915624235873052e-02, -6.3323612663850467e-03, 1.9072705611707299e-03,
-4.8952154409208630e-04, 1.0706418061086808e-04, -1.9953977032952183e-05, 3.1690392718593555e-06, -4.2888282845949171e-07, 4.9460963193639726e-08, -4.8607073229860768e-09,  4.0705037970490321e-10, -2.9048571613839486e-11, 1.7657268970665185e-12,
-9.2567920916448894e-14, 3.4960563188109505e-15	};

    const std::vector<double> bn = { 2.6223499752091563e-17, -3.7755273619645299e-17, 8.1379620271750278e-17, -8.2679262975899290e-17, 3.5775787773059993e-16, 2.4757768672610516e-17, -2.8265739808891720e-17, -2.6120284222304456e-16, -3.1532838906412898e-18,
-7.2950269496059298e-16, 9.8101847751245039e-16, -3.6324921102983220e-16, -1.1792591716610814e-15, -4.4030064198317788e-16, 1.8338937189851727e-15, -2.4992000830124649e-16, -5.3663433720978846e-17, 8.2821191805862088e-17, 1.3592529500408239e-16, -1.5644912748822963e-15 };

    double gaussian_periodic(double x) {
        // Evaluate Fourier series
        return PeriodicFunctions::PeriodicSeries(x, a0, an, bn);
	}
}

namespace PeriodicFunctions::bimodal {
    const double a0 = 1.5957691062629392e-01;
    const std::vector<double> an = { -3.0650464552593487e-09, -1.7317100891853296e-01, -3.0372115972901532e-09, -7.1615693357362259e-02, -2.9828983428199540e-09, 1.4774552742577363e-01, -2.9046711953897060e-09, -7.1789475086402932e-02, -2.8060306559965609e-09, -2.7503222644971376e-09, -2.6910728169895891e-09, 1.4494041842830990e-02, -2.5641238028081491e-09, -6.0224359434162744e-03, -2.4294154732096473e-09, 5.8937665589190228e-04, -2.2908220433559505e-09, 2.8773132315834389e-04, -2.1516893463953816e-09, -1.0706626341016840e-04 };

    const std::vector<double> bn = { 1.0511989972143337e-10, 2.3834944183054343e-01, 3.1263600914839784e-10, -2.2041043076185798e-01, 5.1219384396767367e-10, 4.8005433455860468e-02, 6.9916169059363772e-10, 5.2158105423964263e-02, 8.6981803244252374e-10, -4.3192772262388143e-02, 1.0215322886452160e-09, 1.0530540801238919e-02, 1.1528171619772920e-09, 1.9568084564639773e-03, 1.2632544912050875e-09, -1.8139207847607411e-03, 1.3533384629560178e-09, 3.9603263938379297e-04, 1.4242679357387493e-09, 1.4530607711567880e-09 };

    double bimodal(double x) {
        // Evaluate Fourier series
        return PeriodicFunctions::PeriodicSeries(x, PeriodicFunctions::bimodal::a0, PeriodicFunctions::bimodal::an, PeriodicFunctions::bimodal::bn);
    }
}

/// <summary>
/// Represents a shifted Gaussian function as a Fourier series (at 0.75 pi).
/// </summary>
namespace PeriodicFunctions::shiftedGaussian {
	const double a0 = 1.5957691185333889e-01;
    const std::vector<double> an = {
    -0.208325,
    -0.000000,
    0.109848,
    -0.088737,
    0.030542,
    -0.000000,
    -0.004478,
    0.001907,
    -0.000346,
    -0.000000,
    0.000014,
    -0.000003,
    0.000000,
    -0.000000,
    -0.000000,
    0.000000,
    -0.000000,
    -0.000000,
    -0.000000,
    -0.000000
    };

    const std::vector<double> bn = {
        0.208325,
        -0.231753,
        0.109848,
        0.000000,
        -0.030542,
        0.017916,
        -0.004478,
        0.000000,
        0.000346,
        -0.000107,
        0.000014,
        0.000000,
        -0.000000,
        0.000000,
        -0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000
    };
    double shiftedGaussian(double x) {
        // Evaluate Fourier series
        return PeriodicFunctions::PeriodicSeries(x, a0, an, bn);
	}
}