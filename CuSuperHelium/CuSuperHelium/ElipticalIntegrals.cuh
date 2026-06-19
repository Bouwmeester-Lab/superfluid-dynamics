#pragma once
#include <math.h>
#include "cuda_runtime.h"


static constexpr double ELLIPTIC_PI = 3.141592653589793238462643383279502884;

__host__ __device__
inline double max3_abs(double a, double b, double c)
{
    a = fabs(a);
    b = fabs(b);
    c = fabs(c);
    return fmax(a, fmax(b, c));
}

// Carlson symmetric integral RF(x,y,z)
__host__ __device__
inline double carlson_rf(double x, double y, double z)
{
    const double ERRTOL = 0.0025;
    const double C1 = 1.0 / 24.0;
    const double C2 = 0.1;
    const double C3 = 3.0 / 44.0;
    const double C4 = 1.0 / 14.0;

    double xt = x;
    double yt = y;
    double zt = z;

    double ave, delx, dely, delz;

    for (int iter = 0; iter < 40; ++iter)
    {
        const double sx = sqrt(xt);
        const double sy = sqrt(yt);
        const double sz = sqrt(zt);

        const double lambda = sx * (sy + sz) + sy * sz;

        xt = 0.25 * (xt + lambda);
        yt = 0.25 * (yt + lambda);
        zt = 0.25 * (zt + lambda);

        ave = (xt + yt + zt) / 3.0;

        delx = (ave - xt) / ave;
        dely = (ave - yt) / ave;
        delz = (ave - zt) / ave;

        if (max3_abs(delx, dely, delz) <= ERRTOL)
            break;
    }

    const double e2 = delx * dely - delz * delz;
    const double e3 = delx * dely * delz;

    return (1.0 + (C1 * e2 - C2 - C3 * e3) * e2 + C4 * e3) / sqrt(ave);
}

// Carlson symmetric integral RD(x,y,z)
__host__ __device__
inline double carlson_rd(double x, double y, double z)
{
    const double ERRTOL = 0.0015;
    const double C1 = 3.0 / 14.0;
    const double C2 = 1.0 / 6.0;
    const double C3 = 9.0 / 22.0;
    const double C4 = 3.0 / 26.0;
    const double C5 = 0.25 * C3;
    const double C6 = 1.5 * C4;

    double xt = x;
    double yt = y;
    double zt = z;

    double sum = 0.0;
    double fac = 1.0;

    double ave, delx, dely, delz;

    for (int iter = 0; iter < 40; ++iter)
    {
        const double sx = sqrt(xt);
        const double sy = sqrt(yt);
        const double sz = sqrt(zt);

        const double lambda = sx * (sy + sz) + sy * sz;

        sum += fac / (sz * (zt + lambda));
        fac *= 0.25;

        xt = 0.25 * (xt + lambda);
        yt = 0.25 * (yt + lambda);
        zt = 0.25 * (zt + lambda);

        ave = 0.2 * (xt + yt + 3.0 * zt);

        delx = (ave - xt) / ave;
        dely = (ave - yt) / ave;
        delz = (ave - zt) / ave;

        if (max3_abs(delx, dely, delz) <= ERRTOL)
            break;
    }

    const double ea = delx * dely;
    const double eb = delz * delz;
    const double ec = ea - eb;
    const double ed = ea - 6.0 * eb;
    const double ee = ed + 2.0 * ec;

    return 3.0 * sum
        + fac
        * (
            1.0
            + ed * (-C1 + C5 * ed - C6 * delz * ee)
            + delz * (C2 * ee + delz * (-C3 * ec + delz * C4 * ea))
            )
        / (ave * sqrt(ave));
}

__host__ __device__
inline void elliptic_K_E_parameter_from_m_mc(
    double m,
    double mc,   // mc = 1 - m, preferably computed directly
    double* K,
    double* E
)
{
    if (m < 0.0 || mc < 0.0)
    {
        *K = NAN;
        *E = NAN;
        return;
    }

    if (mc == 0.0)
    {
        // m = 1: K diverges, E(1) = 1.
        *K = INFINITY;
        *E = 1.0;
        return;
    }

    if (m == 0.0)
    {
        *K = 0.5 * ELLIPTIC_PI;
        *E = 0.5 * ELLIPTIC_PI;
        return;
    }

    const double RF = carlson_rf(0.0, mc, 1.0);
    const double RD = carlson_rd(0.0, mc, 1.0);

    *K = RF;
    *E = RF - (m / 3.0) * RD;
}

__host__ __device__
inline double elliptic_dKdm_parameter(double m, double mc, double K, double E)
{
    // mc = 1 - m

    if (m < 0.0 || mc < 0.0)
        return NAN;

    if (mc == 0.0)
        return INFINITY;

    if (m < 1e-8)
    {
        // dK/dm = pi/8 + 9*pi*m/64 + 75*pi*m^2/512 + ...
        return ELLIPTIC_PI / 8.0
            + 9.0 * ELLIPTIC_PI * m / 64.0
            + 75.0 * ELLIPTIC_PI * m * m / 512.0;
    }

    return (E - mc * K) / (2.0 * m * mc);
}
