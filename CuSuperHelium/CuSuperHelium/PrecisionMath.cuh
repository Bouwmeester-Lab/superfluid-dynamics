#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cufft.h>
#include "cuDoubleComplexOperators.cuh"
#include "constants.cuh"

#define FMA(x, y, z) __fma_rn(x, y, z) // defines fused multiply-add as a macro for better readability

struct doubledouble
{
	double hi, lo; // high and low parts of the double-double number
};

struct dd_complex
{
	doubledouble real;
	doubledouble imag;
};


//-----------------------------------------------------------------------
// One Newton step for the real reciprocal of a double–double           //
//    r ~ 1/dhi  ->  r += r * (1 - d*r)                                 //
//-----------------------------------------------------------------------
__device__ __forceinline__ double dd_inv_one_step(double dhi, double dlo)
{
	double r = 1.0 / dhi;                             // 1st approx
	double err = FMA(-(dhi + dlo), r, 1.0);            // 1 - d*r   (exact)
	return FMA(r, err, r);                             // refine
}

//-----------------------------------------------------------------------
// Error–free subtraction: a – b = hi + lo                (Dekker/Kahan)
//-----------------------------------------------------------------------
__device__ __host__ __forceinline__ void twoDiff(double  a, double  b,
	double& hi, double& lo)
{
	hi = a - b;
	double t = a - hi;                 // "virtual" b
	lo = (a - (hi + t)) + (t - b);      // exact rounding error
}

__device__ __host__ doubledouble operator+(doubledouble a, doubledouble b)
{
	doubledouble z;
	twoDiff(a.hi, -b.hi, z.hi, z.lo);
	z.lo += a.lo + b.lo; // combine the low parts

	// renormalize
	double t = z.hi + z.lo; // add the high and low parts
	z.lo = z.lo - (t - z.hi); // calculate the new low part
	z.hi = t; // set the new high part
	return z;
}



__device__ inline double operator/(double top, doubledouble bottom) 
{
	auto reciprocal = dd_inv_one_step(bottom.hi, bottom.lo); // calculate the reciprocal of the double-double number
	return top * reciprocal; // multiply the top by the reciprocal to get the result
}

__device__ __forceinline__ void twoProd(double  a, double  b,
	double& hi, double& lo)
{
	hi = a * b;
	lo = __fma_rn(a, b, -hi);  // exact residual: (a * b) - hi
}

__device__ __forceinline__ dd_complex c_twoDiff(std_complex z1, std_complex z2)
{
	dd_complex d;
	twoDiff(z1.real(), z2.real(), d.real.hi, d.real.lo); // calculate the high and low parts of the real part of the difference
	twoDiff(z1.imag(), z2.imag(), d.imag.hi, d.imag.lo); // calculate the high and low parts of the imaginary part of the difference
	return d;
}

__device__ inline dd_complex operator-(std_complex a, std_complex b)
{
	return c_twoDiff(a, b); // returns the double-double complex number representing the difference between the two complex numbers
}

/// <summary>
/// Calculates the inverse of a double-double complex number.
/// </summary>
/// <param name="z"></param>
/// <returns></returns>
__device__ cuDoubleComplex dd_cinv(dd_complex z) 
{
	double r2h, r2l, i2h, i2l; // high and low parts of the real and imaginary parts of the inverse
	doubledouble r2, i2; // real and imaginary parts of the double-double number
	// real part
	twoProd(z.real.hi, z.real.hi, r2.hi, r2.lo); // calculates the high and low parts of the square of the high part of the real part.
	r2l = FMA(z.real.hi, z.real.lo, r2.lo) * 2.0 + z.real.lo * z.real.lo; // https://chatgpt.com/s/t_6867aecacafc8191ac4baf08cf79d2a1 on why doubling r2l is ok to do
	// imaginary part
	twoProd(z.imag.hi, z.imag.hi, i2.hi, i2.lo); // calculates the high and low parts of the square of the high part of the imaginary part.
	i2l = FMA(z.imag.hi, z.imag.lo, i2.lo) * 2.0 + z.imag.lo * z.imag.lo; // https://chatgpt.com/s/t_6867aecacafc8191ac4baf08cf79d2a1 on why doubling i2l is ok to do

	// denominator (x^2 + y^2) using the double-double algorithm (with high and low parts)
	auto denom = r2 + i2; // adds the real and imaginary parts of the double-double number

	// calculate the reciprocal of the double-double number 1/denom
	double invD = 1.0 / denom;

	// calculate the numerator:
	return make_cuDoubleComplex(z.real.hi * invD, -z.imag.hi * invD); // returns the high parts of the real and imaginary parts multiplied by the inverse of the denominator
}

__device__ inline cuDoubleComplex operator/(cuDoubleComplex top, dd_complex bottom)
{
	auto inv = dd_cinv(bottom); // calculates the inverse of the double-double complex number
	return inv * top; // multiplies the top by the inverse to get the result
}

__device__ inline cuDoubleComplex operator/(double top, dd_complex bottom)
{
	auto inv = dd_cinv(bottom); // calculates the inverse of the double-double complex number
	return top * inv; // multiplies the top by the inverse to get the result
}

/// <summary>
/// Returns 1.0 / (Z1 - Z2) using the TwoDiff algorithm for precise subtraction.
/// </summary>
/// <param name="Zk"></param>
/// <param name="Zj"></param>
/// <returns></returns>
__device__ inline std_complex fastPreciseInvSub(std_complex Z1, std_complex Z2)
{
	dd_complex d = Z1 - Z2;
	cuDoubleComplex inv = dd_cinv(d); // calculates the inverse of the double-double complex number
	return std_complex(inv.x, inv.y); // returns the result as a standard complex number
}