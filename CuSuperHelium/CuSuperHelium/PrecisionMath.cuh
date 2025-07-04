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
// Error–free subtraction: a – b = hi + lo                (Dekker/Kahan)
//-----------------------------------------------------------------------
__device__ __forceinline__ void twoDiff(double  a, double  b,
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
	twoDiff(z1.imag(), z2.imag(), d.imag.hi, d.li); // calculate the high and low parts of the imaginary part of the difference
	return d;
}

/// <summary>
/// Calculates the inverse of a double-double complex number.
/// </summary>
/// <param name="z"></param>
/// <returns></returns>
__device__ cuDoubleComplex dd_cinv(dd_complex z) 
{
	double r2h, r2l, i2h, i2l; // high and low parts of the real and imaginary parts of the inverse
	// real part
	twoProd(z.hr, z.hr, r2h, r2l); // calculates the high and low parts of the square of the high part of the real part.
	r2l = FMA(z.hr, z.lr, r2l) * 2.0 + z.lr * z.lr; // https://chatgpt.com/s/t_6867aecacafc8191ac4baf08cf79d2a1 on why doubling r2l is ok to do
	// imaginary part
	twoProd(z.hi, z.hi, i2h, i2l); // calculates the high and low parts of the square of the high part of the imaginary part.
	i2l = FMA(z.hi, z.li, i2l) * 2.0 + z.li * z.li; // https://chatgpt.com/s/t_6867aecacafc8191ac4baf08cf79d2a1 on why doubling i2l is ok to do

	// denominator


}
/// <summary>
/// Returns scale * (Zk - Zj) using the TwoDiff algorithm for precise subtraction.
/// </summary>
/// <param name="Zk"></param>
/// <param name="Zj"></param>
/// <param name="scale"></param>
/// <returns></returns>
__device__ inline std_complex fastPreciseInvSub(std_complex Zk, std_complex Zj, const double scale)
{
	double r2h, r2l, i2h, i2l; // real and imaginary parts of the high and low parts of the difference

	twoProd(hi.real(), scale, r2h, r2l); // calculate the high and low parts of the real part of the difference
	r2l = FMA()
}