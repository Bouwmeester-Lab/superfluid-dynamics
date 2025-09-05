/**
 * Header file used in conjunction with the CUDA complex structures from "cuComplex.h" in order to overload the
 *  +, -, *, / arithmetic operators for the cuComplex structures. This way, two cuComplex structures of the same type,
 *  or one cuComplex and one standard data type, can be used in simple arithmetic without directly calling the
 *  arithmetic functions inside cuComplex.h.
 *
 *
 *  @author Travis W. Thompson
 *  @date 11/28/2017
 *
 *  @file cuComplex_Op.h
 *  @version 1.0
 *	https://github.com/TravisWThompson1/cuComplex_Operators
 */
#pragma once
#include <cuComplex.h>

template <typename T>
T CastTo(double d) {
    if constexpr (std::same_as<T, cuDoubleComplex>) {
        return make_cuDoubleComplex(d, 0.0);
    }
    else if constexpr (std::constructible_from<T, double>) {
        return T(d);
    }
    else {
        static_assert(sizeof(T) == 0, "CastTo<T>: Cannot convert from double");
    }
}

template <typename T>
double CastFrom(T d) 
{
    if constexpr (std::same_as<T, std_complex>) {
        return ((std_complex)d).real();
    }
    else if constexpr (std::same_as<T, cuDoubleComplex>) {
        return cuCreal(d);
    }
    else if constexpr (std::constructible_from<double, T>) {
        return (double)d;
	}
    else {
        static_assert(true, "Case not supported");
    }
}

 /**
  * Complex float overloaded addition operator (cuFloatComplex + cuFloatComplex).
  * @param a cuFloatComplex data type.
  * @param b cuFloatComplex data type.
  * @return Returns the cuFloatComplex sum of a + b.
  */
template <class T>
__device__ __host__ cuFloatComplex operator+(cuFloatComplex a, cuFloatComplex b) { return cuCaddf(a, b); }


/**
 * Complex float overloaded addition operator (cuFloatComplex + <T>).
 * @tparam <T> Standard arithmetic data type (int, long, float, double).
 * @param a cuFloatComplex data type.
 * @param b <T> Standard arithmetic data type (int, long, float, double).
 * @return Returns the cuFloatComplex sum of a + b.
 */
template <class T>
__device__ __host__ cuFloatComplex operator+(cuFloatComplex a, T b) { return cuCaddf(a, make_cuFloatComplex(b, 0)); }


/**
 * Complex float overloaded addition operator (<T> + cuFloatComplex).
 * @tparam <T> Standard arithmetic data type (int, long, float, double).
 * @param a <T> Standard arithmetic data type (int, long, float, double).
 * @param b cuFloatComplex data type.
 * @return Returns the cuFloatComplex sum of a + b.
 */
template <class T>
__device__ __host__ cuFloatComplex operator+(T a, cuFloatComplex b) { return cuCaddf(make_cuFloatComplex(a, 0), b); }


/**
 * Complex float overloaded subtraction operator (cuFloatComplex - cuFloatComplex).
 * @param a cuFloatComplex data type.
 * @param b cuFloatComplex data type.
 * @return Returns the cuFloatComplex subtraction of a - b.
 */
__device__ __host__ cuFloatComplex operator-(cuFloatComplex a, cuFloatComplex b) { return cuCsubf(a, b); }


/**
 * Complex float overloaded subtraction operator (cuFloatComplex - <T>).
 * @tparam <T> Standard arithmetic data type (int, long, float, double).
 * @param a cuFloatComplex data type.
 * @param b <T> Standard arithmetic data type (int, long, float, double).
 * @return Returns the cuFloatComplex subtraction of a - b.
 */
template <class T>
__device__ __host__ cuFloatComplex operator-(cuFloatComplex a, T b) { return cuCsubf(a, make_cuFloatComplex(b, 0)); }


/**
 * Complex float overloaded subtraction operator (<T> - cuFloatComplex).
 * @tparam <T> Standard arithmetic data type (int, long, float, double).
 * @param a <T> Standard arithmetic data type (int, long, float, double).
 * @param b cuFloatComplex data type.
 * @return Returns the cuFloatComplex subtraction of a - b.
 */
template <class T>
__device__ __host__ cuFloatComplex operator-(T a, cuFloatComplex b) { return cuCsubf(make_cuFloatComplex(a, 0), b); }


/**
 * Complex float overloaded multiplication operator (cuFloatComplex * cuFloatComplex).
 * @param a cuFloatComplex data type.
 * @param b cuFloatComplex data type.
 * @return Returns the cuFloatComplex multiplication of a * b.
 */
__device__ __host__ cuFloatComplex operator*(cuFloatComplex a, cuFloatComplex b) { return cuCmulf(a, b); }


/**
 * Complex float overloaded multiplication operator (cuFloatComplex * <T>).
 * @tparam <T> Standard arithmetic data type (int, long, float, double).
 * @param a cuFloatComplex data type.
 * @param b <T> Standard arithmetic data type (int, long, float, double).
 * @return Returns the cuFloatComplex multiplication of a * b.
 */
template <class T>
__device__ __host__ cuFloatComplex operator*(cuFloatComplex a, T b) { return cuCmulf(a, make_cuFloatComplex(b, 0)); }


/**
 * Complex float overloaded multiplication operator (<T> * cuFloatComplex).
 * @tparam <T> Standard arithmetic data type (int, long, float, double).
 * @param a <T> Standard arithmetic data type (int, long, float, double).
 * @param b cuFloatComplex data type.
 * @return Returns the cuFloatComplex multiplication of a * b.
 */
template <class T>
__device__ __host__ cuFloatComplex operator*(T a, cuFloatComplex b) { return cuCmulf(make_cuFloatComplex(a, 0), b); }


/**
 * Complex float overloaded division operator (cuFloatComplex / cuFloatComplex).
 * @param a cuFloatComplex data type.
 * @param b cuFloatComplex data type.
 * @return Returns the cuFloatComplex division of a / b.
 */
__device__ __host__ cuFloatComplex operator/(cuFloatComplex a, cuFloatComplex b) { return cuCdivf(a, b); }


/**
 * Complex float overloaded division operator (cuFloatComplex / <T>).
 * @tparam <T> Standard arithmetic data type (int, long, float, double).
 * @param a cuFloatComplex data type.
 * @param b <T> Standard arithmetic data type (int, long, float, double).
 * @return Returns the cuFloatComplex division of a / b.
 */
template <class T>
__device__ __host__ cuFloatComplex operator/(cuFloatComplex a, T b) { return cuCdivf(a, make_cuFloatComplex(b, 0)); }


/**
 * Complex float overloaded division operator (<T> / cuFloatComplex).
 * @tparam <T> Standard arithmetic data type (int, long, float, double).
 * @param a <T> Standard arithmetic data type (int, long, float, double).
 * @param b cuFloatComplex data type.
 * @return Returns the cuFloatComplex division of a / b.
 */
template <class T>
__device__ __host__ cuFloatComplex operator/(T a, cuFloatComplex b) { return cuCdivf(make_cuFloatComplex(a, 0), b); }






/**
 * Complex double overloaded addition operator (cuDoubleComplex + cuDoubleComplex).
 * @param a cuDoubleComplex data type.
 * @param b cuDoubleComplex data type.
 * @return Returns the cuDoubleComplex sum of a + b.
 */
__device__ __host__ cuDoubleComplex operator+(cuDoubleComplex a, cuDoubleComplex b) { return cuCadd(a, b); }

__device__ __host__ cuDoubleComplex& operator+=(cuDoubleComplex& a, cuDoubleComplex b) { return a = cuCadd(a, b); }

__device__ __host__ cuDoubleComplex& operator-=(cuDoubleComplex& a, cuDoubleComplex b) { return a = cuCsub(a, b); }

/**
 * Complex double overloaded addition operator (cuDoubleComplex + <T>).
 * @tparam <T> Standard arithmetic data type (int, long, double, double).
 * @param a cuDoubleComplex data type.
 * @param b <T> Standard arithmetic data type (int, long, double, double).
 * @return Returns the cuDoubleComplex sum of a + b.
 */
template <class T>
__device__ __host__ cuDoubleComplex operator+(cuDoubleComplex a, T b) { return cuCadd(a, make_cuDoubleComplex(b, 0)); }


/**
 * Complex double overloaded addition operator (<T> + cuDoubleComplex).
 * @tparam <T> Standard arithmetic data type (int, long, double, double).
 * @param a <T> Standard arithmetic data type (int, long, double, double).
 * @param b cuDoubleComplex data type.
 * @return Returns the cuDoubleComplex sum of a + b.
 */
template <class T>
__device__ __host__ cuDoubleComplex operator+(T a, cuDoubleComplex b) { return cuCadd(make_cuDoubleComplex(a, 0), b); }


/**
 * Complex double overloaded subtraction operator (cuDoubleComplex - cuDoubleComplex).
 * @param a cuDoubleComplex data type.
 * @param b cuDoubleComplex data type.
 * @return Returns the cuDoubleComplex subtraction of a - b.
 */
__device__ __host__ cuDoubleComplex operator-(cuDoubleComplex a, cuDoubleComplex b) { return cuCsub(a, b); }


/**
 * Complex double overloaded subtraction operator (cuDoubleComplex - <T>).
 * @tparam <T> Standard arithmetic data type (int, long, double, double).
 * @param a cuDoubleComplex data type.
 * @param b <T> Standard arithmetic data type (int, long, double, double).
 * @return Returns the cuDoubleComplex subtraction of a - b.
 */
template <class T>
__device__ __host__ cuDoubleComplex operator-(cuDoubleComplex a, T b) { return cuCsub(a, make_cuDoubleComplex(b, 0)); }


/**
 * Complex double overloaded subtraction operator (<T> - cuDoubleComplex).
 * @tparam <T> Standard arithmetic data type (int, long, double, double).
 * @param a <T> Standard arithmetic data type (int, long, double, double).
 * @param b cuDoubleComplex data type.
 * @return Returns the cuDoubleComplex subtraction of a - b.
 */
template <class T>
__device__ __host__ cuDoubleComplex operator-(T a, cuDoubleComplex b) { return cuCsub(make_cuDoubleComplex(a, 0), b); }


/**
 * Complex double overloaded multiplication operator (cuDoubleComplex * cuDoubleComplex).
 * @param a cuDoubleComplex data type.
 * @param b cuDoubleComplex data type.
 * @return Returns the cuDoubleComplex multiplication of a * b.
 */
__device__ __host__ cuDoubleComplex operator*(cuDoubleComplex a, cuDoubleComplex b) { return cuCmul(a, b); }


/**
 * Complex double overloaded multiplication operator (cuDoubleComplex * <T>).
 * @tparam <T> Standard arithmetic data type (int, long, double, double).
 * @param a cuDoubleComplex data type.
 * @param b <T> Standard arithmetic data type (int, long, double, double).
 * @return Returns the cuDoubleComplex multiplication of a * b.
 */
template <class T>
__device__ __host__ cuDoubleComplex operator*(cuDoubleComplex a, T b) { return cuCmul(a, make_cuDoubleComplex(b, 0)); }


/**
 * Complex double overloaded multiplication operator (<T> * cuDoubleComplex).
 * @tparam <T> Standard arithmetic data type (int, long, double, double).
 * @param a <T> Standard arithmetic data type (int, long, double, double).
 * @param b cuDoubleComplex data type.
 * @return Returns the cuDoubleComplex multiplication of a * b.
 */
template <class T>
__device__ __host__ cuDoubleComplex operator*(T a, cuDoubleComplex b) { return cuCmul(make_cuDoubleComplex(a, 0), b); }


/**
 * Complex double overloaded division operator (cuDoubleComplex / cuDoubleComplex).
 * @param a cuDoubleComplex data type.
 * @param b cuDoubleComplex data type.
 * @return Returns the cuDoubleComplex division of a / b.
 */
__device__ __host__ cuDoubleComplex operator/(cuDoubleComplex a, cuDoubleComplex b) { return cuCdiv(a, b); }


/**
 * Complex double overloaded division operator (cuDoubleComplex / <T>).
 * @tparam <T> Standard arithmetic data type (int, long, double, double).
 * @param a cuDoubleComplex data type.
 * @param b <T> Standard arithmetic data type (int, long, double, double).
 * @return Returns the cuDoubleComplex division of a / b.
 */
template <class T>
__device__ __host__ cuDoubleComplex operator/(cuDoubleComplex a, T b) { return cuCdiv(a, make_cuDoubleComplex(b, 0)); }


/**
 * Complex double overloaded division operator (<T> / cuDoubleComplex).
 * @tparam <T> Standard arithmetic data type (int, long, double, double).
 * @param a <T> Standard arithmetic data type (int, long, double, double).
 * @param b cuDoubleComplex data type.
 * @return Returns the cuDoubleComplex division of a / b.
 */
template <class T>
__device__ __host__ cuDoubleComplex operator/(T a, cuDoubleComplex b) { return cuCdiv(make_cuDoubleComplex(a, 0), b); }













