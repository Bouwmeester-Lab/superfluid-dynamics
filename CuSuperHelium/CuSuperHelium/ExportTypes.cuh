#pragma once
#include <complex>
#include <type_traits>

extern "C"
{
	struct __declspec(align(16)) c_double { double re; double im; };
}


static_assert(std::is_trivially_copyable_v<c_double>);
static_assert(std::is_trivially_copyable_v<std::complex<double>>);

static_assert(sizeof(c_double) == sizeof(std::complex<double>),
	"Size mismatch: cannot bit_cast/memcpy elementwise");
static_assert(alignof(std_complex) == 16, "Your premise must hold");
static_assert(alignof(c_double) == 16, "Over-alignment failed");
