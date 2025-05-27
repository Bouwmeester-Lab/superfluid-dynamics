#pragma once
#include "utilities.cuh"

__global__ void complex_pointwise_mul(
    const cufftDoubleComplex* a,
    const cufftDoubleComplex* b,
    cufftDoubleComplex* result,
    const int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        cufftDoubleComplex x = a[i];
        cufftDoubleComplex y = b[i];
        result[i].x = x.x * y.x - x.y * y.y;  // real
        result[i].y = x.x * y.y + x.y * y.x;  // imag
    }
}

__global__ void vector_subtract_complex_real(const cufftDoubleComplex* a, const double* b, cufftDoubleComplex* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i].x = a[i].x - b[i]; // modifies only the real part.
    }
}

__global__ void vector_scalar_add_complex_real(const cufftDoubleComplex* a, const double b, cufftDoubleComplex* out, int n, int start)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[start + i].x = a[start + i].x + b; // modifies only the real part.
    }
}
/// <summary>
/// Calculates the cotangent using cot(z) = cos(z)/sin(z).
/// </summary>
/// <param name="a"></param>
/// <param name="out"></param>
/// <param name="n"></param>
__global__ void cotangent_complex(const cufftDoubleComplex* a, cufftDoubleComplex* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        cufftDoubleComplex cs;
        cufftDoubleComplex cc;

        cos(a[i], cc);
        sin(a[i], cs);

        auto z = cuCdiv(cc, cs);

        out[i].x = z.x;
        out[i].y = z.y;
    }
}

__global__ void real_to_complex(const double* x, cuDoubleComplex* x_c, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N)
        x_c[idx] = make_cuDoubleComplex(x[idx], 0.0);
}

__device__ static cufftDoubleComplex cotangent_complex(cufftDoubleComplex a)
{
    
        cufftDoubleComplex cs;
        cufftDoubleComplex cc;

        cos(a, cc);
        sin(a, cs);

        return cuCdiv(cc, cs);   
}

__device__ static void cos(cufftDoubleComplex z, cufftDoubleComplex& out)
{
    out.x = cos(z.x) * cosh(z.y);
    out.y = -sin(z.x) * sinh(z.y);
}

__device__ static void sin(cufftDoubleComplex z, cufftDoubleComplex& zout) {
    zout.x = sinh(z.x) * cos(z.y);
    zout.y = cosh(z.x) * sin(z.y);
}

__device__ cufftDoubleComplex fromReal(double a)
{
    cufftDoubleComplex out;
    out.x = a;
    out.y = 0.0;
    return out;
}

cufftDoubleComplex cMulScalar(double a, cufftDoubleComplex z)
{
    cufftDoubleComplex out(z);

    out.x = a * out.x;
    out.y = a * out.y;

    return out;
}

void checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(result) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void checkCusolver(cusolverStatus_t status) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "cuSolver Error" << std::endl;
        exit(EXIT_FAILURE);
    }
}
