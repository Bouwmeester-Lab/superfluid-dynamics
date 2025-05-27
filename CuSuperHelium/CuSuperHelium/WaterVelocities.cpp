#include "WaterVelocities.hpp"

__global__ void createVelocityMatrices(cufftDoubleComplex* ZPhi, cufftDoubleComplex* ZPhiPrime, cufftDoubleComplex* Zpp, int N, cufftDoubleComplex* out1, cufftDoubleComplex* out2, bool lower)
{
    int k = blockIdx.y * blockDim.y + threadIdx.y; // row
    int j = blockIdx.x * blockDim.x + threadIdx.x; // col

    if (k < N && j < N) {
        int indx = k + j * N; // column major index
        if (k == j)
        {
            // we are in the diagonal:
            cuDoubleComplex val = cuCdiv(make_cuDoubleComplex(1.0, 0), ZPhiPrime[k]);
            out1[indx] = cuCmul(make_cuDoubleComplex(0, Coeff_Vel),  cuCdiv(Zpp[k], cuCmul(ZPhi[k], ZPhi[k]))); // no normalization by N since it would be cancelled out by the division.
            if (lower) 
            {
				out1[indx] = cuCadd(out1[indx], cMulScalar(0.5 * N, val)); // the *N is because the derivatives obtained by the FFT derivatives are not normalized by N so this will apply the normalization.
            }
            else 
            {
				out1[indx] = cuCsub(out1[indx], cMulScalar(0.5*N, val));
            }
            out2[k] = cMulScalar(-2.0, val);
        }
        else
        {
			out1[indx] = cuCmul(make_cuDoubleComplex(0, Coeff_Vel), cotangent_complex(cMulScalar(0.5, cuCsub(ZPhi[k], ZPhi[j])))); // no /N here since Z is not obtained by an FFT.
        }
    }
}

__global__ void calculateDiagonalVectorMultiplication(cufftDoubleComplex* diag, cufftDoubleComplex* vec, cufftDoubleComplex* out, int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		out[i] = cuCmul(diag[i], vec[i]);
	}
}

//  
//{  
//   // Create cuBLAS handle  
//   cublasHandle_t handle;  
//   cublasCreate(&handle);  
//
//   // Allocate memory for the result  
//   cufftDoubleComplex* result1;  
//   cudaMalloc(&result1, N * sizeof(cufftDoubleComplex));  
//
//   // Perform matrix-vector multiplication: result1 = V1 * a  
//   const cufftDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);  
//   const cufftDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);  
//   cublasZgemv(handle, CUBLAS_OP_N, N, N, &alpha, V1, N, a, 1, &beta, result1, 1);  
//
//   // Clean up  
//   cublasDestroy(handle);  
//   cudaFree(result1);  
//}

VelocityCalculator::VelocityCalculator()
{
    cublasCreate(&handle);
}

VelocityCalculator::~VelocityCalculator()
{
	if (handle) {
		cublasDestroy(handle);
	}
}

void VelocityCalculator::calculateVelocities(cufftDoubleComplex* a, cufftDoubleComplex* aprime, cufftDoubleComplex* V1, cufftDoubleComplex* V2, int N, cufftDoubleComplex* velocities)
{
	const int threads = 256;
	const int blocks = (N + threads - 1) / threads;

    // calculate v2*aprime
	calculateDiagonalVectorMultiplication << <blocks, threads >> > (V2, aprime, velocities, N);

	// calculate V1 * a + v2*aprime
	const cufftDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
	const cufftDoubleComplex beta = make_cuDoubleComplex(1.0, 0.0);
	cublasZgemv(handle, CUBLAS_OP_N, N, N, &alpha, V1, N, a, 1, &beta, velocities, 1);
}
