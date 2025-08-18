#pragma once
#include "Solver.cuh"
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include <CuDenseSolvers/Solvers/Jacobi.cuh>

template <size_t N>
class JacobiSolver : public Solver
{
public:
	JacobiSolver();
	~JacobiSolver();
	virtual void solve(double* devA, double* devb, double* x) override;
private:
	cublasHandle_t handle;
	DoubleMatrixOperator<N> matrixOp;
	Jacobi<N> solver;
};


template <size_t N>
JacobiSolver<N>::JacobiSolver()
{
	cublasCreate(&handle);
	solver.setOperator(&matrixOp);
	cudaDeviceSynchronize();
}

template <size_t N>
JacobiSolver<N>::~JacobiSolver()
{
	cublasDestroy(handle);
}

template <size_t N>
void JacobiSolver<N>::solve(double* devA, double* devb, double* x)
{
	matrixOp.setDevPointer(devA);
	solver.solve(x, devb, 100, 1e-10);
}
