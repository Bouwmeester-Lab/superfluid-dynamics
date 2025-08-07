#pragma once
#include <CuDenseSolvers/Operators/LinearOperator.h>
#include <CuDenseSolvers/Operators/MatrixOperator.cuh>
#include "MatrixOperations.cuh"
#include "createM.cuh"

struct FiniteDepthFunctor {
	FiniteDepthFunctor(const std_complex* Z, const std_complex* Zp, const std_complex* Zpp, double h)
		: Z(Z), Zp(Zp), Zpp(Zpp), h(h) {
	}
	const std_complex* Z;
	const std_complex* Zp;
	const std_complex* Zpp;
	double h;

	__device__ inline double operator()(size_t i, size_t j) const {
		return FiniteDepthMij(i, j, Z, Zp, Zpp, h);
	}
};

template <size_t N>
class BoundaryMatrixOperator : public DoubleMatrixOperator<N>
{
public:
	virtual void createMatrix(std_complex* Z, std_complex* Zp, std_complex* Zpp, double h) = 0;
	virtual int size() const override;
	virtual void apply(const double* x, double* y, cudaStream_t stream = cudaStreamPerThread) const override;
};

template<size_t N>
int BoundaryMatrixOperator<N>::size() const
{
	return N;
}

template<size_t N>
void BoundaryMatrixOperator<N>::apply(const double* x, double* y, cudaStream_t stream) const
{
	DoubleMatrixOperator<N>::apply(x, y, stream);
}


template <size_t N>
class FiniteDepthOperator : public BoundaryMatrixOperator<N>
{
public:
	FiniteDepthOperator();


	virtual void createMatrix(std_complex* Z, std_complex* Zp, std_complex* Zpp, double h) override;
private:
	const std_complex* Z;   // Pointer to the Z array
	const std_complex* Zp;  // Pointer to the Z' array
	const std_complex* Zpp; // Pointer to the Z'' array
	double h;              // Finite depth parameter
};

template<size_t N>
FiniteDepthOperator<N>::FiniteDepthOperator() : BoundaryMatrixOperator<N>()
{

}

template<size_t N>
void FiniteDepthOperator<N>::createMatrix(std_complex* Z, std_complex* Zp, std_complex* Zpp, double h)
{
	const int threads = 256;
	const int blocks = (N + threads - 1) / threads;
	createFiniteDepthMKernel << <blocks, threads >> > (this->devData, Z, Zp, Zpp, h, N);
}
