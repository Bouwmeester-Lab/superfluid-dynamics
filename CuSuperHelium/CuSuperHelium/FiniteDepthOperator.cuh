#pragma once
#include <CuDenseSolvers/Operators/LinearOperator.h>
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
class FiniteDepthOperator : public LinearOperator<double>
{
public:
	FiniteDepthOperator(const std_complex* Z, const std_complex* Zp, const std_complex* Zpp, double h);
	virtual ~FiniteDepthOperator() override;
	virtual void apply(const double* x, double* y, cudaStream_t stream = cudaStreamPerThread) const override;
	virtual int size() const override;
private:
	const std_complex* Z;   // Pointer to the Z array
	const std_complex* Zp;  // Pointer to the Z' array
	const std_complex* Zpp; // Pointer to the Z'' array
	double h;              // Finite depth parameter
};

template<size_t N>
FiniteDepthOperator<N>::FiniteDepthOperator(const std_complex* Z, const std_complex* Zp, const std_complex* Zpp, double h) : LinearOperator<double>(), Z(Z), Zp(Zp), Zpp(Zpp), h(h)
{
}

template<size_t N>
FiniteDepthOperator<N>::~FiniteDepthOperator()
{
	LinearOperator<double>::~LinearOperator();
}

template<size_t N>
void FiniteDepthOperator<N>::apply(const double* x, double* y, cudaStream_t stream) const
{
	FiniteDepthFunctor functor(Z, Zp, Zpp, h);
	int blockSize = 256;
	int gridSize = (N + blockSize - 1) / blockSize;
	int sharedMemSize = blockSize * sizeof(double);

	matVectMultiply << <gridSize, blockSize, sharedMemSize, stream >> > (x, y, N, functor);
	if (cudaGetLastError() != cudaSuccess) {
		fprintf(stderr, "Error in FiniteDepthOperator::apply: %s\n", cudaGetErrorString(cudaGetLastError()));
	}
}

template<size_t N>
int FiniteDepthOperator<N>::size() const
{
	return N;
}
