#pragma once
#include <CuDenseSolvers/Operators/LinearOperator.h>
#include <gtest/gtest.h>


template <size_t N>
class FiniteDepthOperator : public LinearOperator<double>
{
public:
	FiniteDepthOperator();
	virtual ~FiniteDepthOperator() override;
	virtual void apply(const double* x, double* y, cudaStream_t stream = cudaStreamPerThread) const override;
	virtual int size() const override;
private:

};

template<size_t N>
FiniteDepthOperator<N>::FiniteDepthOperator()
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

}