#pragma once
#include <CuDenseSolvers/Operators/LinearOperator.h>
#include <gtest/gtest.h>



class FiniteDepthOperator : public LinearOperator<double>
{
public:
	FiniteDepthOperator();
	~FiniteDepthOperator();

private:

};

FiniteDepthOperator::FiniteDepthOperator()
{
}

FiniteDepthOperator::~FiniteDepthOperator()
{
}