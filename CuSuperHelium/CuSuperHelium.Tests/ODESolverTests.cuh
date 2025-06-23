#pragma once
#include <gtest/gtest.h>
#include <complex>
#include <array>
#include "Derivatives.cuh"
#include "WaterVelocities.cuh"
#include "matplotlibcpp.h"
#include "utilities.cuh"
#include "AutonomousProblem.h"
#include "AutonomousRungeKuttaStepper.cuh"

__global__ void flip_x_y(cuDoubleComplex* data, cuDoubleComplex* out, int size) 
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) 
	{
		cuDoubleComplex temp = data[idx];
		out[idx].x = -temp.y;
		data[idx].y = temp.x;
	}
}

template <int N>
class OscillatoryProblem : public AutonomousProblem<cuDoubleComplex, N>
{
private:
	const int threads = 256;
	const int blocks = (N + threads - 1) / threads;
public:
	using AutonomousProblem<cuDoubleComplex, N>::AutonomousProblem;
	virtual void run(cuDoubleComplex* initialState) override
	{
		// Define the system of equations for the oscillatory problem
		flip_x_y << <blocks, threads >> > (initialState, this->devTimeEvolutionRhs, N); // implements: dx/dt=-y, dy/dt=x
	}
};

TEST(ODE_Solvers, RK4) 
{
	const int N = 16; // number of particles in the system
	// set the initial conditions
	cuDoubleComplex initialState[N];// = { {1.0, 0.0} };//, { 0.0, 1.0 }, { 1.0, 0.0 }, { 0.0, 1.0 }, { 1.0, 0.0 }, { 0.0, 1.0 }, { 1.0, 0.0 }, { 0.0, 1.0 },{1.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {0.0, 1.0} };

	for(int i = 0; i < N; ++i) 
	{
		initialState[i].x = cos(2*PI_d * i * 0.1);
		initialState[i].y = sin(2 * PI_d * i * 0.1);
	}


	cuDoubleComplex* devInitialState;
	cudaMalloc(&devInitialState, N * sizeof(cuDoubleComplex));
	cudaMemcpy(devInitialState, initialState, N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);


	// create a simple test problem
	OscillatoryProblem<N> problem;
	double dt = 0.01;
	int steps = 10;

	AutonomousRungeKuttaStepper<cuDoubleComplex, N> stepper(problem, dt);
	stepper.initialize(devInitialState);
	for(int i = 0; i < steps; ++i) 
	{
		stepper.runStep();
	}

	// Copy the results back to the host
	cuDoubleComplex* results = new cuDoubleComplex[N];
	cudaMemcpy(results, devInitialState, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

	// plot the results using matplotlibcpp
	matplotlibcpp::figure_size(800, 600);
	
	std::vector<double> x(N), y(N), x0(N), y0(N);

	for (int i = 0; i < N; ++i) 
	{
		x0[i] = initialState[i].x;
		y0[i] = initialState[i].y;

		x[i] = results[i].x;
		y[i] = results[i].y;
	}
	plt::plot(x0, {{"label", "initial x0"}});
	plt::plot(y0, { {"label", "initial y0"} });
	plt::plot(x, { {"label", "final x"} });
	plt::plot(y, { {"label", "final y"} });
	
	plt::legend();

	// clean up
	delete[] results;
}
