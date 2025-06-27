#pragma once

#include "AutonomousProblem.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

template <typename T, int N>
__global__ void computePerturbedState(T* state, double delta, int i)
{
	state[i] = state[i] + delta;
}

template <int N>
__global__ void computePerturbedState(std_complex* state, double delta, int i)
{
	state[i] += delta;
}

template <int N>
__global__ void computePerturbedState(std_complex* state, std_complex delta, int i) 
{
	state[i] += delta;
}

template <int N>
__global__ void copyRealPartsOnly(std_complex* dst, const std_complex* src) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		dst[i] = std_complex(src[i].real(), dst[i].imag());
	}
}

template <int N>
__global__ void copyImaginaryPartsOnly(std_complex* dst, const std_complex* src) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		dst[i] = std_complex(dst[i].real(), src[i].imag());
	}
}


/// <summary>
/// Class for computing the Jacobian matrix of an autonomous system.
/// </summary>
/// 
template <typename T, int N>
class JacobianBase
{
public:
	JacobianBase(AutonomousProblem<T, N>& autonomousProblem);
	void setDelta(double delta) { this->delta = delta; }
	~JacobianBase();
	virtual void calculateJacobian(T* equilibriumState); ///< Sets the equilibrium state for which the Jacobian will be computed.
	T* jacobianMatrix; ///< Pointer to the Jacobian matrix array.
protected:
	AutonomousProblem<T, N>& autonomousProblem; ///< Reference to the autonomous problem for which the Jacobian is computed.
	double delta = 1e-6; ///< Small perturbation value used for jacobian calculation.
	T* perturbedState; ///< Pointer to the perturbed state array.
};

template <typename T, int N>
class Jacobian : public JacobianBase<T, N>
{

};

template <int N>
class Jacobian<std_complex, N> : public JacobianBase<std_complex, N>
{
public:
	using JacobianBase<std_complex, N>::JacobianBase;
	virtual void calculateJacobian(std_complex* equilibriumState) override;
};

template <typename T, int N>
JacobianBase<T, N>::JacobianBase(AutonomousProblem<T, N>& autonomousProblem) : autonomousProblem(autonomousProblem)
{
	cudaMalloc(&perturbedState, N * sizeof(T));
	cudaMalloc(&jacobianMatrix, N * N * sizeof(T));
}

template <typename T, int N>
JacobianBase<T, N>::~JacobianBase()
{
	cudaFree(perturbedState);
	cudaFree(jacobianMatrix);
}

template<typename T, int N>
void JacobianBase<T, N>::calculateJacobian(T* equilibriumState)
{
	std::vector<T> perturbed_host(N);
	cudaMemcpy(perturbedState, equilibriumState, N * sizeof(T), cudaMemcpyDeviceToDevice);
	cudaDeviceSynchronize();
	cudaMemcpy(perturbed_host.data(), equilibriumState, N * sizeof(T), cudaMemcpyDeviceToHost);
	for (int i = 0; i < N; i++) {
		computePerturbedState<T, N> << <1, 1 >> > (perturbedState, delta, i);
		cudaDeviceSynchronize();
		cudaMemcpy(perturbed_host.data(), perturbedState, N * sizeof(T), cudaMemcpyDeviceToHost);

		// Run the autonomous problem with the perturbed state
		autonomousProblem.run(perturbedState);
		// Compute the Jacobian entry for this perturbation
		cudaMemcpy(&jacobianMatrix[i * N], autonomousProblem.devTimeEvolutionRhs, N * sizeof(T), cudaMemcpyDeviceToDevice);
	}
}

template<int N>
void Jacobian<std_complex, N>::calculateJacobian(std_complex* equilibriumState) 
{
	const int threads = 256;
	const int blocks = (N + threads - 1) / threads;

	std::vector<std_complex> perturbed_host(N);
	cudaMemcpy(this->perturbedState, equilibriumState, N * sizeof(std_complex), cudaMemcpyDeviceToDevice);
	cudaDeviceSynchronize();
	cudaMemcpy(perturbed_host.data(), equilibriumState, N * sizeof(std_complex), cudaMemcpyDeviceToHost);
	for (int i = 0; i < N; i++) {
		// let's perturbe the real part first
		computePerturbedState<N> << <1, 1 >> > (this->perturbedState, std_complex(this->delta, 0.0), i);
		this->autonomousProblem.run(this->perturbedState);

		copyRealPartsOnly<N> << <blocks, threads >> > (&this->jacobianMatrix[i * N], this->autonomousProblem.devTimeEvolutionRhs);

		cudaMemcpy(this->perturbedState, equilibriumState, N * sizeof(std_complex), cudaMemcpyDeviceToDevice);
		computePerturbedState<N> << <1, 1 >> > (this->perturbedState, std_complex(0.0, this->delta), i);
		this->autonomousProblem.run(this->perturbedState);

		copyImaginaryPartsOnly<N> << <blocks, threads >> > (&this->jacobianMatrix[i * N], this->autonomousProblem.devTimeEvolutionRhs);
		
		cudaDeviceSynchronize();
		cudaMemcpy(perturbed_host.data(), this->autonomousProblem.devTimeEvolutionRhs, N * sizeof(std_complex), cudaMemcpyDeviceToHost);
		cudaMemcpy(this->perturbedState, equilibriumState, N * sizeof(std_complex), cudaMemcpyDeviceToDevice);
	}
}