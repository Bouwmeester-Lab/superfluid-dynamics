#pragma once
#ifndef TIMESTEP_MANAGER_H
#define TIMESTEP_MANAGER_H


#include "ProblemProperties.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cufft.h"
#include "utilities.cuh"
#include "constants.cuh"
#include "Derivatives.cuh"
#include "createM.cuh"
#include "WaterVelocities.cuh"
#include "MatrixSolver.cuh"
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;
template<int N>
class TimeStepManager
{
public:
	TimeStepManager(ProblemProperties& problemProperties);
	~TimeStepManager();

	/// <summary>
	/// Initializes the time step manager by copying the host data to the device memory for the initial conditions. Use this only once when setting up the problem from the host side.
	/// </summary>
	/// <param name="Z0"></param>
	/// <param name="Phi0"></param>
	void initialize_device(cufftDoubleComplex* Z0, cufftDoubleComplex* Phi0, cufftDoubleComplex*& devZ, cufftDoubleComplex*& devPhi);
	void runTimeStep();
	cufftDoubleComplex* devVelocitiesLower; ///< Device pointer to the velocities array (lower fluid)
	cufftDoubleComplex* devVelocitiesUpper; ///< Device pointer to the velocities array (upper fluid)
	cufftDoubleComplex* devRhsPhi; ///< Device pointer to the right-hand side of the phi equation (derivative of Phi/dt)

	double* devPhiPrime; ///< Device pointer to the PhiPrime array (derivative of Phi)
private:
	cufftDoubleComplex* devZPhi; ///< Device pointer to the ZPhi array
	cufftDoubleComplex* devZPhiPrime; ///< Device pointer to the ZPhiPrime array
	
	cufftDoubleComplex* devZpp; ///< Device pointer to the Zpp array

	double* devM; ///< Device pointer to the matrix M (NxN, double precision)
	double* deva; ///< Device pointer to the solution vector a
	cufftDoubleComplex* devaComplex; ///< Device pointer to the solution vector a in complex form (for compatibility with velocity calculations)

	cufftDoubleComplex* devaprime; ///< Device pointer to the derivative of a

	cufftDoubleComplex* devV1; ///< Device pointer to the V1 matrix
	cufftDoubleComplex* devV2; ///< Device pointer to the V2 diagonal vector

	
	

	ProblemProperties& problemProperties; ///< Reference to the problem properties for configuration
	ZPhiDerivative<N> zPhiDerivative; ///< Derivative calculator for Z and Phi
	FftDerivative<N, 1> fftDerivative; ///< FFT derivative calculator for single batch
	MatrixSolver<N> matrixSolver; ///< Matrix solver for solving the vorticities.
	VelocityCalculator<N> velocityCalculator; ///< Velocity calculator for calculating the velocities based on the vorticities and matrices.
	const int threads = 256; ///< Number of threads per block for CUDA kernels
	const int blocks = (N + threads - 1) / threads; ///< Number of blocks for CUDA kernels, ensuring all elements are covered

	const dim3 matrix_threads;// (16, 16);     // 256 threads per block in 2D
	const dim3 matrix_blocks; // ((N + 15) / 16, (N + 15) / 16);


	void calculatePhiRhs(); ///< Helper function to calculate the right-hand side of the phi equation
};

template<int N>
TimeStepManager<N>::TimeStepManager(ProblemProperties& problemProperties) : problemProperties(problemProperties), zPhiDerivative(problemProperties), 
matrix_threads(16, 16), matrix_blocks((N + 15) / 16, (N + 15) / 16)
{
	cudaMalloc(&devZPhi,2 * N * sizeof(cufftDoubleComplex));
	cudaMalloc(&devZPhiPrime, 2 * N * sizeof(cufftDoubleComplex));
	cudaMalloc(&devPhiPrime, N * sizeof(double)); // Device pointer for the PhiPrime array (derivative of Phi)
	cudaMalloc(&devZpp, N * sizeof(cufftDoubleComplex));
	cudaMalloc(&devM, N * N * sizeof(double)); // Matrix M for solving the system
	cudaMalloc(&deva, N * sizeof(double));
	cudaMalloc(&devaComplex, N * sizeof(cufftDoubleComplex)); // Device pointer for the solution vector a in complex form
	cudaMalloc(&devaprime, N * sizeof(cufftDoubleComplex));
	cudaMalloc(&devV1, N * N * sizeof(cufftDoubleComplex));
	cudaMalloc(&devV2, N * sizeof(cufftDoubleComplex));
	cudaMalloc(&devVelocitiesLower, N * sizeof(cufftDoubleComplex));
	cudaMalloc(&devVelocitiesUpper, N * sizeof(cufftDoubleComplex));
	cudaMalloc(&devRhsPhi, N * sizeof(cufftDoubleComplex)); // Right-hand side of the phi equation (derivative of Phi)
	fftDerivative.initialize(); // Initialize the FFT derivative calculator
}

template<int N>
TimeStepManager<N>::~TimeStepManager()
{
	cudaFree(devZPhi);
	cudaFree(devZPhiPrime);
	cudaFree(devZpp);
	cudaFree(deva);
	cudaFree(devaprime);
	cudaFree(devV1);
	cudaFree(devV2);
	cudaFree(devVelocitiesLower);
	cudaFree(devVelocitiesUpper);
	cudaFree(devRhsPhi);
}

template<int N>
inline void TimeStepManager<N>::initialize_device(cufftDoubleComplex* Z0, cufftDoubleComplex* Phi0, cufftDoubleComplex*& devZ, cufftDoubleComplex*& devPhi)
{
	// Copy initial conditions to device memory
	cudaMemcpy(devZPhi, Z0, N * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
	cudaMemcpy(devZPhi + N, Phi0, N * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);

	devZ = this->devZPhi; // Set the device pointer for ZPhi
	devPhi = this->devZPhi + N; // Set the device pointer for Phi (second half of ZPhi)

	//// Initialize other device arrays as needed
	//cudaMemset(devZpp, 0, N * sizeof(cufftDoubleComplex));
	//cudaMemset(deva, 0, N * sizeof(double));
	//cudaMemset(devaprime, 0, N * sizeof(double));
	//cudaMemset(devV1, 0, N * N * sizeof(cufftDoubleComplex));
	//cudaMemset(devV2, 0, N * sizeof(cufftDoubleComplex));
	//cudaMemset(devVelocitiesLower, 0, N * sizeof(cufftDoubleComplex));
	//cudaMemset(devVelocitiesUpper, 0, N * sizeof(cufftDoubleComplex));
}

template<int N>
inline void TimeStepManager<N>::runTimeStep()
{
	// Here you would implement the logic to run a time step of the simulation.
	// This would typically involve:
	// 1. Calculating ZPhiPrime and Zpp from devZPhi.
	// 2. Create the M matrix using devZPhi, devZPhiPrime, and devZpp.
	// 3. Solve Ma = phi' to obtain the vorticities (a).
	// 4. Calculate the derivatives of a.
	// 5. Creating the V1 and V2 matrices using devZPhi, devZPhiPrime, and devZpp, a'
	// 6. Calculating the velocities for both lower and upper fluids.
	// 7. Updating the RHS of state variables (e.g., deva, devaprime) based on the calculated velocities.

	zPhiDerivative.exec(devZPhi, devZPhiPrime, devZpp); // Calculate derivatives of Z and Phi

	createMKernel << <matrix_blocks, matrix_threads >> > (devM, devZPhi, devZPhiPrime, devZpp, problemProperties.rho,  N); // Create the M matrix
	
	complex_to_real << <blocks, threads >> > (devZPhiPrime + N, devPhiPrime, N); // Convert ZPhiPrime to real PhiPrime (takes only the real part).
	
	matrixSolver.solve(devM, devPhiPrime, deva); // Solve the system Ma = phi' to get the vorticities (a)
#ifdef DEBUG_DERIVATIVES
	cudaDeviceSynchronize(); // Ensure all previous operations are complete before proceeding

	//cudaDeviceSynchronize(); // wait for the solver to finish

	
	//cudaDeviceSynchronize(); // Ensure all previous operations are complete before proceeding
	std::vector<double> PhiPrime_host(N, 0);
	std::vector<double> xprime(N, 0.0);
	std::vector<double> yprime(N, 0.0);

	std::vector<double> x(N, 0.0); // Host vectors to store the real and imaginary parts of ZPhiPrime for plotting
	std::vector<double> y(N, 0.0); // Host vectors to store the real and imaginary parts of ZPhiPrime for plotting

	std::vector<cuDoubleComplex> ZPhi_host(2*N, make_cuDoubleComplex(0,0));
	std::vector<cuDoubleComplex> ZPhiPrime_host(2 * N, make_cuDoubleComplex(0, 0)); // Host vectors to store the results of  ZPhiPrime
	std::vector<double> Phi(N, 0);

	cudaMemcpy(ZPhiPrime_host.data(), devZPhiPrime, 2 * N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost); // Copy the ZPhiPrime from device to host for debugging or further processing
	cudaMemcpy(ZPhi_host.data(), devZPhi, 2*N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost); // Copy the Phi from device to host for debugging or further processing
	cudaMemcpy(PhiPrime_host.data(), devPhiPrime, N * sizeof(double), cudaMemcpyDeviceToHost); // Copy the ZPhiPrime from device to host for debugging or further processing

	for (int i = 0; i < 2 * N; i++)
	{
		if (i < N)
		{
			x[i] = ZPhi_host[i].x; // Store the real part of ZPhi for plotting
			y[i] = ZPhi_host[i].y; // Store the imaginary part of ZPhi for plotting
			xprime[i] = ZPhiPrime_host[i].x; // Store the real part of ZPhi for plotting
			yprime[i] = ZPhiPrime_host[i].y; // Store the imaginary part of ZPhi for plotting
		}
		else
		{
			Phi[i - N] = ZPhi_host[i].x;
		}
	}
	plt::figure();
	plt::title("Phi and Phi Prime");
	plt::plot(Phi); // Plot the Phi values
	plt::plot(PhiPrime_host);
	plt::figure();
	plt::title("x and x prime");
	plt::plot(x); // Plot the real part of ZPhi
	plt::plot(xprime);
	plt::figure();
	plt::title("y and y prime");
	plt::plot(y); // Plot the imaginary part of ZPhi
	plt::plot(yprime);

	std::vector<cuDoubleComplex> a_host(N, make_cuDoubleComplex(0,0));
	
#endif
	real_to_complex << <blocks, threads >> > (deva, devaComplex, N); // Convert the real vorticities to complex form for velocity calculations
	//
	fftDerivative.exec(devaComplex, devaprime); // Calculate the derivative of a (vorticities)
	std::vector<double> x(N, 0.0); // Host vectors to store the real and imaginary parts of ZPhiPrime for plotting
	std::vector<cuDoubleComplex> ZPhi_host(2 * N, make_cuDoubleComplex(0, 0));
	std::vector<cuDoubleComplex> ZPhiPrime_host(2 * N, make_cuDoubleComplex(0, 0)); // Host vectors to store the results of  ZPhiPrime
	std::vector<cuDoubleComplex> a_host(N, make_cuDoubleComplex(0, 0));
	std::vector<double> a(N, 0.0);
	std::vector<double> aReal(N, 0.0); // Real part of the vorticities for debugging or further processing
	std::vector<double> Phi(N, 0);
	std::vector<double> PhiPrime_host(N, 0);
	cudaDeviceSynchronize();
	cudaMemcpy(ZPhiPrime_host.data(), devZPhiPrime, 2 * N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost); // Copy the ZPhiPrime from device to host for debugging or further processing
	cudaMemcpy(ZPhi_host.data(), devZPhi, 2 * N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
	cudaMemcpy(a.data(), deva, N * sizeof(double), cudaMemcpyDeviceToHost); // Copy the vorticities from device to host for debugging or further processing
	cudaMemcpy(a_host.data(), devaprime, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost); // Copy the vorticities from device to host for debugging or further processing
	cudaMemcpy(PhiPrime_host.data(), devPhiPrime, N * sizeof(double), cudaMemcpyDeviceToHost); // Copy the ZPhiPrime from device to host for debugging or further processing

	for (int i = 0; i < N; i++) {
		printf("%f\n", a[i]);
		aReal[i] = a_host[i].x; // Store the real part of the vorticities for further processing
		x[i] = ZPhi_host[i].x; // Store the real part of ZPhi for plotting
		Phi[i] = ZPhiPrime_host[N + i].x; // Store the real part of ZPhi for plotting
		//Phi[i] = ZPhiPrime_host[N + i].y; // Store the imaginary part of ZPhi for plotting
	}
	plt::figure();
	plt::plot(x, Phi, { {"label", "Phi'"} });
	plt::title("a, aprime");
	/*plt::plot(x, a, {{"label", "a"}});
	plt::plot(x, aReal, {{"label", "a prime"}});*/
	plt::legend();
	plt::show();

	// std::cin.get(); // Wait for user input to continue

	velocityCalculator.calculateVelocities(devZPhi, devZPhiPrime, devZpp, devaComplex, devaprime, devV1, devV2, devVelocitiesLower, true); // Calculate the velocities based on the vorticities and matrices

	velocityCalculator.calculateVelocities(devZPhi, devZPhiPrime, devZpp, devaComplex, devaprime, devV1, devV2, devVelocitiesUpper, false); // Calculate the velocities for the upper fluid
	// 7. the RHS of the X, Y are the velocities above.
	// The RHS of Phi is -(1 + rho) * Y + 1/2 * q1^2 + 1/2 * rho * q2^2 - rho * q1 . q2  + kappa / R
	calculatePhiRhs();
}
template<int N>
void TimeStepManager<N>::calculatePhiRhs()
{
	compute_rhs_phi_expression<<<blocks, threads>>>(devZPhi, devVelocitiesLower, devVelocitiesUpper, devRhsPhi, problemProperties.rho, N); // Calculate the right-hand side of the phi equation
}
#endif // TIMESTEP_MANAGER_H