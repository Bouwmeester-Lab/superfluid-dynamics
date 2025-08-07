#pragma once
#ifndef TIMESTEP_MANAGER_H
#define TIMESTEP_MANAGER_H

#include <CuDenseSolvers/Solvers/BiCGStab.cuh>
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
#include "AutonomousProblem.h"
#include "Energies.cuh"
#include "FiniteDepthOperator.cuh"
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;
template<int N>
class BoundaryProblem {
protected:
	const dim3 matrix_threads;// (16, 16);     // 256 threads per block in 2D
	const dim3 matrix_blocks; // ((N + 15) / 16, (N + 15) / 16);
	const int threads = 256; ///< Number of threads per block for CUDA kernels
	const int blocks = (N + threads - 1) / threads; ///< Number of blocks for CUDA kernels, ensuring all elements are covered
	std::unique_ptr<LinearOperator<double>> linearOperator; ///< Linear operator for applying the M matrix to the vorticities
public:
	EnergyContainer<N> energyContainer; ///< Energy container for storing the energies calculated during the simulation

	BoundaryProblem(EnergyBase<N>* kinetic, EnergyBase<N>* potential, EnergyBase<N>* surface) : matrix_threads(16, 16), matrix_blocks((N + 15) / 16, (N + 15) / 16), energyContainer(kinetic, potential, surface)
	{}
	virtual ~BoundaryProblem() 
	{
		/*if (linearOperator != nullptr)
		{
			delete linearOperator; ///< Clean up the linear operator if it was created
		}*/
	}
	/// <summary>
	/// reates the M matrix for the boundary integral problem.
	/// </summary>
	/// <param name="M">Output device pointer</param>
	/// <param name="Z">Input Z device pointer</param>
	/// <param name="Zp">Input Zp</param>
	/// <param name="Zpp"></param>
	/// <param name="rho"></param>
	/// <param name="n"></param>
	virtual void CreateMMatrix(double* M, std_complex* Z, std_complex* Zp, std_complex* Zpp, ProblemProperties& properties, int n) = 0;
	virtual void CalculateVelocities(std_complex* Z,
		std_complex* Zp,
		std_complex* Zpp,
		std_complex* a,
		std_complex* aprime,
		std_complex* V1,
		std_complex* V2,
		std_complex* velocities,
		ProblemProperties& properties,
		bool lower) = 0;
	virtual void CalculateRhsPhi(const std_complex* Z, const std_complex* V1, const std_complex* V2, std_complex* result, ProblemProperties& properties, int N) = 0;
	virtual void CalculateEnergy(const DevicePointers& devPointers) 
	{
		energyContainer.CalculateEnergy(devPointers); ///< Calculate the energies based on the device pointers containing the state variables
	};
	virtual LinearOperator<double>* GetLinearOperator() const
	{
		return linearOperator.get(); ///< Get the linear operator for applying the M matrix to the vorticities
	}
};

template<int N>
class WaterBoundaryProblem : public BoundaryProblem<N>
{
	VelocityCalculator<N> velocityCalculator; ///< Velocity calculator for calculating the velocities based on the vorticities and matrices.
public:
	WaterBoundaryProblem(ProblemProperties& properties) : BoundaryProblem<N>(new KineticEnergy<N>(properties), new GravitationalEnergy<N>(properties), new SurfaceEnergy<N>(properties)), velocityCalculator()
	{
		// Constructor for the water boundary problem, initializing the velocity calculator with the problem properties
	}

	virtual void CreateMMatrix(double* M, std_complex* Z, std_complex* Zp, std_complex* Zpp, ProblemProperties& properties, int n) override
	{
		createMKernel << <this->matrix_blocks, this->matrix_threads >> > (M, Z, Zp, Zpp, properties.rho, n);
	}
	virtual void CalculateVelocities(std_complex* Z,
		std_complex* Zp,
		std_complex* Zpp,
		std_complex* a,
		std_complex* aprime,
		std_complex* V1,
		std_complex* V2,
		std_complex* velocities,
		ProblemProperties& properties,
		bool lower) override
	{
		// create the V1 matrix and V2 diagonal vector
		createVelocityMatrices << <this->matrix_blocks, this->matrix_threads >> > (Z, Zp, Zpp, N, V1, V2, lower);
		velocityCalculator.calculateVelocities(Z, Zp, Zpp, a, aprime, V1, V2, velocities, lower);
	}
	virtual void CalculateRhsPhi(const std_complex* Z, const std_complex* V1, const std_complex* V2, std_complex* result, ProblemProperties& properties, int N) override
	{
		compute_rhs_phi_expression << <this->blocks, this->threads >> > (Z, V1, V2, result, properties.rho, N);
	}
};

template<int N>
class HeliumBoundaryProblem : public BoundaryProblem<N>
{
	VelocityCalculator<N> velocityCalculator; ///< Velocity calculator for calculating the velocities based on the vorticities and matrices.
public:
	HeliumBoundaryProblem(ProblemProperties& properties) : BoundaryProblem<N>(new KineticEnergy<N>(properties), new VanDerWaalsEnergy<N>(properties), new SurfaceEnergy<N>(properties)), velocityCalculator()
	{
		// Constructor for the helium boundary problem, initializing the velocity calculator with the problem properties
	}
	virtual void CreateMMatrix(double* M, std_complex* Z, std_complex* Zp, std_complex* Zpp, ProblemProperties& properties, int n) override
	{
		// createFiniteDepthMKernel<< <this->matrix_blocks, this->matrix_threads >> > (M, Z, Zp, Zpp, properties.depth, n);
		if(this->linearOperator == nullptr)
		{
			this->linearOperator = std::make_unique<FiniteDepthOperator<N>>(Z, Zp, Zpp, properties.depth);
		}
	}

	virtual void CalculateVelocities(std_complex* Z,
		std_complex* Zp,
		std_complex* Zpp,
		std_complex* a,
		std_complex* aprime,
		std_complex* V1,
		std_complex* V2,
		std_complex* velocities,
		ProblemProperties& properties,
		bool lower) override
	{
		// create the V1 matrix and V2 diagonal vector
		createHeliumVelocityMatrices << <this->matrix_blocks, this->matrix_threads >> > (Z, Zp, Zpp,properties.depth,  N, V1, V2, lower);
		velocityCalculator.calculateVelocities(Z, Zp, Zpp, a, aprime, V1, V2, velocities, lower);
	}
	virtual void CalculateRhsPhi(const std_complex* Z, const std_complex* V1, const std_complex* V2, std_complex* result, ProblemProperties& properties, int N) override
	{
		compute_rhs_helium_phi_expression << <this->blocks, this->threads >> > (Z, V1, result, properties.depth, N);
	}
};

template <int N>
class HeliumInfiniteDepthBoundaryProblem : public BoundaryProblem<N>
{
	VelocityCalculator<N> velocityCalculator; ///< Velocity calculator for calculating the velocities based on the vorticities and matrices.
public:
	HeliumInfiniteDepthBoundaryProblem(ProblemProperties& properties) : BoundaryProblem<N>(new KineticEnergy<N>(properties), new VanDerWaalsEnergy<N>(properties), new SurfaceEnergy<N>(properties)), velocityCalculator()
	{
		// Constructor for the helium infinite depth boundary problem, initializing the velocity calculator with the problem properties
	}
	virtual void CreateMMatrix(double* M, std_complex* Z, std_complex* Zp, std_complex* Zpp, ProblemProperties& properties, int n) override
	{
		createMKernel << <this->matrix_blocks, this->matrix_threads >> > (M, Z, Zp, Zpp, properties.depth, n);
	}
	virtual void CalculateVelocities(std_complex* Z,
		std_complex* Zp,
		std_complex* Zpp,
		std_complex* a,
		std_complex* aprime,
		std_complex* V1,
		std_complex* V2,
		std_complex* velocities,
		ProblemProperties& properties,
		bool lower) override
	{
		// create the V1 matrix and V2 diagonal vector
		createVelocityMatrices << <this->matrix_blocks, this->matrix_threads >> > (Z, Zp, Zpp, N, V1, V2, lower);
		velocityCalculator.calculateVelocities(Z, Zp, Zpp, a, aprime, V1, V2, velocities, lower);
	}
	virtual void CalculateRhsPhi(const std_complex* Z, const std_complex* V1, const std_complex* V2, std_complex* result, ProblemProperties& properties, int N) override
	{
		compute_rhs_helium_phi_expression << <this->blocks, this->threads >> > (Z, V1, result, properties.depth,N);
	}
};


template<int N>
class BoundaryIntegralCalculator : public AutonomousProblem<std_complex, 2*N>
{
public:
	BoundaryIntegralCalculator(ProblemProperties& problemProperties, BoundaryProblem<N>& boundaryProblem);
	~BoundaryIntegralCalculator();

	VolumeFlux<N> volumeFlux; ///< Volume flux calculator for the water boundary integral problem

	/// <summary>
	/// Initializes the time step manager by copying the host data to the device memory for the initial conditions. Use this only once when setting up the problem from the host side.
	/// </summary>
	/// <param name="Z0"></param>
	/// <param name="Phi0"></param>
	void initialize_device(std_complex* Z0, std_complex* Phi0);
	void setZPhi(std_complex* devZ) { this->devZ = devZ; }
	void runTimeStep();

	virtual void run(std_complex* initialState) override;

	std_complex* devVelocitiesLower; ///< Device pointer to the velocities array (lower fluid)
	std_complex* devVelocitiesUpper; ///< Device pointer to the velocities array (upper fluid)
	std_complex* devRhsPhi; ///< Device pointer to the right-hand side of the phi equation (derivative of Phi/dt)

	double* devPhiPrime; ///< Device pointer to the PhiPrime array (derivative of Phi)

	std_complex* getDevZ() { return devZ; } ///< Getter for the device pointer to the Z array
	std_complex* getDevPhi() { return devPhi; } ///< Getter for the device pointer to the Phi array

	virtual std_complex* getY0() override { return devZ; } ///< Getter for the initial state (Z array)
private:
	cuda::std::complex<double>* devZ; ///< Device pointer to the Z array
	cuda::std::complex<double>* devPhi; ///< Device pointer to the Phi array
	BoundaryProblem<N>& boundaryProblem; ///< Reference to the boundary problem for creating the M matrix and calculating velocities

	std_complex* devZp; ///< Device pointer to the ZPhiPrime array
	std_complex* devPhiPrimeComplex; ///< Device pointer to the PhiPrime array (derivative of Phi)
	std_complex* devZpp; ///< Device pointer to the Zpp array

	double* devM; ///< Device pointer to the matrix M (NxN, double precision)
	double* deva; ///< Device pointer to the solution vector a
	std_complex* devaComplex; ///< Device pointer to the solution vector a in complex form (for compatibility with velocity calculations)

	std_complex* devaprime; ///< Device pointer to the derivative of a

	std_complex* devV1; ///< Device pointer to the V1 matrix
	std_complex* devV2; ///< Device pointer to the V2 diagonal vector

	
	

	ProblemProperties& problemProperties; ///< Reference to the problem properties for configuration
	ZPhiDerivative<N> zPhiDerivative; ///< Derivative calculator for Z and Phi
	FftDerivative<N, 1> fftDerivative; ///< FFT derivative calculator for single batch
	// MatrixSolver<N> matrixSolver; ///< Matrix solver for solving the vorticities.
	DoubleBiCGStab linearSolver; ///< Linear solver for solving the system of equations
	VelocityCalculator<N> velocityCalculator; ///< Velocity calculator for calculating the velocities based on the vorticities and matrices.
	const int threads = 256; ///< Number of threads per block for CUDA kernels
	const int blocks = (N + threads - 1) / threads; ///< Number of blocks for CUDA kernels, ensuring all elements are covered

	const dim3 matrix_threads;// (16, 16);     // 256 threads per block in 2D
	const dim3 matrix_blocks; // ((N + 15) / 16, (N + 15) / 16);


	void calculatePhiRhs(); ///< Helper function to calculate the right-hand side of the phi equation
};

template<int N>
BoundaryIntegralCalculator<N>::BoundaryIntegralCalculator(ProblemProperties& problemProperties, BoundaryProblem<N>& boundaryProblem) : AutonomousProblem<cufftDoubleComplex, 2*N>(), boundaryProblem(boundaryProblem), problemProperties(problemProperties), volumeFlux(problemProperties),
	zPhiDerivative(problemProperties), 
matrix_threads(16, 16), matrix_blocks((N + 15) / 16, (N + 15) / 16)
{
	// Allocate device memory for the various arrays used in the water boundary integral calculation
	cudaMalloc(&devZp, N * sizeof(std_complex));
	cudaMalloc(&devPhiPrimeComplex, N * sizeof(std_complex)); // Device pointer for the PhiPrime array (derivative of Phi in complex form)

	cudaMalloc(&devPhiPrime, N * sizeof(double)); // Device pointer for the PhiPrime array (derivative of Phi)
	cudaMalloc(&devZpp, N * sizeof(std_complex));
	cudaMalloc(&devM, N * N * sizeof(double)); // Matrix M for solving the system
	cudaMalloc(&deva, N * sizeof(double));
	cudaMalloc(&devaComplex, N * sizeof(cufftDoubleComplex)); // Device pointer for the solution vector a in complex form
	cudaMalloc(&devaprime, N * sizeof(cufftDoubleComplex));
	cudaMalloc(&devV1, N * N * sizeof(cufftDoubleComplex));
	cudaMalloc(&devV2, N * sizeof(cufftDoubleComplex));
	cudaMalloc(&devVelocitiesUpper, N * sizeof(cufftDoubleComplex)); // Device pointer for the velocities of the upper fluid

	// create device pointers for the velocities and right-hand side of the phi equation
	devVelocitiesLower = this->devTimeEvolutionRhs; // Device pointer for the velocities of the lower fluid
	devRhsPhi = this->devTimeEvolutionRhs + N; // Device pointer for the right-hand side of the phi equation

	fftDerivative.initialize(); // Initialize the FFT derivative calculator
}

template<int N>
BoundaryIntegralCalculator<N>::~BoundaryIntegralCalculator()
{
	cudaFree(devZ);
	cudaFree(devPhi);
	cudaFree(devZp);
	cudaFree(devPhiPrimeComplex);
	cudaFree(devPhiPrime);
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
inline void BoundaryIntegralCalculator<N>::initialize_device(std_complex* Z0, std_complex* Phi0)
{
	cudaMalloc(&devZ, 2 * N * sizeof(std_complex)); // we allocate 2*N for Z and Phi, in order to have them in the same memory space
	devPhi = devZ + N; // Set the device pointer for Phi (second half of devZ) // but we use different pointers for Z and Phi to avoid confusion

	// Copy initial conditions to device memory
	cudaMemcpy(devZ, Z0, N * sizeof(std_complex), cudaMemcpyHostToDevice);
	cudaMemcpy(devPhi, Phi0, N * sizeof(std_complex), cudaMemcpyHostToDevice);
}

template<int N>
inline void BoundaryIntegralCalculator<N>::runTimeStep()
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

	zPhiDerivative.exec(devZ, devPhi, devZp, devPhiPrimeComplex, devZpp); // Calculate derivatives of Z and Phi

	boundaryProblem.CreateMMatrix(devM, devZ, devZp, devZpp, problemProperties, N); // Create the M matrix linear operator for the boundary integral problem
	
	complex_to_real << <blocks, threads >> > (devPhiPrimeComplex, devPhiPrime, N); // Convert ZPhiPrime to real PhiPrime (takes only the real part).
	// set the linear operator for the solver
	linearSolver.setOperator(*boundaryProblem.GetLinearOperator());

	linearSolver.solve(devPhiPrime, deva, problemProperties.maxIterations, problemProperties.tolerance); // Solve the system Ma = phi' to get the vorticities (a)
	if (linearSolver.getNumIterations() >= problemProperties.maxIterations) {
		printf("Warning: Linear solver reached maximum iterations (%d) without converging to the specified tolerance (%e).\n", problemProperties.maxIterations, problemProperties.tolerance);
		printf("Consider increasing the tolerance or checking the problem setup.\n");
		printf("Solver residual norm: %e\n", linearSolver.getResidualNorm());
	}
	//matrixSolver.solve(devM, devPhiPrime, deva); // Solve the system Ma = phi' to get the vorticities (a)
#ifdef DEBUG_DERIVATIVES_2
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

	cudaMemcpy(ZPhiPrime_host.data(), devZ, 2 * N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost); // Copy the ZPhiPrime from device to host for debugging or further processing
	cudaMemcpy(ZPhi_host.data(), devZ, 2*N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost); // Copy the Phi from device to host for debugging or further processing
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
	fftDerivative.exec(devaComplex, devaprime, false, 2.0*PI_d / N); // Calculate the derivative of a (vorticities) 2.0*PI_d / static_cast<double>(N)
	
	//force_real_only << <blocks, threads >> > (devaprime, N); // Force the imaginary part of the primed vorticities to be zero
#ifdef DEBUG_DERIVATIVES_3
	std::vector<double> x(N, 0.0); // Host vectors to store the real and imaginary parts of ZPhiPrime for plotting
	std::vector<cuDoubleComplex> ZPhi_host(2 * N, make_cuDoubleComplex(0, 0));
	std::vector<cuDoubleComplex> ZPhiPrime_host(2 * N, make_cuDoubleComplex(0, 0)); // Host vectors to store the results of  ZPhiPrime
	std::vector<cuDoubleComplex> aPrimeHost(N, make_cuDoubleComplex(0, 0));
	std::vector<double> aHost(N, 0.0);
	std::vector<double> aPrimeReal(N, 0.0); // Real part of the vorticities for debugging or further processing
	std::vector<double> Phi(N, 0);
	std::vector<double> PhiPrime_host(N, 0);
	cudaDeviceSynchronize();
	cudaMemcpy(ZPhiPrime_host.data(), devZ, 2 * N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost); // Copy the ZPhiPrime from device to host for debugging or further processing
	cudaMemcpy(ZPhi_host.data(), devZ, 2 * N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
	cudaMemcpy(aHost.data(), deva, N * sizeof(double), cudaMemcpyDeviceToHost); // Copy the vorticities from device to host for debugging or further processing
	cudaMemcpy(aPrimeHost.data(), devaprime, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost); // Copy the vorticities from device to host for debugging or further processing
	cudaMemcpy(PhiPrime_host.data(), devPhiPrime, N * sizeof(double), cudaMemcpyDeviceToHost); // Copy the ZPhiPrime from device to host for debugging or further processing

	for (int i = 0; i < N; i++) {
		printf("%f\n", aHost[i]);
		aPrimeReal[i] = aPrimeHost[i].x; // Store the real part of the vorticities for further processing
		x[i] = ZPhi_host[i].x; // Store the real part of ZPhi for plotting
		Phi[i] = ZPhiPrime_host[N + i].x; // Store the real part of ZPhi for plotting
		//Phi[i] = ZPhiPrime_host[N + i].y; // Store the imaginary part of ZPhi for plotting
	}
	plt::figure();
	//plt::plot(x, Phi, { {"label", "Phi'"} });
	plt::title("a, aprime");
	plt::plot(x, aHost, {{"label", "a"}});
	plt::plot(x, aPrimeReal, {{"label", "a prime"}});
	plt::legend();
#endif
	//plt::show();

	// std::cin.get(); // Wait for user input to continue

	boundaryProblem.CalculateVelocities(devZ, devZp, devZpp, devaComplex, devaprime, devV1, devV2, devVelocitiesLower, problemProperties, true); // Calculate the velocities based on the vorticities and matrices

	boundaryProblem.CalculateVelocities(devZ, devZp, devZpp, devaComplex, devaprime, devV1, devV2, devVelocitiesUpper, problemProperties, false); // Calculate the velocities for the upper fluid
#ifdef DEBUG_VELOCITIES
	cudaDeviceSynchronize(); // Ensure all previous operations are complete before proceeding
	std::array<cufftDoubleComplex, N> VelocitiesLower; // Host array to store the velocities for the lower fluid
	std::vector<double> xVelocitiesLower(N, 0.0); // Host vector to store the real part of the velocities for the lower fluid
	std::vector<double> yVelocitiesLower(N, 0.0); // Host vector to store the imaginary part of the velocities for the lower fluid

	cudaMemcpy(VelocitiesLower.data(), devVelocitiesLower, N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost); // Copy the velocities for the lower fluid from device to host
	for (int i = 0; i < N; i++) {
		xVelocitiesLower[i] = VelocitiesLower[i].x; // Store the real part of the velocities for the lower fluid
		yVelocitiesLower[i] = VelocitiesLower[i].y; // Store the imaginary part of the velocities for the lower fluid
	}

	plt::figure();
	plt::title("Velocities Lower Fluid");
	plt::plot(xVelocitiesLower, { {"label", "X Velocities Lower Fluid"} }); // Plot the velocities for the lower fluid
	plt::plot(yVelocitiesLower, { {"label", "Y Velocities Lower Fluid"} }); // Plot the imaginary part of the velocities for the lower fluid
	plt::legend();
	plt::show(); // Show the plot for the velocities of the lower fluid
#endif
	// 7. the RHS of the X, Y are the velocities above.
	// The RHS of Phi is -(1 + rho) * Y + 1/2 * q1^2 + 1/2 * rho * q2^2 - rho * q1 . q2  + kappa / R
	/*calculatePhiRhs();*/
	boundaryProblem.CalculateRhsPhi(devZ, devVelocitiesLower, devVelocitiesUpper, devRhsPhi, problemProperties, N); // Calculate the right-hand side of the phi equation

	// calculate the evolution constants like the energy:
	//kineticEnergy.CalculateEnergy(devPhi, devZ, devZp, devVelocitiesLower); // Calculate the kinetic energy based on the current state
	//gravitationalEnergy.CalculateEnergy(devZ);// , devZp); // Calculate the gravitational energy based on the current state
	//surfaceEnergy.CalculateEnergy(devZp); // Calculate the surface energy based on the current state
	boundaryProblem.CalculateEnergy({ devZ, devZp, devZpp, devPhi, devVelocitiesLower }); // Calculate the energies based on the device pointers containing the state variables

	volumeFlux.CalculateEnergy(devZp, devVelocitiesLower); // Calculate the volume flux based on the current state
}
template<int N>
void BoundaryIntegralCalculator<N>::run(std_complex* initialState)
{
	this->setZPhi(initialState); // Set the initial state for ZPhi
	this->runTimeStep(); // Run the time step calculation
}

template<int N>
void BoundaryIntegralCalculator<N>::calculatePhiRhs()
{
	compute_rhs_phi_expression<<<blocks, threads>>>(devZ, devVelocitiesLower, devVelocitiesUpper, devRhsPhi, problemProperties.rho, N); // Calculate the right-hand side of the phi equation
}
#endif // TIMESTEP_MANAGER_H