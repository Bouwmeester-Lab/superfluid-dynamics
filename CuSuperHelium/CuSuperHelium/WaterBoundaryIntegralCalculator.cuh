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
#include "AutonomousProblem.h"
#include "Energies.cuh"

#include "matplotlibcpp.h"

struct ProblemPointers 
{
public:
	const std_complex* Z; ///< Device pointer to the Z array (complex representation of the boundary)
	const std_complex* Zp; ///< Device pointer to the Zp array (derivative of Z)
	const std_complex* Zpp; ///< Device pointer to the Zpp array (second derivative of Z)
	const std_complex* Phi; ///< Device pointer to the Phi array (potential function on the boundary)
	const std_complex* VelocitiesUpper; ///< Device pointer to the velocities array (calculated velocities on the boundary)
	const std_complex* VelocitiesLower; ///< Device pointer to the velocities array (calculated velocities on the boundary)
};

namespace plt = matplotlibcpp;
template<int N, size_t batchSize>
class BoundaryProblem {
protected:
	const dim3 matrix_threads;// (16, 16);     // 256 threads per block in 2D
	const dim3 matrix_blocks; // ((N + 15) / 16, (N + 15) / 16);
	const int threads = 256; ///< Number of threads per block for CUDA kernels
	const int blocks = (batchSize * N + threads - 1) / threads; ///< Number of blocks for CUDA kernels, ensuring all elements are covered
public:
	EnergyContainer<N> energyContainer; ///< Energy container for storing the energies calculated during the simulation

	BoundaryProblem(EnergyBase<N>* kinetic, EnergyBase<N>* potential, EnergyBase<N>* surface) : matrix_threads(16, 16, 1), matrix_blocks((N + 15) / 16, (N + 15) / 16, batchSize), energyContainer(kinetic, potential, surface)
	{}
	virtual ~BoundaryProblem() {}
	/// <summary>
	/// reates the M matrix for the boundary integral problem.
	/// </summary>
	/// <param name="M">Output device pointer</param>
	/// <param name="Z">Input Z device pointer</param>
	/// <param name="Zp">Input Zp</param>
	/// <param name="Zpp"></param>
	/// <param name="rho"></param>
	/// <param name="n"></param>
	virtual void CreateMMatrix(double* M, const std_complex* Z, const std_complex* Zp, const std_complex* Zpp, ProblemProperties& properties) = 0;
	virtual void CalculateVelocities(const std_complex* Z,
		const std_complex* Zp,
		const std_complex* Zpp,
		std_complex* a,
		std_complex* aprime,
		std_complex* V1,
		std_complex* V2,
		std_complex* velocities,
		ProblemProperties& properties,
		bool lower) = 0;
	virtual void CalculateRhsPhi(const ProblemPointers problemPointers, std_complex* result, ProblemProperties& properties) = 0;
	virtual void CalculateEnergy(const DevicePointers& devPointers, cudaStream_t stream)
	{
		energyContainer.CalculateEnergy(devPointers, stream); ///< Calculate the energies based on the device pointers containing the state variables
	};
};

template<int N, size_t batchSize>
class WaterBoundaryProblem : public BoundaryProblem<N, batchSize>
{
	VelocityCalculator<N, batchSize> velocityCalculator; ///< Velocity calculator for calculating the velocities based on the vorticities and matrices.
public:
	WaterBoundaryProblem(ProblemProperties& properties) : BoundaryProblem<N, batchSize>(new KineticEnergy<N>(properties), new GravitationalEnergy<N>(properties), new SurfaceEnergy<N>(properties)), velocityCalculator()
	{
		// Constructor for the water boundary problem, initializing the velocity calculator with the problem properties
	}

	virtual void CreateMMatrix(double* M, const std_complex* Z, const std_complex* Zp, const std_complex* Zpp, ProblemProperties& properties) override
	{
		createMKernel << <this->matrix_blocks, this->matrix_threads >> > (M, Z, Zp, Zpp, properties.rho, N, batchSize);
	}
	virtual void CalculateVelocities(const std_complex* Z,
		const std_complex* Zp,
		const std_complex* Zpp,
		std_complex* a,
		std_complex* aprime,
		std_complex* V1,
		std_complex* V2,
		std_complex* velocities,
		ProblemProperties& properties,
		bool lower) override
	{
		// create the V1 matrix and V2 diagonal vector
		createVelocityMatrices << <this->matrix_blocks, this->matrix_threads >> > (Z, Zp, Zpp, N, V1, V2, lower, batchSize);
		velocityCalculator.calculateVelocities(Z, Zp, Zpp, a, aprime, V1, V2, velocities, lower);
	}
	virtual void CalculateRhsPhi(const ProblemPointers problemPointers, std_complex* result, ProblemProperties& properties) override
	{
		compute_rhs_phi_expression <<<this->blocks, this->threads >>> (problemPointers.Z, problemPointers.VelocitiesLower, problemPointers.VelocitiesUpper, result, properties.rho, batchSize * N);
	}
};

template<int N, size_t batchSize>
class HeliumBoundaryProblem : public BoundaryProblem<N, batchSize>
{
	VelocityCalculator<N, batchSize> velocityCalculator; ///< Velocity calculator for calculating the velocities based on the vorticities and matrices.
public:
	HeliumBoundaryProblem(ProblemProperties& properties) : BoundaryProblem<N, batchSize>(new KineticEnergy<N>(properties), new VanDerWaalsEnergy<N>(properties), new SurfaceEnergy<N>(properties)), velocityCalculator()
	{
		// Constructor for the helium boundary problem, initializing the velocity calculator with the problem properties
	}
	virtual void CreateMMatrix(double* M, const std_complex* Z, const std_complex* Zp, const std_complex* Zpp, ProblemProperties& properties) override
	{
		createFiniteDepthMKernel<< <this->matrix_blocks, this->matrix_threads >> > (M, Z, Zp, Zpp, properties.depth, N, batchSize);
	}

	virtual void CalculateVelocities(const std_complex* Z,
		const std_complex* Zp,
		const std_complex* Zpp,
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
	virtual void CalculateRhsPhi(const ProblemPointers problemPointers, std_complex* result, ProblemProperties& properties) override
	{
		if(properties.kappa != 0.0)
			compute_rhs_helium_phi_expression_with_surface_tension << <this->blocks, this->threads >> > (problemPointers.Z, problemPointers.Zp, problemPointers.Zpp, problemPointers.VelocitiesLower, result, properties.depth, properties.kappa, batchSize * N);
		else
			compute_rhs_helium_phi_expression << <this->blocks, this->threads >> > (problemPointers.Z, problemPointers.VelocitiesLower, result, properties.depth, batchSize * N);
	}
};

template <int N, size_t batchSize>
class HeliumInfiniteDepthBoundaryProblem : public BoundaryProblem<N, batchSize>
{
	VelocityCalculator<N, batchSize> velocityCalculator; ///< Velocity calculator for calculating the velocities based on the vorticities and matrices.
public:
	HeliumInfiniteDepthBoundaryProblem(ProblemProperties& properties) : BoundaryProblem<N, batchSize>(new KineticEnergy<N>(properties), new VanDerWaalsEnergy<N>(properties), new SurfaceEnergy<N>(properties)), velocityCalculator()
	{
		// Constructor for the helium infinite depth boundary problem, initializing the velocity calculator with the problem properties
	}
	virtual void CreateMMatrix(double* M, const std_complex* Z, const std_complex* Zp, const std_complex* Zpp, ProblemProperties& properties) override
	{
		createMKernel << <this->matrix_blocks, this->matrix_threads >> > (M, Z, Zp, Zpp, properties.depth, N, batchSize);
	}
	virtual void CalculateVelocities(const std_complex* Z,
		const std_complex* Zp,
		const std_complex* Zpp,
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
	virtual void CalculateRhsPhi(const ProblemPointers problemPointers, std_complex* result, ProblemProperties& properties) override
	{
		compute_rhs_helium_phi_expression << <this->blocks, this->threads >> > (problemPointers.Z, problemPointers.VelocitiesLower, result, properties.depth, batchSize * N);
	}
};

template<int N, size_t batchSize>
class BoundaryIntegralCalculator final : public AutonomousProblem<std_complex, 2*N*batchSize>
{
public:
	BoundaryIntegralCalculator(ProblemProperties& problemProperties, BoundaryProblem<N, batchSize>& boundaryProblem);
	~BoundaryIntegralCalculator();

	VolumeFlux<N> volumeFlux; ///< Volume flux calculator for the water boundary integral problem

	
	void runTimeStep(const std_complex* initialState, std_complex* rhs);
	void calculateVorticities(const std_complex* initialState);
	double* getDevA() ///< Getter for the device pointer to the vorticities array
	{
		return deva;
	}
	std_complex* getDevZp() {
		return devZp;
	} ///< Getter for the device pointer to the Zp array
	std_complex* getDevZpp() {
		return devZpp;
	} ///< Getter for the device pointer to the Zpp array

	virtual void run(std_complex* initialState, std_complex* rhs) override;

	// TODO: implement the setStream function to allow for asynchronous operations
	virtual void setStream(cudaStream_t stream) override
	{
		opsStream = stream;
	} ///< Set the CUDA stream for asynchronous operations

	// std_complex* devVelocitiesLower; ///< Device pointer to the velocities array (lower fluid)
	std_complex* devVelocitiesUpper; ///< Device pointer to the velocities array (upper fluid)
	// std_complex* devRhsPhi; ///< Device pointer to the right-hand side of the phi equation (derivative of Phi/dt)

	double* devPhiPrime; ///< Device pointer to the PhiPrime array (derivative of Phi)

	// std_complex* getDevZ() { return devZ; } ///< Getter for the device pointer to the Z array
	// std_complex* getDevPhi() { return devPhi; } ///< Getter for the device pointer to the Phi array

	//virtual std_complex* getY0() override { return devZ; } ///< Getter for the initial state (Z array)
private:
	//cuda::std::complex<double>* devZ; ///< Device pointer to the Z array
	//cuda::std::complex<double>* devPhi; ///< Device pointer to the Phi array
	BoundaryProblem<N, batchSize>& boundaryProblem; ///< Reference to the boundary problem for creating the M matrix and calculating velocities

	std_complex* devZp; ///< Device pointer to the ZPhiPrime array
	std_complex* devPhiPrimeComplex; ///< Device pointer to the PhiPrime array (derivative of Phi)
	std_complex* devZpp; ///< Device pointer to the Zpp array

	double* devM; ///< Device pointer to the matrix M (NxN, double precision)
	double* deva; ///< Device pointer to the solution vector a
	std_complex* devaComplex; ///< Device pointer to the solution vector a in complex form (for compatibility with velocity calculations)

	std_complex* devaprime; ///< Device pointer to the derivative of a

	std_complex* devV1; ///< Device pointer to the V1 matrix
	std_complex* devV2; ///< Device pointer to the V2 diagonal vector

	cudaStream_t opsStream = cudaStreamPerThread; /// < CUDA stream for computations.
	
	

	ProblemProperties& problemProperties; ///< Reference to the problem properties for configuration
	ZPhiDerivative<N, batchSize> zPhiDerivative; ///< Derivative calculator for Z and Phi
	FftDerivative<N, batchSize> fftDerivative; ///< FFT derivative calculator for single batch
	MatrixSolver<N, batchSize> matrixSolver; ///< Matrix solver for solving the vorticities.
	VelocityCalculator<N, batchSize> velocityCalculator; ///< Velocity calculator for calculating the velocities based on the vorticities and matrices.
	const int threads = 256; ///< Number of threads per block for CUDA kernels
	const int blocks = (batchSize * N + threads - 1) / threads; ///< Number of blocks for CUDA kernels, ensuring all elements are covered

	const dim3 matrix_threads;// (16, 16);     // 256 threads per block in 2D
	const dim3 matrix_blocks; // ((N + 15) / 16, (N + 15) / 16);
};

template<int N, size_t batchSize>
BoundaryIntegralCalculator<N, batchSize>::BoundaryIntegralCalculator(ProblemProperties& problemProperties, BoundaryProblem<N, batchSize>& boundaryProblem) : AutonomousProblem<cufftDoubleComplex, 2*N>(), boundaryProblem(boundaryProblem), problemProperties(problemProperties), volumeFlux(problemProperties),
	zPhiDerivative(problemProperties), 
matrix_threads(16, 16), matrix_blocks((N + 15) / 16, (N + 15) / 16)
{
	// Allocate device memory for the various arrays used in the water boundary integral calculation
	cudaMalloc(&devZp, batchSize * N * sizeof(std_complex));
	cudaMalloc(&devPhiPrimeComplex, batchSize * N * sizeof(std_complex)); // Device pointer for the PhiPrime array (derivative of Phi in complex form)

	cudaMalloc(&devPhiPrime, batchSize * N * sizeof(double)); // Device pointer for the PhiPrime array (derivative of Phi)
	cudaMalloc(&devZpp, batchSize * N * sizeof(std_complex));
	cudaMalloc(&devM, batchSize * N * N * sizeof(double)); // Matrix M for solving the system
	cudaMalloc(&deva, batchSize * N * sizeof(double));
	cudaMalloc(&devaComplex, batchSize * N * sizeof(cufftDoubleComplex)); // Device pointer for the solution vector a in complex form
	cudaMalloc(&devaprime, batchSize * N * sizeof(cufftDoubleComplex));
	cudaMalloc(&devV1, batchSize * N * N * sizeof(cufftDoubleComplex));
	cudaMalloc(&devV2, batchSize * N * sizeof(cufftDoubleComplex));
	cudaMalloc(&devVelocitiesUpper, batchSize * N * sizeof(cufftDoubleComplex)); // Device pointer for the velocities of the upper fluid

	fftDerivative.initialize(); // Initialize the FFT derivative calculator
}

template<int N, size_t batchSize>
BoundaryIntegralCalculator<N, batchSize>::~BoundaryIntegralCalculator()
{
	// cudaFree(devZ);
	// cudaFree(devPhi);
	cudaFreeAsync(devZp, opsStream);
	cudaFreeAsync(devPhiPrimeComplex, opsStream);
	cudaFreeAsync(devPhiPrime, opsStream);
	cudaFreeAsync(devZpp, opsStream);
	cudaFreeAsync(devM, opsStream);
	cudaFreeAsync(deva, opsStream);
	cudaFreeAsync(devaComplex, opsStream);
	cudaFreeAsync(devaprime, opsStream);
	cudaFreeAsync(devV1, opsStream);
	cudaFreeAsync(devV2, opsStream);
	// cudaFree(devVelocitiesLower);
	cudaFreeAsync(devVelocitiesUpper, opsStream);
	// cudaFree(devRhsPhi);
}

template<int N, size_t batchSize>
inline void BoundaryIntegralCalculator<N, batchSize>::runTimeStep(const std_complex* initialState, std_complex* rhs)
{
	// set the right-hand pointers using the rhs pointer, this creates the variables in here to avoid global variables
	std_complex* const devVelocitiesLower = rhs; // Device pointer for the velocities of the lower fluid
	std_complex* const devRhsPhi = rhs + batchSize * N; // Device pointer for the right-hand side of the phi equation
	// set the state variables using the initialState pointer, this creates the variables in here to avoid global variables
	const std_complex* devZ = initialState; // Device pointer for the Z array
	const std_complex* devPhi = initialState + batchSize * N; // Device pointer for the Phi array
	const ProblemPointers problemPointers{ devZ, devZp, devZpp, devPhi, devVelocitiesUpper, devVelocitiesLower };

	calculateVorticities(initialState); // Calculate the vorticities based on the current state
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
	real_to_complex << <blocks, threads >> > (deva, devaComplex, batchSize * N); // Convert the real vorticities to complex form for velocity calculations
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
	boundaryProblem.CalculateRhsPhi(problemPointers, devRhsPhi, problemProperties); // Calculate the right-hand side of the phi equation

	// calculate the evolution constants like the energy:
	//kineticEnergy.CalculateEnergy(devPhi, devZ, devZp, devVelocitiesLower); // Calculate the kinetic energy based on the current state
	//gravitationalEnergy.CalculateEnergy(devZ);// , devZp); // Calculate the gravitational energy based on the current state
	//surfaceEnergy.CalculateEnergy(devZp); // Calculate the surface energy based on the current state
	boundaryProblem.CalculateEnergy({ devZ, devZp, devZpp, devPhi, devVelocitiesLower }, opsStream); // Calculate the energies based on the device pointers containing the state variables

	volumeFlux.CalculateEnergy(devZp, devVelocitiesLower, opsStream); // Calculate the volume flux based on the current state
}

template<int N, size_t batchSize>
void BoundaryIntegralCalculator<N, batchSize>::calculateVorticities(const std_complex* initialState)
{
	// set the state variables using the initialState pointer, this creates the variables in here to avoid global variables
	const std_complex* devZ = initialState; // Device pointer for the Z array
	const std_complex* devPhi = initialState + batchSize * N; // Device pointer for the Phi array
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

	boundaryProblem.CreateMMatrix(devM, devZ, devZp, devZpp, problemProperties); // Create the M matrix

	complex_to_real << <blocks, threads >> > (devPhiPrimeComplex, devPhiPrime, batchSize * N); // Convert ZPhiPrime to real PhiPrime (takes only the real part).

	matrixSolver.solve(devM, devPhiPrime, deva); // Solve the system Ma = phi' to get the vorticities (a)
}


template<int N, size_t batchSize>
void BoundaryIntegralCalculator<N, batchSize>::run(std_complex* initialState, std_complex* rhs)
{
	
	this->runTimeStep(initialState, rhs); // Run the time step calculation
}

#endif // TIMESTEP_MANAGER_H
