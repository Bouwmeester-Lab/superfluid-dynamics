
//#define DEBUG_DERIVATIVES
//#define DEBUG_RUNGE_KUTTA
//#define DEBUG_VELOCITIES
#include "ProblemProperties.hpp"
#include "array"
#include "vector"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <cufft.h>

#include "constants.cuh"
#include <complex>

#include "WaterBoundaryIntegralCalculator.cuh"
#include "SimpleEuler.cuh"
#include "AutonomousRungeKuttaStepper.cuh"

#include <format>
//#include "math.h"
//#include "complex.h"
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

#define j_complex std::complex<double>(0, 1)
cudaError_t setDevice();
cudaError_t fftDerivative();


__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

double X(double j, double h, double omega, double t) {
    return j - h * std::sin((j - omega * t));
}

double Y(double j, double h, double omega, double t) {
	return h * std::cos((j - omega * t));
}

double Phi(double j, double h, double omega, double t, double rho) {
    return h * ((1 + rho) * omega * std::sin((j - omega * t)));
}

int runTimeStep() 
{

    cudaError_t cudaStatus;
    cudaStatus = setDevice();
    if (cudaStatus != cudaSuccess) {
        return cudaStatus;
    }

	ProblemProperties problemProperties;
    problemProperties.rho = 0;
	problemProperties.kappa = 0;
    problemProperties.U = 0;

    const int N = 128;
    
	const double stepSize = PI_d/4000;
	const int steps = 2.81 / stepSize;
	WaterBoundaryIntegralCalculator<N> timeStepManager(problemProperties);

	std::array<double, N> j;
    std::vector<double> x0;
	std::vector<double> y0;

    std::array<std_complex, N> Z0;
	std::array<std_complex, N> PhiArr;
    std::vector<double> phiPrime;
    std::vector<double> phi0;
    std::vector<std_complex> dPhi(N);

	std::array<std_complex, N> VelocitiesLower;
    std::array<std_complex, N> VelocitiesUpper;

    std::array<std_complex, N> ZVect;
    std::array<std_complex, N> PhiVect;



    std::vector<double> x;
	std::vector<double> y;
	
    std::vector<double> KE(steps, 0);
	std::vector<double> PE(steps, 0);
	std::vector<double> SurfaceEnergy(steps, 0);
	std::vector<double> TotalEnergy(steps, 0);
	std::vector<double> VolumeFlux(steps, 0);

	x0.resize(N, 0);
	y0.resize(N, 0);
	phi0.resize(N, 0);
    x.resize(N, 0);
	y.resize(N, 0);
	phiPrime.resize(N, 0);
    double h = 0.5;
    double omega = 1;
    double t0 = 0;
	for (int i = 0; i < N; i++) {
		j[i] = 2.0 * PI_d * i / (1.0 * N);
		Z0[i] = std_complex(X(j[i], h, omega, t0), Y(j[i], h, omega, t0));
		x0[i] = Z0[i].real();
		y0[i] = Z0[i].imag();

        PhiArr[i] = Phi(j[i], h, omega, t0, problemProperties.rho);
		phi0[i] = PhiArr[i].real();
	}
    plt::figure();
    plt::title("Interface And Potential");
    plt::plot(x0, y0, {{"label", "Interface"}});
	plt::plot(x0, phi0, {{"label", "Potential"}});
    plt::legend();
    //plt::show();
    
	// Initialize the time step manager with the initial conditions.
    timeStepManager.initialize_device(Z0.data(), PhiArr.data());
    
    timeStepManager.runTimeStep();
    cudaDeviceSynchronize();
	cudaMemcpy(dPhi.data(), timeStepManager.devRhsPhi, N * sizeof(std_complex), cudaMemcpyDeviceToHost);
    cudaMemcpy(VelocitiesLower.data(), timeStepManager.devVelocitiesLower, N * sizeof(std_complex), cudaMemcpyDeviceToHost);
    printf("\velocities after 1: ");
    for (int i = 0; i < N; i++) {
        printf("{%f, %f} ", VelocitiesLower[i].real(), VelocitiesLower[i].imag());
        x[i] = VelocitiesLower[i].real();
        y[i] = VelocitiesLower[i].imag();
        phiPrime[i] = dPhi[i].real();
    }
    plt::figure();
    // plt::plot(x0, phi0);
    plt::title(std::format("Starting RHS using CUDA C++ using N = {}", N));
    plt::plot(x0, x, {{"label", "vx"}});
    plt::plot(x0, y, {{"label", "vy"}});
    plt::plot(x0, phiPrime, { {"label", "dPhi"} });
    plt::legend();
    // plt::show();
    // create Euler stepper
	AutonomousRungeKuttaStepper<std_complex, 2*N> rungeKunta(timeStepManager, stepSize);
	// Euler<N> euler(timeStepManager, stepSize);
	/*euler.setDevZ(devZ);
	euler.setDevPhi(devPhi);*/
    rungeKunta.initialize(timeStepManager.getY0());
	rungeKunta.setTimeStep(stepSize);

	for (int i = 0; i < steps; i++) {
        // Perform a time step
        rungeKunta.runStep();
        KE[i] = timeStepManager.kineticEnergy.getEnergy();
		PE[i] = timeStepManager.gravitationalEnergy.getEnergy();
		SurfaceEnergy[i] = timeStepManager.surfaceEnergy.getEnergy();
		VolumeFlux[i] = timeStepManager.volumeFlux.getEnergy();

		TotalEnergy[i] = KE[i] + PE[i] + SurfaceEnergy[i];
	}
	

    // timeStepManager.runTimeStep();
	cudaDeviceSynchronize();

	cudaMemcpy(ZVect.data(), timeStepManager.getDevZ(), N * sizeof(std_complex), cudaMemcpyDeviceToHost);
    cudaMemcpy(PhiVect.data(), timeStepManager.getDevPhi(), N * sizeof(std_complex), cudaMemcpyDeviceToHost);
    cudaMemcpy(VelocitiesLower.data(), timeStepManager.devVelocitiesLower, N * sizeof(std_complex), cudaMemcpyDeviceToHost);
    //cudaMemcpy(PhiVect.data(), devPhi, N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);

	printf("\velocities after 1: ");
	double t = steps * stepSize;
	std::vector<double> x_fin(N, 0);
	std::vector<double> y_fin(N, 0);

	for (int i = 0; i < N; i++) {
		printf("{%f, %f} ", VelocitiesLower[i].real(), VelocitiesLower[i].imag());
        x[i] = ZVect[i].real();
        y[i] = ZVect[i].imag();

		x_fin[i] = X(j[i], h, omega, t);
		y_fin[i] = Y(j[i], h, omega, t);
	}
	printf("\n");
    printf("\nPhi: ");
    for (int i = 0; i < N; i++) {
        printf("{%f, %f} ", PhiVect[i].real(), -1 * PhiVect[i].imag());
    }
    plt::figure();
    auto title = std::format("Interface And Potential at t={:.4f}", steps * stepSize);
	plt::title(title);

    //plt::plot(x_fin, y_fin, {{"label", "Interface at t=" + std::to_string(t)}});
    // Plot the initial position and the result of the Euler method

	plt::plot(x0, y0, {{"label", "Initial Position"}});
    plt::plot(x, y);
    plt::legend();

    plt::figure();
	plt::title("Kinetic, Potential and Surface Energy");
	plt::plot(KE, { {"label", "Kinetic Energy"} });
	plt::plot(PE, { {"label", "Potential Energy"} });
	plt::plot(SurfaceEnergy, { {"label", "Surface Energy"} });
	plt::plot(TotalEnergy, { {"label", "Total Energy"} });

	plt::legend();
	plt::xlabel("Time Steps");
	plt::ylabel("Energy");

	plt::figure();
	plt::title("Volume Flux");
	plt::plot(VolumeFlux, { {"label", "Volume Flux"} });
	plt::xlabel("Time Steps");
	plt::ylabel("Volume Flux");
	plt::legend();

    plt::show();
    

    return 0;
}

int main()
{
    //Py_SetPythonHome(L"C:/ProgramData/anaconda3");

    runTimeStep();
    //const int arraySize = 8;
    //const int a[arraySize] = { 1, 2, 3, 4, 5, 6, 7 };
    //const int b[arraySize] = { 10, 20, 30, 40, 50, 60, 70 };

    //const double X0[arraySize] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    //const double Y0[arraySize] = { 0, 1, 2, 3, 4, 5, 6, 7 };

    //int c[arraySize] = { 0 };

    //// test fft derivative
    //cudaError_t cudaStatus = fftDerivative();
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "addWithCuda failed!");
    //    return 1;
    //}

    //

    //// cudaDeviceReset must be called before exiting in order for profiling and
    //// tracing tools such as Nsight and Visual Profiler to show complete traces.
    //cudaStatus = cudaDeviceReset();
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaDeviceReset failed!");
    //    return 1;
    //}

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t setDevice()
{
    
    cudaError_t cudaStatus;
    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return cudaStatus;
    }

    return cudaStatus;
    
    
//    // cudaDeviceSynchronize waits for the kernel to finish, and returns
//    // any errors encountered during the launch.
//    cudaStatus = cudaDeviceSynchronize();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
//        goto Error;
//    }
//
//    // Copy output vector from GPU buffer to host memory.
//    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//Error:
//    cudaFree(dev_c);
//    cudaFree(dev_a);
//    cudaFree(dev_b);
//    
//    return cudaStatus;
}
