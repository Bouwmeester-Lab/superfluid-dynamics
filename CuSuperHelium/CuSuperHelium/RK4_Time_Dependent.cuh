#pragma once
#ifndef RK4_TIME_DEPENDENT_H

#include "TimedBoundaryIntegrator.cuh"
#include "BaseBoundaryIntegrator.cuh"
#include "OdeSolver.h"
#include "cublas_v2.h"
#include "RK4Options.h"
#include "DataLogger.cuh"
#include "ValueLogger.h"
#include <thrust/device_vector.h>
#include "VectorUtilities.cuh"
//struct RK4_Options {
//	double timeStep = 1e-3;
//	// Add more options as needed, such as error tolerances for adaptive time stepping
//};

template <typename T, int N>
class RungeKuttaStepperBase : public OdeSolver
{
public:
	RungeKuttaStepperBase(TimedProblem<T, N>& timedProblem, double tstep = 1e-2);
	virtual ~RungeKuttaStepperBase();

	void setTimeStep(double tstep)
	{
		timeStep = CastTo<T>(tstep);
		halfTimeStep = CastTo<T>(tstep * 0.5);
		sixthTimeStep = CastTo<T>(tstep / 6.0);
	}

	void initialize(T* devY0, bool onDevice = false);
	void runStep(int _step);
	void setOptions(const RK4Options& options)
	{
		setTimeStep(options.initial_timestep);
		this->options = options;
	}
	virtual OdeSolverResult runEvolution(double startTime, double endTime) override;

	void setCurrentStream(cudaStream_t stream)
	{
		this->stream = stream;
	}
protected:
	const int threads = 256;
	const int blocks = (N + threads - 1) / threads;
	bool allocatedY0 = false; ///< Flag to indicate if devY0 was allocated by the stepper
	RK4Options options;

	thrust::device_vector<double> devTimes;
	thrust::device_vector<cuda::std::array<std_complex, N>> devYs;

	T* k1;
	T* k2;
	T* k3;
	T* k4;

	T* devY0;
	T* devY1;
	T* devY2;
	T* devY3;

	TimedProblem<T, N>& timedProblem; ///< Instance of the TimeStepManager to handle time-stepping operations
	cublasHandle_t handle; ///< CUBLAS handle for matrix operations

	T timeStep;
	T halfTimeStep;
	T sixthTimeStep;
	std::vector<std::shared_ptr<ValueLogger>> valueLoggers;

	double currentTime = 0.0;
	cudaStream_t stream = cudaStreamPerThread; // Use the default stream for now, can be customized later

	virtual void step(const int i) = 0;
public:
	int copyTimesToHost(double** hostTimes, size_t* countHost);
	int copyStatesToHost(std_complex** hostStates, size_t* countHost);
};


template<typename T, int N>
int RungeKuttaStepperBase<T, N>::copyTimesToHost(double** hostTimes, size_t* countHost)
{
	if (this->options.returnTrajectory)
	{
		double* t = static_cast<double*>(std::malloc(devTimes.size() * sizeof(double)));
		if (!t) {
			return -1; // memory allocation failed
		}
		// copy times to host
		checkCuda(cudaMemcpyAsync(t, thrust::raw_pointer_cast(devTimes.data()), devTimes.size() * sizeof(double), cudaMemcpyDeviceToHost, this->stream), __func__, __FILE__, __LINE__);
		*hostTimes = t;
		*countHost = devTimes.size();
	}
	else
	{
		// no times were recorded
		*hostTimes = nullptr;
		*countHost = 0;
	}
	return 0;
}

template<typename T, int N>
int RungeKuttaStepperBase<T, N>::copyStatesToHost(std_complex** hostStates, size_t* countHost)
{
	if (this->options.returnTrajectory)
	{
		std_complex* s = static_cast<std_complex*>(std::malloc(devYs.size() *  N * sizeof(std_complex)));
		//double* s = static_cast<double*>(std::malloc(devYs.size() * 3 * N * sizeof(double)));
		if (!s) {
			return -1; // memory allocation failed
		}
		// copy states to host
		checkCuda(cudaMemcpyAsync(s, thrust::raw_pointer_cast(devYs.data()), devYs.size() * N * sizeof(std_complex), cudaMemcpyDeviceToHost, this->stream), __func__, __FILE__, __LINE__);
		*hostStates = s;
		*countHost = devYs.size();
	}
	else
	{
		// no states were recorded so copy the latest state calculated.
		std_complex* s = static_cast<std_complex*>(std::malloc(N * sizeof(std_complex)));
		if (!s) {
			return -1; // memory allocation failed
		}
		checkCuda(cudaMemcpyAsync(s, this->devY0, N * sizeof(std_complex), cudaMemcpyDeviceToHost, this->stream), __func__, __FILE__, __LINE__);

		*hostStates = s;
		*countHost = 1;
	}
	return 0;
}




template <typename T, int N>
RungeKuttaStepperBase<T, N>::RungeKuttaStepperBase(TimedProblem<T, N>& timedProblem, double tstep) : timedProblem(timedProblem)
{
	cublasCreate(&handle);
	cudaMalloc(&k1, N * sizeof(T)); // allocate memory for k1
	cudaMalloc(&k2, N * sizeof(T)); // allocate memory for k2
	cudaMalloc(&k3, N * sizeof(T)); // allocate memory for k3
	cudaMalloc(&k4, N * sizeof(T)); // allocate memory for k4

	cudaMalloc(&devY1, N * sizeof(T)); // allocate memory for devZphi1
	cudaMalloc(&devY2, N * sizeof(T)); // allocate memory for devZphi2
	cudaMalloc(&devY3, N * sizeof(T)); // allocate memory for devZphi3

	setTimeStep(tstep);
}

template <typename T, int N>
RungeKuttaStepperBase<T, N>::~RungeKuttaStepperBase()
{
	if (allocatedY0)
		cudaFree(devY0);
	cudaFree(k1);
	cudaFree(k2);
	cudaFree(k3);
	cudaFree(k4);

	cudaFree(devY1);
	cudaFree(devY2);
	cudaFree(devY3);
	cublasDestroy(handle);
}

template <typename T, int N>
void RungeKuttaStepperBase<T, N>::runStep(int _step)
{
	timedProblem.setCurrentTime(CastFrom<std_complex>(currentTime)); // set the current time in the autonomous problem
	timedProblem.setSaveProgress(true);
	timedProblem.run(devY0, k1); // run the autonomous problem with the initial state devY0 and store the result in k1

	// copyk(1); // copy k1 from the autonomousProblem
#ifdef DEBUG_RUNGE_KUTTA
	cudaDeviceSynchronize(); // synchronize the device to ensure all operations are completed
	std::vector<cufftDoubleComplex> k1_host(N);
	std::vector<cufftDoubleComplex> y_host(N);

	std::vector<double> x(N/2);
	std::vector<double> y(N/2);
	std::vector<double> Phi(N/2);
	std::vector<double> dx(N/2);
	std::vector<double> dy(N/2);
	std::vector<double> dPhi(N/2);

	cudaMemcpy(k1_host.data(), k1, N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost); // copy k1 to host for debugging
	cudaMemcpy(y_host.data(), devY0, N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost); // copy devY0 to host for debugging

	for (int i = 0; i < N/2; i++) {
		x[i] = cuCreal(y_host[i]);
		y[i] = cuCimag(y_host[i]);
		// Phi[i] = cuCreal(y_host[i + N]); // assuming the second half of the array contains Phi values

		dx[i] = x[i] + halfTimeStep.real() * cuCreal(k1_host[i]);
		dy[i] = y[i] + halfTimeStep.real() * cuCimag(k1_host[i]);
		// dPhi[i] = Phi[i] + halfTimeStep.x * cuCreal(k1_host[i + N]); // assuming the second half of k1 contains Phi values
	}

	plt::figure();
	plt::plot(x, y, { {"label", "initial pos"} }); // plot the positions
	//plt::plot(x, Phi, { {"label", "initial phi"} }); // plot the Phi values
	plt::plot(dx, dy, { {"label", "perturbed pos"} });
	//plt::plot(x, dy, { {"label", "k1 y + dy"} });
	// plt::plot(dx, dPhi, { {"label", "k1 phi + phi"} }); // plot the Phi component of k1

	/*plt::xlabel("x");
	plt::ylabel("y");*/
	plt::legend();
	plt::title("Positions after Runge-Kutta Step");

#endif
	// x_n+1 = x_n + h*0.5 * v_n -> x_n is a N dimensional vector, h is the time step, v_n is the velocity at time n
	step(1); // step 1 of the Runge-Kutta method

#ifdef DEBUG_RUNGE_KUTTA
	cudaDeviceSynchronize(); // synchronize the device to ensure all operations are completed
	cudaMemcpy(y_host.data(), devY1, N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost); // copy devY1 to host for debugging

	for (int i = 0; i < N/2; i++) {
		x[i] = cuCreal(y_host[i]);
		y[i] = cuCimag(y_host[i]);
		//Phi[i] = cuCreal(y_host[i + N]); // assuming the second half of the array contains Phi values
		//dx[i] = x[i] + halfTimeStep.x * cuCreal(k1_host[i]);
		//dy[i] = y[i] + halfTimeStep.x * cuCimag(k1_host[i]);
		//dPhi[i] = Phi[i] + halfTimeStep.x * cuCreal(k1_host[i + N]); // assuming the second half of k1 contains Phi values
	}
	plt::figure();
	plt::plot(x, y, { {"label", "pertubed calculated positions"} }); // plot the positions
	//plt::plot(x, Phi, { {"label", "pertubed calculated phi"} }); // plot the Phi values


#endif
	timedProblem.setCurrentTime(CastFrom<std_complex>(currentTime) + CastFrom<std_complex>(halfTimeStep)); // set the current time in the autonomous problem to the midpoint of the interval for the second step
	timedProblem.setSaveProgress(false);
	timedProblem.run(devY1, k2);

	//copyk(2); // copy k2 from the autonomousProblem

	step(2); // step 2 of the Runge-Kutta method

#ifdef DEBUG_RUNGE_KUTTA
	cudaDeviceSynchronize(); // synchronize the device to ensure all operations are completed
	cudaMemcpy(y_host.data(), devY2, N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost); // copy devY1 to host for debugging

	for (int i = 0; i < N/2; i++) {
		x[i] = cuCreal(y_host[i]);
		y[i] = cuCimag(y_host[i]);
		//Phi[i] = cuCreal(y_host[i + N]); // assuming the second half of the array contains Phi values
		//dx[i] = x[i] + halfTimeStep.x * cuCreal(k1_host[i]);
		//dy[i] = y[i] + halfTimeStep.x * cuCimag(k1_host[i]);
		//dPhi[i] = Phi[i] + halfTimeStep.x * cuCreal(k1_host[i + N]); // assuming the second half of k1 contains Phi values
	}

	plt::plot(x, y, { {"label", "3rd pertubed calculated positions"} }); // plot the positions
	//plt::plot(x, Phi, { {"label", "3rd pertubed calculated phi"} }); // plot the Phi values

	//plt::legend();
#endif
	timedProblem.setCurrentTime(CastFrom<std_complex>(currentTime) + CastFrom<std_complex>(halfTimeStep)); // set the current time in the autonomous problem to the midpoint of the interval for the third step
	timedProblem.run(devY2, k3);

	//copyk(3); // copy k3 from the autonomousProblem

	step(3);

	timedProblem.setCurrentTime(CastFrom<std_complex>(currentTime) + CastFrom<std_complex>(timeStep)); // set the current time in the autonomous problem to the end of the interval for the fourth step
	timedProblem.run(devY3, k4);

	//copyk(4); // copy k4 from the autonomousProblem

#ifdef DEBUG_RUNGE_KUTTA
	cudaDeviceSynchronize(); // synchronize the device to ensure all operations are completed
	cudaMemcpy(y_host.data(), devY3, N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost); // copy devY1 to host for debugging

	for (int i = 0; i < N; i++) {
		x[i] = cuCreal(y_host[i]);
		y[i] = cuCimag(y_host[i]);
		//Phi[i] = cuCreal(y_host[i + N]); // assuming the second half of the array contains Phi values
		//dx[i] = x[i] + halfTimeStep.x * cuCreal(k1_host[i]);
		//dy[i] = y[i] + halfTimeStep.x * cuCimag(k1_host[i]);
		//dPhi[i] = Phi[i] + halfTimeStep.x * cuCreal(k1_host[i + N]); // assuming the second half of k1 contains Phi values
	}

	plt::plot(x, y, { {"label", "4th pertubed calculated positions"} }); // plot the positions
	//plt::plot(x, Phi, { {"label", "4th pertubed calculated phi"} }); // plot the Phi values

	plt::legend();
#endif

	add_k_vectors << <blocks, threads >> > (k1, k2, k3, k4, k1, N); // add the four vectors together

	//cudaDeviceSynchronize(); // synchronize the device to ensure all operations are completed

	//std::vector<cufftDoubleComplex> k1_host(2 * N);

	//cudaMemcpy(k1_host.data(), k1, 2 * N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost); // copy k1 to host for debugging
	//printf("%f h/6\n", cuCreal(sixthTimeStep)); // print the time step for debugging
	//for(int i = 0; i < 2*N; i++) {
	//	printf("k[%d] = (%f, %f)\n", i, cuCreal(k1_host[i]), cuCimag(k1_host[i]));
	//}
	//std::cin.get(); // wait for user input to continue
#ifdef DEBUG_RUNGE_KUTTA
	cudaDeviceSynchronize(); // synchronize the device to ensure all operations are completed
	cudaMemcpy(y_host.data(), devY0, N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost); // copy devY1 to host for debugging
	cudaMemcpy(k1_host.data(), k1, N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; i++) {
		x[i] = cuCreal(y_host[i]); // +sixthTimeStep.x * k1_host[i].x;
		y[i] = cuCimag(y_host[i]);// +sixthTimeStep.x * k1_host[i].y;
		//Phi[i] = cuCreal(y_host[i + N]);// +sixthTimeStep.x * k1_host[i + N].x; // assuming the second half of the array contains Phi values
		//dx[i] = x[i] + halfTimeStep.x * cuCreal(k1_host[i]);
		//dy[i] = y[i] + halfTimeStep.x * cuCimag(k1_host[i]);
		//dPhi[i] = Phi[i] + halfTimeStep.x * cuCreal(k1_host[i + N]); // assuming the second half of k1 contains Phi values
	}
	plt::figure();
	plt::plot(x, y, { {"label", "Initial"} }); // plot the positions
	//plt::plot(x, Phi, { {"label", "Initial phi"} }); // plot the Phi values

#endif



	step(4); // step 4 of the Runge-Kutta method, which combines the results of the previous steps, it should overwrite devY0 with the final result

#ifdef DEBUG_RUNGE_KUTTA
	cudaDeviceSynchronize(); // synchronize the device to ensure all operations are completed
	cudaMemcpy(y_host.data(), devY0, N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost); // copy devY1 to host for debugging

	for (int i = 0; i < N; i++) {
		x[i] = cuCreal(y_host[i]);
		y[i] = cuCimag(y_host[i]);
		//Phi[i] = cuCreal(y_host[i + N]); // assuming the second half of the array contains Phi values
		dx[i] = sixthTimeStep.real() * cuCreal(k1_host[i]);

		dy[i] = sixthTimeStep.real() * cuCimag(k1_host[i]);
		//dPhi[i] = sixthTimeStep.x * cuCreal(k1_host[i + N]); // assuming the second half of k1 contains Phi values
	}

	plt::plot(x, y, { {"label", "Final"} }); // plot the positions
	//plt::plot(x, Phi, { {"label", "Final phi"} }); // plot the Phi values

	plt::legend();

	plt::figure();
	plt::plot(dx, { {"label", "dx"} }); // plot the x component of k1
	plt::plot(dy, { {"label", "dy"} }); // plot the y component of k1
	//plt::plot(dPhi, { {"label", "dPhi"} }); // plot the Phi component of k1
	plt::legend();

	plt::show();
#endif
	//if (logger.shouldCopy(_step))
	//	logger.setReadyToCopy(devY0, cudaStreamPerThread, currentTime + CastFrom<T>(timeStep), true); //TODO: make this work with a custom stream.
	//cudaDeviceSynchronize(); // synchronize the device to ensure all operations are completed
	initialize(devY0, true); // reinitialize the stepper with the initial state which is already on the device
}

template <typename T, int N>
void RungeKuttaStepperBase<T, N>::initialize(T* devY0, bool onDevice)
{
	if (onDevice)
	{
		this->devY0 = devY0; // store the initial state
		cudaMemcpy(devY1, devY0, N * sizeof(T), cudaMemcpyDeviceToDevice); // copy initial state to devY1
		cudaMemcpy(devY2, devY0, N * sizeof(T), cudaMemcpyDeviceToDevice); // copy initial state to devY2
		cudaMemcpy(devY3, devY0, N * sizeof(T), cudaMemcpyDeviceToDevice); // copy initial state to devY3
	}
	else {
		allocatedY0 = true;
		cudaMalloc(&this->devY0, N * sizeof(T)); // allocate memory for devZphi1
		cudaMemcpy(this->devY0, devY0, N * sizeof(T), cudaMemcpyHostToDevice); // copy initial state to devY1
		cudaMemcpy(devY1, devY0, N * sizeof(T), cudaMemcpyHostToDevice); // copy initial state to devY1
		cudaMemcpy(devY2, devY0, N * sizeof(T), cudaMemcpyHostToDevice); // copy initial state to devY2
		cudaMemcpy(devY3, devY0, N * sizeof(T), cudaMemcpyHostToDevice); // copy initial state to devY3
	}
	
	//cudaMemcpy(devZPhi4, devY0, 2 * N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice); // copy initial state to devZPhi4
}


template<typename T, int N>
OdeSolverResult RungeKuttaStepperBase<T, N>::runEvolution(double startTime, double endTime)
{
	currentTime = startTime;
	size_t steps = static_cast<size_t>((endTime - startTime) / CastFrom<T>(timeStep));
	timedProblem.setStartingTime(startTime);
	for (size_t step = 0; step < steps; step++)
	{
		runStep(step);
		
//#pragma unroll
//		for (auto& logger : valueLoggers)
//		{
//			if (logger->shouldLog(step))
//			{
//				logger->logValue();
//			}
//		}
		if (this->options.returnTrajectory)
		{
			this->devTimes.push_back(currentTime);
			appendToVector<N>(this->devYs, this->devY0, this->stream);
		}
		currentTime += CastFrom<T>(timeStep);
	}
	if (currentTime >= endTime)
		return OdeSolverResult::ReachedEndTime;
}


template <typename T, int N>
class RungeKuttaStepper : public RungeKuttaStepperBase<T, N>
{
public:
	using RungeKuttaStepperBase<T, N>::RungeKuttaStepperBase;
};

template <int N>
class RungeKuttaStepper<std_complex, N> : public RungeKuttaStepperBase<std_complex, N>
{
public:
	using RungeKuttaStepperBase<std_complex, N>::RungeKuttaStepperBase;
protected:
	virtual void step(const int i) override
	{
		cublasStatus_t result;
		if (i == 1)
		{
			result = cublasZaxpy(this->handle, N, reinterpret_cast<const cuDoubleComplex*>(&this->halfTimeStep), reinterpret_cast<cuDoubleComplex*>(this->k1), 1, reinterpret_cast<cuDoubleComplex*>(this->devY1), 1);
		}
		else if (i == 2)
		{
			result = cublasZaxpy(this->handle, N, reinterpret_cast<cuDoubleComplex*>(&this->halfTimeStep), reinterpret_cast<cuDoubleComplex*>(this->k2), 1, reinterpret_cast<cuDoubleComplex*>(this->devY2), 1);
		}
		else if (i == 3)
		{
			result = cublasZaxpy(this->handle, N, reinterpret_cast<cuDoubleComplex*>(&this->timeStep), reinterpret_cast<cuDoubleComplex*>(this->k3), 1, reinterpret_cast<cuDoubleComplex*>(this->devY3), 1);
		}
		else if (i == 4)
		{
			result = cublasZaxpy(this->handle, N, reinterpret_cast<cuDoubleComplex*>(&this->sixthTimeStep), reinterpret_cast<cuDoubleComplex*>(this->k1), 1, reinterpret_cast<cuDoubleComplex*>(this->devY0), 1); // k1 must contain the sum of k1, k2, k3, and k4
		}
		else
		{
			fprintf(stderr, "Invalid step index: %d\n", i);
			return;
		}


		if (result != CUBLAS_STATUS_SUCCESS) {
			fprintf(stderr, "cublasZaxpy failed with error code %d\n", result);
			return;
		}
	}
};

#endif