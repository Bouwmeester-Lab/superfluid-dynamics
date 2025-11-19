#pragma once

#include "GLCoefficients.hpp"
#include "JacobianCalculator.cuh"
#include "OdeSolver.h"
#include <math.h>
#include <thrust/device_vector.h>
#include <cuda/std/array>
#include "VectorUtilities.cuh"
#include <indicators/progress_bar.hpp>
#include <cublas_v2.h>
#include "MatrixSolver.cuh"
#include "ExportTypes.cuh"
#include <indicators/cursor_control.hpp>

__global__ void calculateNextStateKernel(const double* devY, const double stepSize, const double* k1, const double* k2, double* devYnext, const GL_Coefficients coeffs, const size_t N)
{
	// ynext = devY + stepSize * ( B1 * k1 + B2 * k2 ) // devY, k1 k2 are two vectors of size 3N
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < 3 * N) {
		devYnext[idx] = devY[idx] + stepSize * (coeffs.b1 * k1[idx] + coeffs.b2 * k2[idx]);
	}
}

__global__ void stageStatesKernel(const double* devY, const double stepSize, const double* k1, const double* k2, double* y1, double* y2, const GL_Coefficients coeffs, const size_t N)
{
	// y1 = devY + stepSize * ( A11 * k1 + A12 * k2 ) // devY, k1 k2 are two vectors of size 3N
	// y2 = devY + stepSize * ( A21 * k1 + A22 * k2 )
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < 3 * N) {
		y1[idx] = devY[idx] + stepSize * (coeffs.A11 * k1[idx] + coeffs.A12 * k2[idx]);
		y2[idx] = devY[idx] + stepSize * (coeffs.A21 * k1[idx] + coeffs.A22 * k2[idx]);
	}
}

__global__ void createKTrial(const double* k1, const double* k2, const double alpha, const double* dK1, const double* dK2, double* ktrial1, double* ktrial2, const size_t N)
{
	// ktrial1 = k1 + alpha * dK1
	// ktrial2 = k2 + alpha * dK2
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < 3 * N) {
		ktrial1[idx] = k1[idx] + alpha * dK1[idx];
		ktrial2[idx] = k2[idx] + alpha * dK2[idx];
	}
}

__global__ void fillMMatrix(const double* devJ1, const double* devJ2, const double stepSize, const GL_Coefficients coeffs, double* devM, const size_t N) {
	// M = [ I - h * A11 * J1,   - h * A12 * J2
	//       - h * A21 * J1,     I - h * A22 * J2 ]
	size_t row = blockIdx.y * blockDim.y + threadIdx.y;
	size_t col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < 3 * N && col < 3 * N) {
		// Top-left block
		devM[row * (6 * N) + col] = (row == col ? 1.0 : 0.0) - stepSize * coeffs.A11 * devJ1[row * (3 * N) + col];
		// Top-right block
		devM[row * (6 * N) + (col + 3 * N)] = -stepSize * coeffs.A12 * devJ2[row * (3 * N) + col];
		// Bottom-left block
		devM[(row + 3 * N) * (6 * N) + col] = -stepSize * coeffs.A21 * devJ1[row * (3 * N) + col];
		// Bottom-right block
		devM[(row + 3 * N) * (6 * N) + (col + 3 * N)] = (row == col ? 1.0 : 0.0) - stepSize * coeffs.A22 * devJ2[row * (3 * N) + col];
	}
}

__global__ void multiplyVector(const double a, double* v, double* vout, size_t N)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= N) return;

	vout[idx] = a * v[idx];
}

__global__ void computeResidualsKernel(const double* fY1, const double* fY2, const double* k1, const double* k2, double* R1, double* R2, const size_t N) {
	// R1 = k1 - fY1
	// R2 = k2 - fY2
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < 3 * N) {
		R1[idx] = k1[idx] - fY1[idx];
		R2[idx] = k2[idx] - fY2[idx];
	}
}

struct GaussLegendre2Options {
	double stepSize;
	double newtonTolerance;
	size_t maxNewtonIterations;
	bool allowSimplifiedFallback = true;
	bool returnTrajectory = true;
	double armijo_c = 1e-4;
	double backtrack = 0.5;
	double minAlpha = 1e-6;
	size_t maxStepsHalves = 6;

	GaussLegendre2Options() :
		stepSize(0.01),
		newtonTolerance(1e-10),
		maxNewtonIterations(20),
		allowSimplifiedFallback(false),
		returnTrajectory(true)
	{
	}
};

GaussLegendre2Options createOptionsFromCOptions(const GaussLegendreOptions& cOptions)
{
	GaussLegendre2Options options;
	options.stepSize = cOptions.stepSize;
	options.newtonTolerance = cOptions.newtonTolerance;
	options.maxNewtonIterations = cOptions.maxNewtonIterations;
	options.allowSimplifiedFallback = cOptions.allowSimplifiedFallback;
	options.returnTrajectory = cOptions.returnTrajectory;
	options.armijo_c = cOptions.armijo_c;
	options.backtrack = cOptions.backtrack;
	options.minAlpha = cOptions.minAlpha;
	options.maxStepsHalves = cOptions.maxStepsHalves;
	return options;
}

size_t to_percent(double t, double t0, double t1)
{
	if (t <= t0) return 0;
	if (t >= t1) return 100;
	double frac = (t - t0) / (t1 - t0);
	return static_cast<std::size_t>(std::round(frac * 100.0));
};


template <size_t N>
class GaussLegendre2 : public OdeSolver
{
public:
	GaussLegendre2(AutonomousProblem<double, 3*N>& problem, JacobianCalculator<N>& jacobianCalculator, GaussLegendre2Options options = GaussLegendre2Options());
	~GaussLegendre2();
	virtual OdeSolverResult runEvolution(double startTime, double endTime) override;
	/// <summary>
	/// Sets the initial state for the solver.
	/// </summary>
	/// <param name="initialState"></param>
	/// <param name="onDevice"></param>
	void initialize(double* initialState, bool onDevice = false);
	void setStream(cudaStream_t stream) { this->stream = stream; }

	int copyTimesToHost(double** hostTimes, size_t* countHost);
	int copyStatesToHost(double** hostStates, size_t* countHost);
private:
	AutonomousProblem<double, 3 * N>& problem;
	JacobianCalculator<N>& jacobianCalculator;
	MatrixSolver<6 * N, 1> matrixSolver;
	GaussLegendre2Options options;
	void setOptions(const GaussLegendre2Options options) { this->options = options; }
	
	bool devStateOwned = false;

	double* devState = nullptr;
	double* devTempState = nullptr;
	double* devTempNextState = nullptr;

	double* y1 = nullptr;
	double* y2 = nullptr;

	double* k1 = nullptr;
	double* k2 = nullptr;

	double* ktrial = nullptr;
	double* ktrial1 = nullptr;
	double* ktrial2 = nullptr;

	double* devfY1 = nullptr;
	double* devfY2 = nullptr;

	double* R = nullptr;
	double* R1 = nullptr;
	double* R2 = nullptr;

	double* dK = nullptr;
	double* dK1 = nullptr;
	double* dK2 = nullptr;

	double* devJFrozen = nullptr;
	double* devJ1 = nullptr;
	double* devJ2 = nullptr;

	double* devM = nullptr;

	cublasHandle_t cublasHandle;

	GL_Coefficients glCoeffs;
	thrust::device_vector<double> devTimes;
	thrust::device_vector<cuda::std::array<double, 3 * N>> devYs;

	double hmin = -1;

	cudaStream_t stream = cudaStreamPerThread;

	std::unique_ptr<indicators::ProgressBar> progressBar;

	struct StepResult {
		size_t numberIterations;
		bool converged;
		double residualNorm;
		bool simplifiedFallbackUsed;
	} stepRes;

	struct StagingResult {
		double residualNorm;
		double phi;
		double normK;
	} stagingRes;

	void gaussLegendreS2Step(double* devCurrentState, double* devDestination, const double stepSize, StepResult& result);
	void stageStates(const double* devY, const double stepSize);
	void residualAndPhi(const double* devY, const double* k1, const double* k2, const double stepSize, StagingResult& stagingResult);
	void freeDeviceMemory();
};

template <size_t N>
GaussLegendre2<N>::GaussLegendre2(AutonomousProblem<double, 3*N>& problem, JacobianCalculator<N>& jacobianCalculator, GaussLegendre2Options options) : problem(problem), jacobianCalculator(jacobianCalculator), options(options)
{
}
template <size_t N>
GaussLegendre2<N>::~GaussLegendre2()
{
	// free device memory
	freeDeviceMemory();
}

template<size_t N>
OdeSolverResult GaussLegendre2<N>::runEvolution(double startTime, double endTime)
{
	if (this->options.stepSize < 0.0) {
		throw new std::invalid_argument("Step size must be positive.");
	}

	int nSteps = static_cast<int>(std::ceil(std::abs(endTime - startTime) / this->options.stepSize));
	double totalT = std::abs(endTime - startTime);

	if (hmin < 0) {
		hmin = totalT / (std::pow(2, 20));
	}


	if(this->devState == nullptr) {
		throw std::runtime_error("Initial state not set. Call initialize() before running evolution.");
	}
	// copy initial state to temp state
	cudaMemcpyAsync(this->devTempState, this->devState, 3 * N * sizeof(double), cudaMemcpyDeviceToDevice, this->stream);

	// create the trajectory storage
	if (options.returnTrajectory)
	{
		/*checkCuda(cudaMallocAsync(&this->devTimes, (nSteps + 1) * sizeof(double), this->stream), "GaussLegendre2<N>::runEvolution", "GaussLegendre.cuh", 85));
		linspace<<<(nSteps + 1 + 255) / 256, 256, 0, this->stream>>>(this->devTimes, startTime, endTime, nSteps + 1);*/
		this->devTimes.push_back(startTime);
		appendToVector<double, 3 * N>(this->devYs, this->devTempState, this->stream);
	}

	double currentTime = startTime;
	double forward = (endTime >= startTime) ? 1.0 : -1.0;
	double heff;
	double htry;
	// create the progress bar
	progressBar = std::make_unique<indicators::ProgressBar>(
		indicators::option::BarWidth{ 50 },
		indicators::option::Start{"["},
		indicators::option::Fill{"="},
		indicators::option::Lead{">"},
		indicators::option::Remainder{" "},
		indicators::option::End{"]"},
		indicators::option::PrefixText{"Gauss-Legendre 2nd Order Integration Progress:"},
		indicators::option::ShowPercentage{true},
		indicators::option::ShowElapsedTime{true},
		indicators::option::ShowRemainingTime{true},
		indicators::option::Stream{ std::cout }
	);
	indicators::show_console_cursor(false);

	bool attempt_ok;
	size_t previousPercent = 0;
	size_t percent = 0;
	
	std::cout << "Starting Gauss-Legendre 2nd Order Integration from t=" << startTime << " to t=" << endTime << " with initial step size " << this->options.stepSize << ".\n";
	while ((currentTime - endTime) * forward < 0.0)
	{
		heff = std::min(this->options.stepSize, std::abs(endTime - currentTime)) * forward;
		
		attempt_ok = false;
		htry = heff;
		for (int i = 0; i < this->options.maxStepsHalves + 1; ++i) 
		{
			this->gaussLegendreS2Step(this->devTempState, this->devTempNextState, htry, this->stepRes);
			if(this->stepRes.converged) 
			{
				attempt_ok = true;
				break;
			}
			if(htry <= hmin) 
			{
				break;
			}
			// Halve the step size and try again
			htry *= 0.5;
		}

		if(!attempt_ok) 
		{
			throw std::runtime_error(std::format("Gauss-Legendre 2nd Order method failed to converge at t≈{}; residual={:.3e}; last htry={:.3e}", currentTime, stepRes.residualNorm, htry));
		}

		// Step successful, update state and time
		cudaMemcpyAsync(this->devTempState, this->devTempNextState, 3 * N * sizeof(double), cudaMemcpyDeviceToDevice, this->stream);
		percent = to_percent(currentTime, startTime, endTime);
		if(percent != previousPercent) {
			progressBar->set_progress(percent);
			// std::cout << percent << "% completed.\n";
			previousPercent = percent;
		}

		//progressBar->set_progress();
		currentTime += htry;
		this->options.stepSize = std::abs(htry); // Update step size for next iteration

		if (this->options.returnTrajectory) 
		{
			this->devTimes.push_back(currentTime);
			appendToVector<double, 3 * N>(this->devYs, this->devTempState, this->stream);
		}
	}
	indicators::show_console_cursor(true);
	return OdeSolverResult::ReachedEndTime;
}

template<size_t N>
void GaussLegendre2<N>::initialize(double* initialState, bool onDevice)
{
	size_t stateSizeBytes = 3 * N * sizeof(double);
	if (onDevice) {
		this->devState = initialState;
	}
	else {
		devStateOwned = true;
		 // x, y, phi each of size N.
		cudaError_t err = cudaMalloc(&this->devState, stateSizeBytes);
		if (err != cudaSuccess) {
			throw std::runtime_error("Failed to allocate device memory for initial state.");
		}
		err = cudaMemcpy(this->devState, initialState, stateSizeBytes, cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			throw std::runtime_error("Failed to copy initial state to device memory.");
		}
	}

	checkCuda(cudaMallocAsync(&this->devTempState, stateSizeBytes, this->stream), __func__, __FILE__, __LINE__);
	checkCuda(cudaMallocAsync(&this->devTempNextState, stateSizeBytes, this->stream), __func__, __FILE__, __LINE__);

	// Also allocate k1, k2, y1, y2
	checkCuda(cudaMallocAsync(&this->k1, 2 * stateSizeBytes, this->stream), __func__, __FILE__, __LINE__);
	this->k2 = this->k1 + 3 * N;

	checkCuda(cudaMallocAsync(&this->y1, 2 * stateSizeBytes, this->stream), __func__, __FILE__, __LINE__);
	this->y2 = this->y1 + 3 * N;

	checkCuda(cudaMallocAsync(&this->R, 2 * stateSizeBytes, this->stream), __func__, __FILE__, __LINE__);
	this->R1 = this->R;
	this->R2 = this->R + 3 * N;

	checkCuda(cudaMallocAsync(&this->devfY1, 2 * stateSizeBytes, this->stream), __func__, __FILE__, __LINE__);
	this->devfY2 = this->devfY1 + 3 * N;

	cublasCreate(&this->cublasHandle);

	// allocate the space for the jacobians
	checkCuda(cudaMallocAsync(&this->devJFrozen, (3 * N) * (3 * N) * sizeof(double), this->stream), __func__, __FILE__, __LINE__);
	checkCuda(cudaMallocAsync(&this->devJ1, (3 * N) * (3 * N) * sizeof(double), this->stream), __func__, __FILE__, __LINE__);
	checkCuda(cudaMallocAsync(&this->devJ2, (3 * N) * (3 * N) * sizeof(double), this->stream), __func__, __FILE__, __LINE__);

	checkCuda(cudaMallocAsync(&this->devM, 4 * (3 * N) * (3 * N) * sizeof(double), this->stream), __func__, __FILE__, __LINE__);
	checkCuda(cudaMallocAsync(&this->dK, 2 * stateSizeBytes, this->stream), __func__, __FILE__, __LINE__);
	this->dK1 = this->dK;
	this->dK2 = this->dK + 3 * N;

	checkCuda(cudaMallocAsync(&this->ktrial, 2 * stateSizeBytes, this->stream), __func__, __FILE__, __LINE__);
	this->ktrial1 = this->ktrial;
	this->ktrial2 = this->ktrial + 3 * N;
}

template<size_t N>
int GaussLegendre2<N>::copyTimesToHost(double** hostTimes, size_t* countHost)
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

template<size_t N>
int GaussLegendre2<N>::copyStatesToHost(double** hostStates, size_t* countHost)
{
	if (this->options.returnTrajectory) 
	{
		double* s = static_cast<double*>(std::malloc(devYs.size() * 3 * N * sizeof(double)));
		if (!s) {
			return -1; // memory allocation failed
		}
		// copy states to host
		checkCuda(cudaMemcpyAsync(s, thrust::raw_pointer_cast(devYs.data()), devYs.size() * 3 * N * sizeof(double), cudaMemcpyDeviceToHost, this->stream), __func__, __FILE__, __LINE__);
		*hostStates = s;
		*countHost = devYs.size();
	}
	else
	{
		// no states were recorded so copy the latest state calculated.
		double* s = static_cast<double*>(std::malloc(3 * N * sizeof(double)));
		if (!s) {
			return -1; // memory allocation failed
		}
		checkCuda(cudaMemcpyAsync(s, this->devTempState, 3 * N * sizeof(double), cudaMemcpyDeviceToHost, this->stream), __func__, __FILE__, __LINE__);

		*hostStates = s;
		*countHost = 1;
	}
	return 0;
}

template<size_t N>
void GaussLegendre2<N>::freeDeviceMemory()
{
	if (this->devStateOwned) {
		cudaFreeAsync(this->devState, this->stream);
	}
	cudaFreeAsync(this->devTempState, this->stream);
	cudaFreeAsync(this->devTempNextState, this->stream);
	cudaFreeAsync(this->k1, this->stream);
	cudaFreeAsync(this->y1, this->stream);
	cudaFreeAsync(this->R, this->stream);
	cudaFreeAsync(this->devfY1, this->stream);
	cudaFreeAsync(this->devJFrozen, this->stream);
	cudaFreeAsync(this->devJ1, this->stream);
	cudaFreeAsync(this->devJ2, this->stream);
	cudaFreeAsync(this->devM, this->stream);
	cudaFreeAsync(this->dK, this->stream);
	cudaFreeAsync(this->ktrial, this->stream);

	cublasDestroy(this->cublasHandle);
}

template<size_t N>
void GaussLegendre2<N>::gaussLegendreS2Step(double* devCurrentState, double* devDestination, const double stepSize, StepResult& result)
{
	// k = np.vstack([f(y), f(y)])   # shape(2, m)
	// WIP...
	bool simplifiedUsed = false;
	bool freezeJacobian = false;
	double alpha;
	double target;
	
	double* k_local = k1;
	double* k1_local = k_local;
	double* k2_local = k_local + 3 * N;

	double* ktrial1_local = this->ktrial1;
	double* ktrial2_local = this->ktrial2;


	const dim3 blockSize(16, 16);
	const dim3 gridSize((3 * N + blockSize.x - 1) / blockSize.x, (3 * N + blockSize.y - 1) / blockSize.y);


	// initial guess for k1, k2
	problem.setStream(this->stream);
	problem.run(devCurrentState, k_local);
	cudaMemcpyAsync(k2_local, k_local, 3 * N * sizeof(double), cudaMemcpyDeviceToDevice, this->stream);
	int iteration;
	for (iteration = 0; iteration < this->options.maxNewtonIterations; ++iteration)
	{
		this->residualAndPhi(devCurrentState, k1_local, k2_local, stepSize, this->stagingRes);
		if (this->stagingRes.residualNorm <= this->options.newtonTolerance * (1.0 + this->stagingRes.normK)) 
		{
			result.converged = true;
			break;
		}

		if (freezeJacobian) 
		{
			devJ1 = devJFrozen;
			devJ2 = devJFrozen;
		}
		else 
		{
			jacobianCalculator.setStream(this->stream);
			jacobianCalculator.calculateJacobian(this->y1, this->devJ1);
			jacobianCalculator.calculateJacobian(this->y2, this->devJ2);
		}

		// solve the system
		fillMMatrix << <blockSize, gridSize, 0, this->stream >> > (this->devJ1, this->devJ2, stepSize, this->glCoeffs, this->devM, N);

		// prepare the right-hand side by multiplying R by -1
		multiplyVector << <(6 * N + 255) / 256, 256, 0, this->stream >> > (-1.0, this->R, this->R, 6 * N);
		matrixSolver.setStream(this->stream);
		matrixSolver.solve(this->devM, this->R, this->dK); // solution is stored in dK

		// Armijo backtracking line search on phi(k)
		alpha = 1.0;

		target = this->stagingRes.phi - this->options.armijo_c * alpha * std::pow(this->stagingRes.residualNorm, 2);

		while (true) 
		{
			createKTrial << <(3 * N + 255) / 256, 256, 0, this->stream >> > (k1_local, k2_local, alpha, this->dK1, this->dK2, ktrial1_local, ktrial2_local, N);
			// compute the residual and phi at the trial step
			this->residualAndPhi(devCurrentState, ktrial1_local, ktrial2_local, stepSize, this->stagingRes);

			if(this->stagingRes.phi <= target) 
			{
				// sufficient decrease achieved
				// ktrial becomes the new k
				k_local = this->ktrial;
				// the old k becomes the new ktrial to avoid overwriting and copies.
				ktrial1_local = k1_local;
				ktrial2_local = k2_local;
				// update k1_local and k2_local
				k1_local = k_local;
				k2_local = k_local + 3 * N;
				break;
			}
			// reduce alpha
			alpha *= this->options.backtrack;
			target = this->stagingRes.phi - this->options.armijo_c * alpha * std::pow(this->stagingRes.residualNorm, 2);
			if (alpha < this->options.minAlpha) 
			{
				// line search failed: switch to simplified Newton once
				if(!freezeJacobian && this->options.allowSimplifiedFallback) 
				{
					freezeJacobian = true;
					simplifiedUsed = true;
					jacobianCalculator.calculateJacobian(devCurrentState, this->devJFrozen);
					break; // break the line search and continue with frozen Jacobian
				}
				else
				{
					// give up on this step and abort
					// failed to find a suitable step size
					result.converged = false;
					result.numberIterations = iteration;
					result.residualNorm = this->stagingRes.residualNorm;
					result.simplifiedFallbackUsed = simplifiedUsed;
					return;
				}				
			}
		}
		// continue with updated k.
	}

	if (!result.converged)
	{
		// calculate residual norm for reporting
		this->residualAndPhi(devCurrentState, k1_local, k2_local, stepSize, this->stagingRes);
	}
	// calculate next y
	calculateNextStateKernel << <(3 * N + 255) / 256, 256, 0, this->stream >> > (devCurrentState, stepSize, k1_local, k2_local, devDestination, this->glCoeffs, N);
	result.numberIterations = iteration;
	result.residualNorm = this->stagingRes.residualNorm;
	result.simplifiedFallbackUsed = simplifiedUsed;
}


template<size_t N>
void GaussLegendre2<N>::stageStates(const double* devY, const double stepSize)
{
	// y1 = devY + stepSize * ( A11 * k1 + A12 * k2 ) // devY, k1 k2 are two vectors of size 3N
	// y2 = devY + stepSize * ( A21 * k1 + A22 * k2 )
	// call the kernel
	size_t threadsPerBlock = 256;
	size_t blocks = (3 * N + threadsPerBlock - 1) / threadsPerBlock;
	stageStatesKernel << <blocks, threadsPerBlock, 0, this->stream >> > (devY, stepSize, this->k1, this->k2, this->y1, this->y2, this->glCoeffs, N);
}

template<size_t N>
void GaussLegendre2<N>::residualAndPhi(const double* devY, const double* k1, const double* k2, const double stepSize, StagingResult& stagingResult)
{
	// stage the states
	this->stageStates(devY, stepSize);
	// sync the stream
	cudaStreamSynchronize(this->stream);
	// compute f(y1) and f(y2)
	problem.setStream(this->stream);
	problem.run(this->y1, this->devfY1);
	problem.run(this->y2, this->devfY2);
	
	// compute the residuals
	computeResidualsKernel << <(3 * N + 255) / 256, 256, 0, this->stream >> > (this->devfY1, this->devfY2, k1, k2, this->R1, this->R2, N);
	// compute R dot R
	cublasDdot(cublasHandle, 3*N, this->R, 1, this->R, 1, &stagingResult.phi);
	cublasDdot(cublasHandle, 6 * N, k1, 1, k1, 1, &stagingResult.normK); // important that k1 and k2 are contiguous in memory!
	
	stagingResult.residualNorm = std::sqrt(stagingResult.phi);
	stagingResult.phi *= 0.5;
}
