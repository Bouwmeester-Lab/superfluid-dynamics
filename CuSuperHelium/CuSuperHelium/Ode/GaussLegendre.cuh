#pragma once

#include "Math/GLCoefficients.hpp"
#include "Ode/JacobianProvider.h"
#include "Ode/OdeSolver.h"
#include <math.h>
#include <thrust/device_vector.h>
#include <indicators/progress_bar.hpp>
#include <cublas_v2.h>
#include "LinearAlgebra/DenseLuSolver.cuh"
#include "Ode/OdeProblem.h"
#include <format>
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
	// Newton matrix of the Gauss-Legendre stage equations G_i(k) = k_i - f(y + h sum_j A_ij k_j), i = 1, 2:
	//   M = [ I - h A11 J1,    -h A12 J1   ]      J_i = df/dy at stage state Y_i
	//       [  -h A21 J2,    I - h A22 J2  ]      (row block i uses J_i; column block j carries A_ij)
	// M is 6N x 6N, J1 and J2 are 3N x 3N, all column-major (cuSOLVER). Index row * ld + col therefore addresses
	// element (col, row), on both the read of J and the write of M: the two transposes cancel inside each block,
	// but the off-diagonal blocks trade places, so the write at memory offset (row, col + 3N) is matrix block (2,1).
	size_t row = blockIdx.y * blockDim.y + threadIdx.y;
	size_t col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < 3 * N && col < 3 * N) {
		// block (1,1): I - h A11 J1
		devM[row * (6 * N) + col] = (row == col ? 1.0 : 0.0) - stepSize * coeffs.A11 * devJ1[row * (3 * N) + col];
		// block (2,1): -h A21 J2
		devM[row * (6 * N) + (col + 3 * N)] = -stepSize * coeffs.A21 * devJ2[row * (3 * N) + col];
		// block (1,2): -h A12 J1
		devM[(row + 3 * N) * (6 * N) + col] = -stepSize * coeffs.A12 * devJ1[row * (3 * N) + col];
		// block (2,2): I - h A22 J2
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
	double armijo_c = 1e-4;
	double backtrack = 0.5;
	double minAlpha = 1e-6;
	size_t maxStepsHalves = 6;

	GaussLegendre2Options() :
		stepSize(0.01),
		newtonTolerance(1e-10),
		maxNewtonIterations(20),
		allowSimplifiedFallback(false)
	{
	}
};

inline size_t to_percent(double t, double t0, double t1)
{
	if (t <= t0) return 0;
	if (t >= t1) return 100;
	double frac = (t - t0) / (t1 - t0);
	return static_cast<std::size_t>(std::round(frac * 100.0));
};


class GaussLegendre2 : public OdeSolver<double>
{
public:
	/// jacobianCalculator supplies df/dy: FiniteDifferenceJacobian for any problem, or a problem-specific provider.
	GaussLegendre2(OdeProblem<double>& problem, JacobianProvider& jacobianCalculator, GaussLegendre2Options options = GaussLegendre2Options());
	~GaussLegendre2();
	virtual OdeSolverResult runEvolution(double startTime, double endTime) override;
	/// <summary>
	/// Sets the initial state for the solver.
	/// </summary>
	/// <param name="initialState"></param>
	/// <param name="onDevice"></param>
	void initialize(const double* initialState, bool onDevice = false) override;
	const double* state() const override { return devTempState; }
	double time() const override { return currentTime; }
	void setStream(cudaStream_t stream) { this->stream = stream; }

private:
	const size_t N; ///< Collocation points; the state has 3N real values (x, y, phi)
	OdeProblem<double>& problem;
	JacobianProvider& jacobianCalculator;
	DenseLuSolver matrixSolver{ 6 * static_cast<int>(N) };
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
	double currentTime = 0.0;

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
	void stageStates(const double* devY, const double* k1, const double* k2, const double stepSize);
	void residualAndPhi(const double* devY, const double* k1, const double* k2, const double stepSize, StagingResult& stagingResult);
	void freeDeviceMemory();
};

inline GaussLegendre2::GaussLegendre2(OdeProblem<double>& problem, JacobianProvider& jacobianCalculator, GaussLegendre2Options options) : N(problem.size() / 3), problem(problem), jacobianCalculator(jacobianCalculator), options(options)
{
	if (problem.size() != 3 * N)
	{
		throw std::invalid_argument("GaussLegendre2: the problem size does not match 3 N");
	}
}
inline GaussLegendre2::~GaussLegendre2()
{
	// free device memory
	freeDeviceMemory();
}

inline OdeSolverResult GaussLegendre2::runEvolution(double startTime, double endTime)
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
	CUSH_CHECK(cudaMemcpyAsync(this->devTempState, this->devState, 3 * N * sizeof(double), cudaMemcpyDeviceToDevice, this->stream));

	problem.start(startTime);
	this->notifyStart(startTime, this->devTempState, 3 * N, this->stream);

	currentTime = startTime;
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
		CUSH_CHECK(cudaMemcpyAsync(this->devTempState, this->devTempNextState, 3 * N * sizeof(double), cudaMemcpyDeviceToDevice, this->stream));
		percent = to_percent(currentTime, startTime, endTime);
		if(percent != previousPercent) {
			progressBar->set_progress(percent);
			// std::cout << percent << "% completed.\n";
			previousPercent = percent;
		}

		//progressBar->set_progress();
		currentTime += htry;
		this->options.stepSize = std::abs(htry); // Update step size for next iteration

		this->notifyAcceptedStep(currentTime, this->devTempState, 3 * N, this->stream);
	}
	indicators::show_console_cursor(true);
	return OdeSolverResult::ReachedEndTime;
}

inline void GaussLegendre2::initialize(const double* initialState, bool onDevice)
{
	size_t stateSizeBytes = 3 * N * sizeof(double);
	// The solver always owns its copy of the initial state.
	devStateOwned = true;
	CUSH_CHECK(cudaMalloc(&this->devState, stateSizeBytes));
	CUSH_CHECK(cudaMemcpy(this->devState, initialState, stateSizeBytes, onDevice ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice));

	CUSH_CHECK(cudaMallocAsync(&this->devTempState, stateSizeBytes, this->stream));
	CUSH_CHECK(cudaMallocAsync(&this->devTempNextState, stateSizeBytes, this->stream));

	// Also allocate k1, k2, y1, y2
	CUSH_CHECK(cudaMallocAsync(&this->k1, 2 * stateSizeBytes, this->stream));
	this->k2 = this->k1 + 3 * N;

	CUSH_CHECK(cudaMallocAsync(&this->y1, 2 * stateSizeBytes, this->stream));
	this->y2 = this->y1 + 3 * N;

	CUSH_CHECK(cudaMallocAsync(&this->R, 2 * stateSizeBytes, this->stream));
	this->R1 = this->R;
	this->R2 = this->R + 3 * N;

	CUSH_CHECK(cudaMallocAsync(&this->devfY1, 2 * stateSizeBytes, this->stream));
	this->devfY2 = this->devfY1 + 3 * N;

	CUSH_CHECK(cublasCreate(&this->cublasHandle));

	// allocate the space for the jacobians
	CUSH_CHECK(cudaMallocAsync(&this->devJFrozen, (3 * N) * (3 * N) * sizeof(double), this->stream));
	CUSH_CHECK(cudaMallocAsync(&this->devJ1, (3 * N) * (3 * N) * sizeof(double), this->stream));
	CUSH_CHECK(cudaMallocAsync(&this->devJ2, (3 * N) * (3 * N) * sizeof(double), this->stream));

	CUSH_CHECK(cudaMallocAsync(&this->devM, 4 * (3 * N) * (3 * N) * sizeof(double), this->stream));
	CUSH_CHECK(cudaMallocAsync(&this->dK, 2 * stateSizeBytes, this->stream));
	this->dK1 = this->dK;
	this->dK2 = this->dK + 3 * N;

	CUSH_CHECK(cudaMallocAsync(&this->ktrial, 2 * stateSizeBytes, this->stream));
	this->ktrial1 = this->ktrial;
	this->ktrial2 = this->ktrial + 3 * N;
}

inline void GaussLegendre2::freeDeviceMemory()
{
	if (this->devStateOwned) {
		CUSH_CHECK(cudaFreeAsync(this->devState, this->stream));
	}
	CUSH_CHECK(cudaFreeAsync(this->devTempState, this->stream));
	CUSH_CHECK(cudaFreeAsync(this->devTempNextState, this->stream));
	CUSH_CHECK(cudaFreeAsync(this->k1, this->stream));
	CUSH_CHECK(cudaFreeAsync(this->y1, this->stream));
	CUSH_CHECK(cudaFreeAsync(this->R, this->stream));
	CUSH_CHECK(cudaFreeAsync(this->devfY1, this->stream));
	CUSH_CHECK(cudaFreeAsync(this->devJFrozen, this->stream));
	CUSH_CHECK(cudaFreeAsync(this->devJ1, this->stream));
	CUSH_CHECK(cudaFreeAsync(this->devJ2, this->stream));
	CUSH_CHECK(cudaFreeAsync(this->devM, this->stream));
	CUSH_CHECK(cudaFreeAsync(this->dK, this->stream));
	CUSH_CHECK(cudaFreeAsync(this->ktrial, this->stream));

	CUSH_CHECK(cublasDestroy(this->cublasHandle));
}

inline void GaussLegendre2::gaussLegendreS2Step(double* devCurrentState, double* devDestination, const double stepSize, StepResult& result)
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
	problem.rhs(devCurrentState, currentTime, k_local);
	CUSH_CHECK(cudaMemcpyAsync(k2_local, k_local, 3 * N * sizeof(double), cudaMemcpyDeviceToDevice, this->stream));
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
		fillMMatrix << <gridSize, blockSize, 0, this->stream >> > (this->devJ1, this->devJ2, stepSize, this->glCoeffs, this->devM, N);
		CUSH_CHECK_LAUNCH();

		// prepare the right-hand side by multiplying R by -1
		multiplyVector << <(6 * N + 255) / 256, 256, 0, this->stream >> > (-1.0, this->R, this->R, 6 * N);
		CUSH_CHECK_LAUNCH();
		matrixSolver.setStream(this->stream);
		matrixSolver.solve(this->devM, this->R, this->dK); // solution is stored in dK

		// Armijo backtracking line search on phi(k)
		alpha = 1.0;

		target = this->stagingRes.phi - this->options.armijo_c * alpha * std::pow(this->stagingRes.residualNorm, 2);

		while (true) 
		{
			createKTrial << <(3 * N + 255) / 256, 256, 0, this->stream >> > (k1_local, k2_local, alpha, this->dK1, this->dK2, ktrial1_local, ktrial2_local, N);
			CUSH_CHECK_LAUNCH();
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
	CUSH_CHECK_LAUNCH();
	result.numberIterations = iteration;
	result.residualNorm = this->stagingRes.residualNorm;
	result.simplifiedFallbackUsed = simplifiedUsed;
}


inline void GaussLegendre2::stageStates(const double* devY, const double* k1, const double* k2, const double stepSize)
{
	// y1 = devY + stepSize * ( A11 * k1 + A12 * k2 ) // devY, k1 k2 are two vectors of size 3N
	// y2 = devY + stepSize * ( A21 * k1 + A22 * k2 )
	// call the kernel
	size_t threadsPerBlock = 256;
	size_t blocks = (3 * N + threadsPerBlock - 1) / threadsPerBlock;
	stageStatesKernel << <blocks, threadsPerBlock, 0, this->stream >> > (devY, stepSize, k1, k2, this->y1, this->y2, this->glCoeffs, N);
	CUSH_CHECK_LAUNCH();
}

inline void GaussLegendre2::residualAndPhi(const double* devY, const double* k1, const double* k2, const double stepSize, StagingResult& stagingResult)
{
	// stage the states
	this->stageStates(devY, k1, k2, stepSize);
	// sync the stream
	CUSH_CHECK(cudaStreamSynchronize(this->stream));
	// compute f(y1) and f(y2)
	problem.setStream(this->stream);
	problem.rhs(this->y1, currentTime + glCoeffs.c1 * stepSize, this->devfY1);
	problem.rhs(this->y2, currentTime + glCoeffs.c2 * stepSize, this->devfY2);
	
	// compute the residuals
	computeResidualsKernel << <(3 * N + 255) / 256, 256, 0, this->stream >> > (this->devfY1, this->devfY2, k1, k2, this->R1, this->R2, N);
	CUSH_CHECK_LAUNCH();
	// compute R dot R
	CUSH_CHECK(cublasDdot(cublasHandle, 6 * N, this->R, 1, this->R, 1, &stagingResult.phi)); // both stages: R1 and R2 are contiguous
	CUSH_CHECK(cublasDdot(cublasHandle, 6 * N, k1, 1, k1, 1, &stagingResult.normK)); // important that k1 and k2 are contiguous in memory!
	
	stagingResult.residualNorm = std::sqrt(stagingResult.phi);
	stagingResult.phi *= 0.5;
}
