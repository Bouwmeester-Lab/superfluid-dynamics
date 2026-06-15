#include "ExportInternal.cuh"

#include "../AutonomousProblem.h"
#include "../BaseBoundaryIntegrator.cuh"
#include "../HeliumBoundaryProblem.cuh"
#include "../JacobianCalculator.cuh"

#include <chrono>
#include <memory>

struct JacobianWorkspace
{
	double* devState;
	double* devJac;
};

template <size_t N>
int calculateJacobian(const double* state, double* jac, double L, double rho, double kappa, double depth, double epsilon)
{
	using std::chrono::high_resolution_clock;
	using std::chrono::duration_cast;
	using std::chrono::duration;
	using std::chrono::milliseconds;
	auto t1 = high_resolution_clock::now();
	try {
		ProblemProperties properties;
		properties.rho = rho;
		properties.kappa = kappa;
		properties.depth = depth;
		properties.L = L;

		// adimensionalize properties
		properties = adimensionalizeProperties(properties);

		double* devState;
		double* devJac;

		//checkCuda(setDevice());
		
		checkCuda(cudaMallocAsync(&devState, sizeof(double) * 3 * N, cudaStreamPerThread));
		checkCuda(cudaMallocAsync(&devJac, sizeof(double) * 9 * N * N, cudaStreamPerThread));
		checkCuda(cudaDeviceSynchronize());
	 	auto error = cudaGetLastError();
		if (error != cudaSuccess) {
			std::cerr << "CUDA after trying to allocate error: " << cudaGetErrorString(error) << std::endl;
		}
		
		checkCuda(cudaMemcpyAsync(devState, state, sizeof(double) * 3 * N, cudaMemcpyHostToDevice, cudaStreamPerThread));
		
		{
			HeliumBoundaryProblem<N, 3 * N> heliumProblem(properties);
			std::unique_ptr<AutonomousProblem<std_complex, 6 * N * N>> boundaryIntegralCalculatorPtr = std::make_unique<BaseBoundaryIntegralCalculator<N, 3 * N>>(properties, heliumProblem);

			JacobianCalculator<N> jacobianCalculator(std::move(boundaryIntegralCalculatorPtr));

			jacobianCalculator.setEpsilon(epsilon);
			jacobianCalculator.setStream(cudaStreamPerThread);
			jacobianCalculator.calculateJacobian(devState, devJac);

			error = cudaGetLastError();
			if (error != cudaSuccess) {

				std::cerr << "CUDA error after calculating Jacobian: " << cudaGetErrorString(error) << std::endl;
			}
			cudaDeviceSynchronize();
		}
			error = cudaGetLastError();
			if (error != cudaSuccess) {

				std::cerr << "CUDA error after scope ending: " << cudaGetErrorString(error) << std::endl;
			}
		
		// copy back to host
		checkCuda(cudaMemcpy(jac, devJac, sizeof(double) * 9 * N * N, cudaMemcpyDeviceToHost));
		// make sure we have finished before returning
		error = cudaGetLastError();
		if (error != cudaSuccess) {
			std::cerr << "jac " << jac << std::endl;
			std::cerr << "devJac " << devJac << std::endl;
			std::cerr << "CUDA error after copying: " << cudaGetErrorString(error) << std::endl;
		}

		// free device memory
		checkCuda(cudaFree(devState));
		error = cudaGetLastError();
		if (error != cudaSuccess) {
			std::cerr << "devState " << devState << std::endl;
			std::cerr << "CUDA error after freeing: " << cudaGetErrorString(error) << std::endl;
		}
		checkCuda(cudaFree(devJac));
		error = cudaGetLastError();
		if (error != cudaSuccess) {
			std::cerr << "CUDA error before freeing devJac: " << cudaGetErrorString(error) << std::endl;
		}

		auto t2 = high_resolution_clock::now();
		duration<double, std::milli> ms_double = t2 - t1;
		// std::cout << ms_double.count() << std::endl;

		error = cudaGetLastError();
		if (error != cudaSuccess) {
			std::cerr << "CUDA error before returning: " << cudaGetErrorString(error) << std::endl;
		}

		return 0;
	}
	catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return -1;
	}
}

int calculateJacobian(const double* state, double* jac, double L, double rho, double kappa, double depth, double epsilon, size_t N)
{
	switch (N)
	{
		case 32:
			return calculateJacobian<32>(state, jac, L, rho, kappa, depth, epsilon);
		case 64:
			return calculateJacobian<64>(state, jac, L, rho, kappa, depth, epsilon);
		case 128:
			return calculateJacobian<128>(state, jac, L, rho, kappa, depth, epsilon);
		case 256:
			return calculateJacobian<256>(state, jac, L, rho, kappa, depth, epsilon);
		case 512:
			return calculateJacobian<512>(state, jac, L, rho, kappa, depth, epsilon);
		case 1024:
			return calculateJacobian<1024>(state, jac, L, rho, kappa, depth, epsilon);
		case 2048:
			return calculateJacobian<2048>(state, jac, L, rho, kappa, depth, epsilon);
	default:
		std::cerr << "Error: Unsupported N size " << N << std::endl;
		std::cerr << "Supported N sizes are: 32, 64, 128, 256, 512, 1024, 2048" << std::endl;
		break;
	}
}

int calculatePerturbedStates256(const double* x, const double* y, const double* phi, c_double* Zperturbed, double L, double rho, double kappa, double depth, double epsilon)
{
	try {
		const size_t N = 256;
		const int blockSize = N;
		const int numBlocks = (N + blockSize - 1) / blockSize;

		const dim3 threads(256, 1, 1);
		const dim3 blocks((2 * 256 + 255) / 256, 3 * N, 1);

		std_complex* initialState;
		std_complex* devZPhiBatched;
		double* devState;

		checkCuda(setDevice());

		checkCuda(cudaMalloc(&devState, sizeof(double) * 3 * N));
		checkCuda(cudaMalloc(&initialState, sizeof(std_complex) * 2 * N));
		checkCuda(cudaMalloc(&devZPhiBatched, sizeof(std_complex) * 6 * N * N));

		// copy the initial data to the devState
		checkCuda(cudaMemcpy(devState, x, sizeof(double) * N, cudaMemcpyHostToDevice));
		checkCuda(cudaMemcpy((devState + N), y, sizeof(double) * N, cudaMemcpyHostToDevice));
		checkCuda(cudaMemcpy((devState + 2 * N), phi, sizeof(double) * N, cudaMemcpyHostToDevice));

		createInitialState <<<N, numBlocks >> > (devState, initialState, N);

		std::cout << "Initial state created on device." << std::endl;
		std::cout << "Threads: " << threads.x << "y: " << threads.y << "z: " << threads.z << std::endl;
		std::cout << "Blocks : x:" << blocks.x << "y :" << blocks.y << " z: " << blocks.z << std::endl;
		createInitialBatchedZ<<<blocks, threads >>> (initialState, devZPhiBatched, epsilon, N);
		std::cout << "Perturbed state created on device." << std::endl;
		std::cout << "N: " << N << std::endl;
		std::cout << "Output address: " << Zperturbed << std::endl;
		// make sure it's all done
		cudaDeviceSynchronize();

		// copy to host for saving
		checkCuda(cudaMemcpy(Zperturbed, devZPhiBatched, sizeof(std_complex) * 6 * N * N , cudaMemcpyDeviceToHost));

		std::cout << "Perturbed states copied to host." << std::endl;

		cudaDeviceSynchronize();

		// free device memory
		checkCuda(cudaFree(devState));
		checkCuda(cudaFree(initialState));
		checkCuda(cudaFree(devZPhiBatched));

		return 0;
	}
	catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return -1;
	}	
}
