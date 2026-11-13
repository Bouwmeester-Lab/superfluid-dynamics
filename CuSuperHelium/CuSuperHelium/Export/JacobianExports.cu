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

		//checkCuda(setDevice());

		DeviceBuffer<double> devState(3 * N);
		DeviceBuffer<double> devJac(9 * N * N);
		CHECK_CUDA(cudaDeviceSynchronize());
	 	auto error = cudaGetLastError();
		if (error != cudaSuccess) {
			std::cerr << "CUDA after trying to allocate error: " << cudaGetErrorString(error) << std::endl;
		}

		CHECK_CUDA(cudaMemcpyAsync(devState.get(), state, devState.bytes(), cudaMemcpyHostToDevice, cudaStreamPerThread));

		{
			HeliumBoundaryProblem<N, 3 * N> heliumProblem(properties);
			std::unique_ptr<AutonomousProblem<std_complex, 6 * N * N>> boundaryIntegralCalculatorPtr = std::make_unique<BaseBoundaryIntegralCalculator<N, 3 * N>>(properties, heliumProblem);

			JacobianCalculator<N> jacobianCalculator(std::move(boundaryIntegralCalculatorPtr));

			jacobianCalculator.setEpsilon(epsilon);
			jacobianCalculator.setStream(cudaStreamPerThread);
			jacobianCalculator.calculateJacobian(devState.get(), devJac.get());

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
		CHECK_CUDA(cudaMemcpy(jac, devJac.get(), devJac.bytes(), cudaMemcpyDeviceToHost));
		// make sure we have finished before returning
		error = cudaGetLastError();
		if (error != cudaSuccess) {
			std::cerr << "CUDA error after copying: " << cudaGetErrorString(error) << std::endl;
		}

		// free device memory
		devState.reset();
		devJac.reset();
		error = cudaGetLastError();
		if (error != cudaSuccess) {
			std::cerr << "CUDA error after freeing: " << cudaGetErrorString(error) << std::endl;
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
		return -1;
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

		CHECK_CUDA(setDevice());

		DeviceBuffer<double> devState(3 * N);
		DeviceBuffer<std_complex> initialState(2 * N);
		DeviceBuffer<std_complex> devZPhiBatched(6 * N * N);

		// copy the initial data to the devState
		CHECK_CUDA(cudaMemcpy(devState.get(), x, sizeof(double) * N, cudaMemcpyHostToDevice));
		CHECK_CUDA(cudaMemcpy(devState.get() + N, y, sizeof(double) * N, cudaMemcpyHostToDevice));
		CHECK_CUDA(cudaMemcpy(devState.get() + 2 * N, phi, sizeof(double) * N, cudaMemcpyHostToDevice));

		createInitialState <<<N, numBlocks >> > (devState.get(), initialState.get(), N);

		std::cout << "Initial state created on device." << std::endl;
		std::cout << "Threads: " << threads.x << "y: " << threads.y << "z: " << threads.z << std::endl;
		std::cout << "Blocks : x:" << blocks.x << "y :" << blocks.y << " z: " << blocks.z << std::endl;
		createInitialBatchedZ<<<blocks, threads >>> (initialState.get(), devZPhiBatched.get(), epsilon, N);
		std::cout << "Perturbed state created on device." << std::endl;
		std::cout << "N: " << N << std::endl;
		std::cout << "Output address: " << Zperturbed << std::endl;
		// make sure it's all done
		cudaDeviceSynchronize();

		// copy to host for saving
		CHECK_CUDA(cudaMemcpy(Zperturbed, devZPhiBatched.get(), devZPhiBatched.bytes(), cudaMemcpyDeviceToHost));

		std::cout << "Perturbed states copied to host." << std::endl;

		cudaDeviceSynchronize();

		return 0;
	}
	catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return -1;
	}	
}
