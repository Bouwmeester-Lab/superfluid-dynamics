#include "Export/ExportInternal.cuh"

#include "Ode/AutonomousProblem.h"
#include "BoundaryIntegral/Periodic2D/BaseBoundaryIntegrator.cuh"
#include "Physics/HeliumBoundaryProblem.cuh"
#include "Ode/JacobianCalculator.cuh"

#include <chrono>
#include <memory>

struct JacobianWorkspace
{
	double* devState;
	double* devJac;
};

template <size_t N>
int calculateJacobian(const double* state, double* jac, SimProperties* simProperties, double epsilon)
{
	using std::chrono::high_resolution_clock;
	using std::chrono::duration_cast;
	using std::chrono::duration;
	using std::chrono::milliseconds;
	auto t1 = high_resolution_clock::now();
	try {
		DimensionlessParameters properties = toDimensionlessParameters(*simProperties);

		//CUSH_CHECK(setDevice());

		DeviceBuffer<double> devState(3 * N);
		DeviceBuffer<double> devJac(9 * N * N);
		CUSH_CHECK(cudaDeviceSynchronize());
	 	CUSH_CHECK(cudaGetLastError());

		CUSH_CHECK(cudaMemcpyAsync(devState.get(), state, devState.bytes(), cudaMemcpyHostToDevice, cudaStreamPerThread));

		{
			HeliumBoundaryProblem<N, 3 * N> heliumProblem(properties);
			std::unique_ptr<AutonomousProblem<std_complex, 6 * N * N>> boundaryIntegralCalculatorPtr = std::make_unique<BaseBoundaryIntegralCalculator<N, 3 * N>>(properties, heliumProblem);

			JacobianCalculator<N> jacobianCalculator(std::move(boundaryIntegralCalculatorPtr));

			jacobianCalculator.setEpsilon(epsilon);
			jacobianCalculator.setStream(cudaStreamPerThread);
			jacobianCalculator.calculateJacobian(devState.get(), devJac.get());

			CUSH_CHECK(cudaGetLastError());
			CUSH_CHECK(cudaDeviceSynchronize());
		}
			CUSH_CHECK(cudaGetLastError());
		
		// copy back to host
		CUSH_CHECK(cudaMemcpy(jac, devJac.get(), devJac.bytes(), cudaMemcpyDeviceToHost));
		// make sure we have finished before returning
		CUSH_CHECK(cudaGetLastError());

		// free device memory
		devState.reset();
		devJac.reset();
		CUSH_CHECK(cudaGetLastError());

		auto t2 = high_resolution_clock::now();
		duration<double, std::milli> ms_double = t2 - t1;
		// std::cout << ms_double.count() << std::endl;

		CUSH_CHECK(cudaGetLastError());

		return 0;
	}
	catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return -1;
	}
}

int calculateJacobian(const double* state, double* jac, SimProperties* simProperties, double epsilon, size_t N)
{
	switch (N)
	{
		case 32:
			return calculateJacobian<32>(state, jac, simProperties, epsilon);
		case 64:
			return calculateJacobian<64>(state, jac, simProperties, epsilon);
		case 128:
			return calculateJacobian<128>(state, jac, simProperties, epsilon);
		case 256:
			return calculateJacobian<256>(state, jac, simProperties, epsilon);
		case 512:
			return calculateJacobian<512>(state, jac, simProperties, epsilon);
		case 1024:
			return calculateJacobian<1024>(state, jac, simProperties, epsilon);
		case 2048:
			return calculateJacobian<2048>(state, jac, simProperties, epsilon);
	default:
		std::cerr << "Error: Unsupported N size " << N << std::endl;
		std::cerr << "Supported N sizes are: 32, 64, 128, 256, 512, 1024, 2048" << std::endl;
		return -1;
	}
}

int calculatePerturbedStates256(const double* x, const double* y, const double* phi, c_double* Zperturbed, SimProperties* simProperties, double epsilon)
{
	try {
		const size_t N = 256;
		const int blockSize = N;
		const int numBlocks = (N + blockSize - 1) / blockSize;

		const dim3 threads(256, 1, 1);
		const dim3 blocks((2 * 256 + 255) / 256, 3 * N, 1);

		CUSH_CHECK(setDevice());

		DeviceBuffer<double> devState(3 * N);
		DeviceBuffer<std_complex> initialState(2 * N);
		DeviceBuffer<std_complex> devZPhiBatched(6 * N * N);

		// copy the initial data to the devState
		CUSH_CHECK(cudaMemcpy(devState.get(), x, sizeof(double) * N, cudaMemcpyHostToDevice));
		CUSH_CHECK(cudaMemcpy(devState.get() + N, y, sizeof(double) * N, cudaMemcpyHostToDevice));
		CUSH_CHECK(cudaMemcpy(devState.get() + 2 * N, phi, sizeof(double) * N, cudaMemcpyHostToDevice));

		createInitialState <<<N, numBlocks >> > (devState.get(), initialState.get(), N);
		CUSH_CHECK_LAUNCH();

		std::cout << "Initial state created on device." << std::endl;
		std::cout << "Threads: " << threads.x << "y: " << threads.y << "z: " << threads.z << std::endl;
		std::cout << "Blocks : x:" << blocks.x << "y :" << blocks.y << " z: " << blocks.z << std::endl;
		createInitialBatchedZ<<<blocks, threads >>> (initialState.get(), devZPhiBatched.get(), epsilon, N);
		CUSH_CHECK_LAUNCH();
		std::cout << "Perturbed state created on device." << std::endl;
		std::cout << "N: " << N << std::endl;
		std::cout << "Output address: " << Zperturbed << std::endl;
		// make sure it's all done
		CUSH_CHECK(cudaDeviceSynchronize());

		// copy to host for saving
		CUSH_CHECK(cudaMemcpy(Zperturbed, devZPhiBatched.get(), devZPhiBatched.bytes(), cudaMemcpyDeviceToHost));

		std::cout << "Perturbed states copied to host." << std::endl;

		CUSH_CHECK(cudaDeviceSynchronize());

		return 0;
	}
	catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return -1;
	}	
}
