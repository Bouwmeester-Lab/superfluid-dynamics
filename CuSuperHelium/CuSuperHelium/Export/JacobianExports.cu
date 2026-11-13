#include "Export/ExportInternal.cuh"

#include "Ode/AutonomousProblem.h"
#include "BoundaryIntegral/Periodic2D/BaseBoundaryIntegrator.cuh"
#include "Physics/HeliumBoundaryProblem.cuh"
#include "Ode/JacobianCalculator.cuh"

#include <memory>

int calculateJacobian(const double* state, double* jac, SimProperties* simProperties, double epsilon, size_t N)
{
	if (!validPeriodicSize(N)) return -1;
	try {
		DimensionlessParameters properties = toDimensionlessParameters(*simProperties);

		DeviceBuffer<double> devState(3 * N);
		DeviceBuffer<double> devJac(9 * N * N);
		CUSH_CHECK(cudaDeviceSynchronize());
		CUSH_CHECK(cudaGetLastError());

		CUSH_CHECK(cudaMemcpyAsync(devState.get(), state, devState.bytes(), cudaMemcpyHostToDevice, cudaStreamPerThread));

		{
			// The Jacobian evaluates the RHS of 3N perturbed copies of the state at once: a batch of 3N interfaces.
			HeliumBoundaryProblem heliumProblem(properties, static_cast<int>(N), 3 * N);
			std::unique_ptr<AutonomousProblem<std_complex>> boundaryIntegralCalculatorPtr = std::make_unique<BaseBoundaryIntegralCalculator>(properties, heliumProblem);

			JacobianCalculator jacobianCalculator(std::move(boundaryIntegralCalculatorPtr), N);

			jacobianCalculator.setEpsilon(epsilon);
			jacobianCalculator.setStream(cudaStreamPerThread);
			jacobianCalculator.calculateJacobian(devState.get(), devJac.get());

			CUSH_CHECK(cudaGetLastError());
			CUSH_CHECK(cudaDeviceSynchronize());
		}
		CUSH_CHECK(cudaGetLastError());

		// copy back to host
		CUSH_CHECK(cudaMemcpy(jac, devJac.get(), devJac.bytes(), cudaMemcpyDeviceToHost));
		CUSH_CHECK(cudaGetLastError());
		return 0;
	}
	catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return -1;
	}
}

int calculatePerturbedStates(const double* x, const double* y, const double* phi, c_double* Zperturbed, SimProperties* simProperties, double epsilon, size_t N)
{
	if (!validPeriodicSize(N)) return -1;
	try {
		const dim3 threads(256, 1, 1);
		const dim3 blocks(static_cast<unsigned int>((2 * N + 255) / 256), static_cast<unsigned int>(3 * N), 1);

		CUSH_CHECK(setDevice());

		DeviceBuffer<double> devState(3 * N);
		DeviceBuffer<std_complex> initialState(2 * N);
		DeviceBuffer<std_complex> devZPhiBatched(6 * N * N);

		// copy the initial data to the devState
		CUSH_CHECK(cudaMemcpy(devState.get(), x, sizeof(double) * N, cudaMemcpyHostToDevice));
		CUSH_CHECK(cudaMemcpy(devState.get() + N, y, sizeof(double) * N, cudaMemcpyHostToDevice));
		CUSH_CHECK(cudaMemcpy(devState.get() + 2 * N, phi, sizeof(double) * N, cudaMemcpyHostToDevice));

		createInitialState <<<static_cast<unsigned int>(N), 1>>> (devState.get(), initialState.get(), N);
		CUSH_CHECK_LAUNCH();

		createInitialBatchedZ<<<blocks, threads >>> (initialState.get(), devZPhiBatched.get(), epsilon, N);
		CUSH_CHECK_LAUNCH();
		CUSH_CHECK(cudaDeviceSynchronize());

		// copy to host for saving
		CUSH_CHECK(cudaMemcpy(Zperturbed, devZPhiBatched.get(), devZPhiBatched.bytes(), cudaMemcpyDeviceToHost));
		CUSH_CHECK(cudaDeviceSynchronize());

		return 0;
	}
	catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return -1;
	}
}
