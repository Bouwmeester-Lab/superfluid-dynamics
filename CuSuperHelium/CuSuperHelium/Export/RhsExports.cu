#include "Export/ExportInternal.cuh"

#include "BoundaryIntegral/Periodic2D/BaseBoundaryIntegrator.cuh"
#include "Math/Derivatives.cuh"
#include "Physics/HeliumBoundaryProblem.cuh"
#include "Export/DeviceParticleLoading.cuh"
#include "IO/StateFile.h"
#include "Math/DerivativeCalculator.cuh"

#include <algorithm>
#include <complex>
#include <string>
#include <vector>

int calculateRHSFromFile(const char* inputFile, const char* outputFile, SimProperties* simProperties, size_t N)
{
	if (!validPeriodicSize(N)) return -1;
	try {
		DimensionlessParameters properties = toDimensionlessParameters(*simProperties);
		const int n = static_cast<int>(N);

		HeliumBoundaryProblem heliumProblem(properties, n);
		BaseBoundaryIntegralCalculator calculator(properties, heliumProblem);

		std::vector<std::complex<double>> Z;
		std::vector<double> Phi;

		loadStateFile(std::string(inputFile), Z, Phi, n, 2.0*PI_d / simProperties->L);

		auto [min_it, max_it] = std::minmax_element(
			Z.begin(), Z.end(),
			[](const auto& a, const auto& b) {
				return a.imag() < b.imag();
			}
		);

		properties.initial_amplitude = (max_it->imag() - min_it->imag()) / 2.0;

		ParticleData particle(Z, Phi);
		DeviceParticleData deviceData;

		// set device
		CUSH_CHECK(setDevice());
		// copy data to device
		loadDataToDevice(particle, deviceData, n);
		// allocate memory for RHS
		DeviceBuffer<std_complex> devRhs(2 * N);

		// calculate RHS
		calculator.run(deviceData.devZ(), devRhs.get());

		// copy result back to host
		std::vector<std::complex<double>> rhs(2 * N);

		std::vector<std::complex<double>> rhsPos(N);
		std::vector<double> phiRhs(N);

		CUSH_CHECK(cudaMemcpy(rhs.data(), devRhs.get(), devRhs.bytes(), cudaMemcpyDeviceToHost));

		// free device memory
		devRhs.reset();
		freeDeviceData(deviceData);

		// prepare data into two vectors, one for position RHS and one for potential RHS
		for (size_t i = 0; i < N; i++)
		{
			rhsPos[i] = rhs[i];
			phiRhs[i] = rhs[i + N].real(); // imaginary part should be zero
		}
		// save to file

		const std::map<std::string, double> attributes{ { "rho", properties.rho }, { "kappa", properties.kappa }, { "U", properties.U },
			{ "depth", properties.depth }, { "initial_amplitude", properties.initial_amplitude } };
		saveStateFile(std::string(outputFile), rhsPos, phiRhs, n, attributes, true);
	}
	catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return -1;
	}

	return 0;
}

/// RHS of batchSize independent interfaces of N points each, stored back to back.
static int calculateRHSBatched(const double* x, const double* y, const double* phi, double* vx, double* vy, double* rhsPhi, SimProperties* simProperties, size_t N, size_t batchSize)
{
	if (!validPeriodicSize(N)) return -1;
	if (batchSize == 0)
	{
		std::cerr << "Error: the batch size must be at least 1" << std::endl;
		return -1;
	}
	try {
		DimensionlessParameters properties = toDimensionlessParameters(*simProperties);
		HeliumBoundaryProblem heliumProblem(properties, static_cast<int>(N), batchSize);
		BaseBoundaryIntegralCalculator calculator(properties, heliumProblem);

		DeviceParticleData deviceData;

		// set device
		CUSH_CHECK(setDevice());
		// copy data to device
		loadDataToDevice(x, y, phi, deviceData, N, batchSize);
		// allocate memory for RHS
		DeviceBuffer<std_complex> devRhs(2 * N * batchSize);

		// calculate RHS
		calculator.run(deviceData.devZ(), devRhs.get());

		// copy result back to host
		std::vector<std::complex<double>> rhs(2 * N * batchSize);

		CUSH_CHECK(cudaMemcpy(rhs.data(), devRhs.get(), devRhs.bytes(), cudaMemcpyDeviceToHost));

		// free device memory
		devRhs.reset();
		freeDeviceData(deviceData);

		// prepare data into two vectors, one for position RHS and one for potential RHS
		for (size_t i = 0; i < batchSize * N; i++)
		{
			vx[i] = rhs[i].real();
			vy[i] = rhs[i].imag();
			rhsPhi[i] = rhs[i + batchSize * N].real(); // imaginary part should be zero
		}
	}
	catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return -1;
	}
	return 0;
}

int calculateRHSFromVectors(const double* x, const double* y, const double* phi, double* vx, double* vy, double* rhsPhi, SimProperties* simProperties, size_t N)
{
	return calculateRHSBatched(x, y, phi, vx, vy, rhsPhi, simProperties, N, 1);
}

int calculateRHSFromVectorsBatched(const double* x, const double* y, const double* phi, double* vx, double* vy, double* rhsPhi, SimProperties* simProperties, size_t N, size_t batchSize)
{
	return calculateRHSBatched(x, y, phi, vx, vy, rhsPhi, simProperties, N, batchSize);
}

int calculateVorticitiesFromVectors(const c_double* Z, const c_double* phi, double* a, c_double* Zp, c_double* Zpp, SimProperties* simProperties, size_t N)
{
	if (!validPeriodicSize(N)) return -1;
	try {
		DimensionlessParameters properties = toDimensionlessParameters(*simProperties);

		HeliumBoundaryProblem heliumProblem(properties, static_cast<int>(N));
		BaseBoundaryIntegralCalculator calculator(properties, heliumProblem);

		DeviceParticleData deviceData;

		// set device
		CUSH_CHECK(setDevice());
		// copy data to device
		loadDataToDevice(Z, phi, deviceData, static_cast<int>(N));

		// calculate RHS
		calculator.calculateVorticities(deviceData.devZ());

		CUSH_CHECK(cudaMemcpy(a, calculator.getDevA(), sizeof(double) * N, cudaMemcpyDeviceToHost));
		if(Zp != nullptr) {
			CUSH_CHECK(cudaMemcpy(Zp, calculator.getDevZp(), sizeof(c_double) * N, cudaMemcpyDeviceToHost));
		}
		if(Zpp != nullptr) {
			CUSH_CHECK(cudaMemcpy(Zpp, calculator.getDevZpp(), sizeof(c_double) * N, cudaMemcpyDeviceToHost));
		}
		// free device memory
		freeDeviceData(deviceData);
	}
	catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return -1;
	}

	return 0;
}


constexpr size_t RADIAL_BESSEL_MODE_COUNT = 1024;

int calculateVelocitiesRadialSymmetryModes(const double* r, const double* z, const double* phi, double* vr, double* vz, SimProperties* simProperties, size_t N, size_t nModes)
{
	if (N < 2)
	{
		std::cerr << "Error: the radial solver needs at least 2 collocation points, got " << N << std::endl;
		return -1;
	}
	if (nModes == 0)
	{
		std::cerr << "Error: the Bessel-series Green function needs at least 1 mode" << std::endl;
		return -1;
	}

	try
	{
		DeviceBuffer<double> dev_r(N);
		DeviceBuffer<double> dev_z(N);
		DeviceBuffer<double> dev_z_prime(N);
		DeviceBuffer<double> dev_z_pp(N);
		DeviceBuffer<double> dev_phi(N);
		DeviceBuffer<double> dev_phi_prime(N);
		DeviceBuffer<double> dev_vr(N);
		DeviceBuffer<double> dev_vz(N);

		CUSH_CHECK(cudaMemcpy(dev_r.get(), r, sizeof(double) * N, cudaMemcpyHostToDevice));
		if (z != nullptr) {
			CUSH_CHECK(cudaMemcpy(dev_z.get(), z, sizeof(double) * N, cudaMemcpyHostToDevice));
		}
		else {
			CUSH_CHECK(cudaMemset(dev_z.get(), 0, sizeof(double) * N));
		}
		CUSH_CHECK(cudaMemcpy(dev_phi.get(), phi, sizeof(double) * N, cudaMemcpyHostToDevice));

		FiniteDifferenceDerivativeCalculator derivativeCalculator;

		derivativeCalculator.calculateFirstDerivative(dev_z.get(), dev_r.get(), dev_z_prime.get(), N);
		derivativeCalculator.calculateFirstDerivative(dev_phi.get(), dev_r.get(), dev_phi_prime.get(), N);
		derivativeCalculator.calculateFirstDerivative(dev_z_prime.get(), dev_r.get(), dev_z_pp.get(), N);

		const RadialSurfaceState surface{
			.r = dev_r.get(),
			.eta = dev_z.get(),
			.etaPrime = dev_z_prime.get(),
			.etaPrimePrime = dev_z_pp.get(),
			.phi = dev_phi.get(),
			.phiPrime = dev_phi_prime.get(),
			.n = N
		};

		const RadialProperties radialProperties = NonDimensionalization::radialGeometry(simProperties->L, simProperties->depth);

		std::cout << "Calculating velocities for N = " << N << ", bessel_mode_count = " << nModes << std::endl;
		std::cout << "Properties: R = " << radialProperties.R << ", depth = " << radialProperties.depth << std::endl;

		RadialVelocityCalculator radialVelocityCalculator(N, nModes);
		radialVelocityCalculator.initialize(radialProperties);
		radialVelocityCalculator.calculateVelocities(dev_vr.get(), dev_vz.get(), surface, radialProperties);

		CUSH_CHECK(cudaMemcpy(vr, dev_vr.get(), sizeof(double) * N, cudaMemcpyDeviceToHost));
		CUSH_CHECK(cudaMemcpy(vz, dev_vz.get(), sizeof(double) * N, cudaMemcpyDeviceToHost));
	}
	catch (const std::exception& e)
	{
		std::cerr << "Error: " << e.what() << std::endl;
		return -1;
	}

	return 0;
}

int calculateVelocitiesRadialSymmetry(const double* r, const double* z, const double* phi, double* vr, double* vz, SimProperties* simProperties, size_t N)
{
	return calculateVelocitiesRadialSymmetryModes(r, z, phi, vr, vz, simProperties, N, RADIAL_BESSEL_MODE_COUNT);
}


int calculateDerivativeFFT(const c_double* input, c_double* output, size_t N)
{
	if (!validPeriodicSize(N)) return -1;
	try
	{
		// copy to device
		CUSH_CHECK(setDevice());
		DeviceBuffer<std_complex> devInput(N);
		DeviceBuffer<std_complex> devOutput(N);

		CUSH_CHECK(cudaMemcpy(devInput.get(), input, devInput.bytes(), cudaMemcpyHostToDevice));

		FftDerivative fftDerivative(static_cast<int>(N));
		fftDerivative.initialize();
		fftDerivative.exec(devInput.get(), devOutput.get(), false);
		CUSH_CHECK(cudaDeviceSynchronize());
		// copy back to host
		CUSH_CHECK(cudaMemcpy(output, devOutput.get(), devOutput.bytes(), cudaMemcpyDeviceToHost));
	}
	catch (const std::exception& ex)
	{
		// Handle exceptions
		std::cerr << "Error: " << ex.what() << std::endl;
		return -1;
	}
	return 0;
}
