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

int calculateRHS256FromFile(const char* inputFile, const char* outputFile, SimProperties* simProperties)
{
	try {
		DimensionlessParameters properties = toDimensionlessParameters(*simProperties);

		HeliumBoundaryProblem<256, 1> heliumProblem(properties);
		BaseBoundaryIntegralCalculator<256, 1> calculator(properties, heliumProblem);

		std::vector<std::complex<double>> Z;
		std::vector<double> Phi;

		loadStateFile(std::string(inputFile), Z, Phi, 256, 2.0*PI_d / simProperties->L);

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
		loadDataToDevice(particle, deviceData, 256);
		// allocate memory for RHS
		DeviceBuffer<std_complex> devRhs(2 * 256);

		// calculate RHS
		calculator.run(deviceData.devZ(), devRhs.get());

		// copy result back to host
		std::vector<std::complex<double>> rhs(2 * 256);

		std::vector<std::complex<double>> rhsPos(256);
		std::vector<double> phiRhs(256);

		CUSH_CHECK(cudaMemcpy(rhs.data(), devRhs.get(), devRhs.bytes(), cudaMemcpyDeviceToHost));

		// free device memory
		devRhs.reset();
		freeDeviceData(deviceData);

		// prepare data into two vectors, one for position RHS and one for potential RHS
		for (int i = 0; i < 256; i++)
		{
			rhsPos[i] = rhs[i];
			phiRhs[i] = rhs[i + 256].real(); // imaginary part should be zero
		}
		// save to file

		const std::map<std::string, double> attributes{ { "rho", properties.rho }, { "kappa", properties.kappa }, { "U", properties.U },
			{ "depth", properties.depth }, { "initial_amplitude", properties.initial_amplitude } };
		saveStateFile(std::string(outputFile), rhsPos, phiRhs, 256, attributes, true);
	}
	catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return -1;
	}

	return 0;
}

int calculateRHS256FromVectors(const double* x, const double* y, const double* phi, double* vx, double* vy, double* rhsPhi, SimProperties* simProperties)
{
	try {
		DimensionlessParameters properties = toDimensionlessParameters(*simProperties);

		HeliumBoundaryProblem<256, 1> heliumProblem(properties);
		BaseBoundaryIntegralCalculator<256, 1> calculator(properties, heliumProblem);

		DeviceParticleData deviceData;

		// set device
		CUSH_CHECK(setDevice());
		// copy data to device
		loadDataToDevice(x, y, phi, deviceData, 256);
		// allocate memory for RHS
		DeviceBuffer<std_complex> devRhs(2 * 256);

		// calculate RHS
		calculator.run(deviceData.devZ(), devRhs.get());

		// copy result back to host
		std::vector<std::complex<double>> rhs(2 * 256);

		CUSH_CHECK(cudaMemcpy(rhs.data(), devRhs.get(), devRhs.bytes(), cudaMemcpyDeviceToHost));

		// free device memory
		devRhs.reset();
		freeDeviceData(deviceData);

		// prepare data into two vectors, one for position RHS and one for potential RHS
		for (int i = 0; i < 256; i++)
		{
			vx[i] = rhs[i].real();
			vy[i] = rhs[i].imag();
			rhsPhi[i] = rhs[i + 256].real(); // imaginary part should be zero
		}
	}
	catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return -1;
	}

	return 0;
}

template<size_t N, size_t batchSize>
int calculateRHSNFromVectors(const double* x, const double* y, const double* phi, double* vx, double* vy, double* rhsPhi, SimProperties* simProperties) 
{
	try {
		DimensionlessParameters properties = toDimensionlessParameters(*simProperties);
		HeliumBoundaryProblem<N, batchSize> heliumProblem(properties);
		BaseBoundaryIntegralCalculator<N, batchSize> calculator(properties, heliumProblem);

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
		for (int i = 0; i < batchSize * N; i++)
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

int calculateRHS2048FromVectors(const double* x, const double* y, const double* phi, double* vx, double* vy, double* rhsPhi, SimProperties* simProperties)
{
	try {
		return calculateRHSNFromVectors<2048, 1>(x, y, phi, vx, vy, rhsPhi, simProperties);
	}
	catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return -1;
	}
	return 0;
}

int calculateRHS256FromVectorsBatched(const double* x, const double* y, const double* phi, double* vx, double* vy, double* rhsPhi, SimProperties* simProperties, int batchSize)
{
	switch (batchSize)
	{
		case 1:
			return calculateRHSNFromVectors<256, 1>(x, y, phi, vx, vy, rhsPhi, simProperties);
		case 2:
			return calculateRHSNFromVectors<256, 2>(x, y, phi, vx, vy, rhsPhi, simProperties);
		case 4:
			return calculateRHSNFromVectors<256, 4>(x, y, phi, vx, vy, rhsPhi, simProperties);
		case 8:
			return calculateRHSNFromVectors<256, 8>(x, y, phi, vx, vy, rhsPhi, simProperties);
		case 16:
			return calculateRHSNFromVectors<256, 16>(x, y, phi, vx, vy, rhsPhi, simProperties);
		case 32:
			return calculateRHSNFromVectors<256, 32>(x, y, phi, vx, vy, rhsPhi, simProperties);
		case 64:
			return calculateRHSNFromVectors<256, 64>(x, y, phi, vx, vy, rhsPhi, simProperties);
		case 128:
			return calculateRHSNFromVectors<256, 128>(x, y, phi, vx, vy, rhsPhi, simProperties);
		case 256:
			return calculateRHSNFromVectors<256, 256>(x, y, phi, vx, vy, rhsPhi, simProperties);
		case 512:
			return calculateRHSNFromVectors<256, 512>(x, y, phi, vx, vy, rhsPhi, simProperties);
		case 768:
			return calculateRHSNFromVectors<256, 768>(x, y, phi, vx, vy, rhsPhi, simProperties);
		case 1024:
			return calculateRHSNFromVectors<256, 1024>(x, y, phi, vx, vy, rhsPhi, simProperties);
		case 1536:
			return calculateRHSNFromVectors<256, 1536>(x, y, phi, vx, vy, rhsPhi, simProperties);
		case 2048:
			return calculateRHSNFromVectors<256, 2048>(x, y, phi, vx, vy, rhsPhi, simProperties);
	default:
		std::cerr << "Error: Unsupported batch size " << batchSize << std::endl;
		std::cerr << "Supported batch sizes are: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768, 1024, 1536, 2048" << std::endl;
		return -1;
	}
}

int calculateRHSFromVectors(const double* x, const double* y, const double* phi, double* vx, double* vy, double* rhsPhi, SimProperties* simProperties, size_t N)
{
	switch (N)
	{
	case 1:
		return calculateRHSNFromVectors<1, 1>(x, y, phi, vx, vy, rhsPhi, simProperties);
	case 2:
		return calculateRHSNFromVectors<2, 1>(x, y, phi, vx, vy, rhsPhi, simProperties);
	case 4:
		return calculateRHSNFromVectors<4, 1>(x, y, phi, vx, vy, rhsPhi, simProperties);
	case 8:
		return calculateRHSNFromVectors<8, 1>(x, y, phi, vx, vy, rhsPhi, simProperties);
	case 16:
		return calculateRHSNFromVectors<16, 1>(x, y, phi, vx, vy, rhsPhi, simProperties);
	case 32:
		return calculateRHSNFromVectors<32, 1>(x, y, phi, vx, vy, rhsPhi, simProperties);
	case 64:
		return calculateRHSNFromVectors<64, 1>(x, y, phi, vx, vy, rhsPhi, simProperties);
	case 128:
		return calculateRHSNFromVectors<128, 1>(x, y, phi, vx, vy, rhsPhi, simProperties);
	case 256:
		return calculateRHSNFromVectors<256, 1>(x, y, phi, vx, vy, rhsPhi, simProperties);
	case 512:
		return calculateRHSNFromVectors<512, 1>(x, y, phi, vx, vy, rhsPhi, simProperties);
	case 768:
		return calculateRHSNFromVectors<768, 1>(x, y, phi, vx, vy, rhsPhi, simProperties);
	case 1024:
		return calculateRHSNFromVectors<1024, 1>(x, y, phi, vx, vy, rhsPhi, simProperties);
	case 1536:
		return calculateRHSNFromVectors<1536, 1>(x, y, phi, vx, vy, rhsPhi, simProperties);
	case 2048:
		return calculateRHSNFromVectors<2048, 1>(x, y, phi, vx, vy, rhsPhi, simProperties);
	case 4096:
		return calculateRHSNFromVectors<4096, 1>(x, y, phi, vx, vy, rhsPhi, simProperties);
	default:
		std::cerr << "Error: Unsupported particle number " << N << std::endl;
		std::cerr << "Supported particle numbers are: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768, 1024, 1536, 2048, 4096" << std::endl;
		return -1;
	}
}

int calculateVorticities256FromVectors(const c_double* Z, const c_double* phi, double* a, c_double* Zp, c_double* Zpp, SimProperties* simProperties)
{
	try {
		DimensionlessParameters properties = toDimensionlessParameters(*simProperties);

		HeliumBoundaryProblem<256, 1> heliumProblem(properties);
		BaseBoundaryIntegralCalculator<256, 1> calculator(properties, heliumProblem);

		DeviceParticleData deviceData;

		// set device
		CUSH_CHECK(setDevice());
		// copy data to device
		loadDataToDevice(Z, phi, deviceData, 256);

		// calculate RHS
		calculator.calculateVorticities(deviceData.devZ());

		CUSH_CHECK(cudaMemcpy(a, calculator.getDevA(), sizeof(double) * 256, cudaMemcpyDeviceToHost));
		if(Zp != nullptr) {
			CUSH_CHECK(cudaMemcpy(Zp, calculator.getDevZp(), sizeof(c_double) * 256, cudaMemcpyDeviceToHost));
		}
		if(Zpp != nullptr) {
			CUSH_CHECK(cudaMemcpy(Zpp, calculator.getDevZpp(), sizeof(c_double) * 256, cudaMemcpyDeviceToHost));
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


int calculateDerivativeFFT256(const c_double* input, c_double* output)
{
	try
	{
		// copy to device
		CUSH_CHECK(setDevice());
		DeviceBuffer<std_complex> devInput(256);
		DeviceBuffer<std_complex> devOutput(256);

		CUSH_CHECK(cudaMemcpy(devInput.get(), input, devInput.bytes(), cudaMemcpyHostToDevice));

		FftDerivative<256, 1> fftDerivative;
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
