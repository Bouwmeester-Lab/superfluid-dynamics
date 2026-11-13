#include "ExportInternal.cuh"

#include "../BaseBoundaryIntegrator.cuh"
#include "../Derivatives.cuh"
#include "../HeliumBoundaryProblem.cuh"
#include "../DeviceParticleLoading.cuh"
#include "../StateFile.h"
#include "../DerivativeCalculator.cuh"

#include <algorithm>
#include <complex>
#include <string>
#include <vector>

int calculateRHS256FromFile(const char* inputFile, const char* outputFile, double L, double rho, double kappa, double depth)
{
	try {
		ProblemProperties properties;
		properties.rho = rho;
		properties.kappa = kappa;
		properties.depth = depth;
		properties.L = L;

		// adimensionalize properties
		properties = adimensionalizeProperties(properties);

		HeliumBoundaryProblem<256, 1> heliumProblem(properties);
		BaseBoundaryIntegralCalculator<256, 1> calculator(properties, heliumProblem);

		std::vector<std::complex<double>> Z;
		std::vector<double> Phi;

		loadStateFile(std::string(inputFile), Z, Phi, 256, 2.0*PI_d / L);

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
		checkCuda(setDevice());
		// copy data to device
		loadDataToDevice(particle, deviceData, 256);
		// allocate memory for RHS
		DeviceBuffer<std_complex> devRhs(2 * 256);

		// calculate RHS
		calculator.run(deviceData.devZ, devRhs.get());

		// copy result back to host
		std::vector<std::complex<double>> rhs(2 * 256);

		std::vector<std::complex<double>> rhsPos(256);
		std::vector<double> phiRhs(256);

		CHECK_CUDA(cudaMemcpy(rhs.data(), devRhs.get(), devRhs.bytes(), cudaMemcpyDeviceToHost));

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

		saveStateFile(std::string(outputFile), rhsPos, phiRhs, 256, properties, true);
	}
	catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return -1;
	}

	return 0;
}

int calculateRHS256FromVectors(const double* x, const double* y, const double* phi, double* vx, double* vy, double* rhsPhi, double L, double rho, double kappa, double depth)
{
	try {
		ProblemProperties properties;
		properties.rho = rho;
		properties.kappa = kappa;
		properties.depth = depth;
		properties.L = L;

		// adimensionalize properties
		properties = adimensionalizeProperties(properties);

		HeliumBoundaryProblem<256, 1> heliumProblem(properties);
		BaseBoundaryIntegralCalculator<256, 1> calculator(properties, heliumProblem);

		DeviceParticleData deviceData;

		// set device
		checkCuda(setDevice());
		// copy data to device
		loadDataToDevice(x, y, phi, deviceData, 256);
		// allocate memory for RHS
		DeviceBuffer<std_complex> devRhs(2 * 256);

		// calculate RHS
		calculator.run(deviceData.devZ, devRhs.get());

		// copy result back to host
		std::vector<std::complex<double>> rhs(2 * 256);

		CHECK_CUDA(cudaMemcpy(rhs.data(), devRhs.get(), devRhs.bytes(), cudaMemcpyDeviceToHost));

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
int calculateRHSNFromVectors(const double* x, const double* y, const double* phi, double* vx, double* vy, double* rhsPhi, double L, double rho, double kappa, double depth) 
{
	try {
		ProblemProperties properties;
		properties.rho = rho;
		properties.kappa = kappa;
		properties.depth = depth;
		properties.L = L;

		// adimensionalize properties
		properties = adimensionalizeProperties(properties);
		HeliumBoundaryProblem<N, batchSize> heliumProblem(properties);
		BaseBoundaryIntegralCalculator<N, batchSize> calculator(properties, heliumProblem);

		DeviceParticleData deviceData;

		// set device
		checkCuda(setDevice());
		// copy data to device
		loadDataToDevice(x, y, phi, deviceData, N, batchSize);
		// allocate memory for RHS
		DeviceBuffer<std_complex> devRhs(2 * N * batchSize);

		// calculate RHS
		calculator.run(deviceData.devZ, devRhs.get());

		// copy result back to host
		std::vector<std::complex<double>> rhs(2 * N * batchSize);

		CHECK_CUDA(cudaMemcpy(rhs.data(), devRhs.get(), devRhs.bytes(), cudaMemcpyDeviceToHost));

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

int calculateRHS2048FromVectors(const double* x, const double* y, const double* phi, double* vx, double* vy, double* rhsPhi, double L, double rho, double kappa, double depth)
{
	try {
		return calculateRHSNFromVectors<2048, 1>(x, y, phi, vx, vy, rhsPhi, L, rho, kappa, depth);
	}
	catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return -1;
	}
	return 0;
}

int calculateRHS256FromVectorsBatched(const double* x, const double* y, const double* phi, double* vx, double* vy, double* rhsPhi, double L, double rho, double kappa, double depth, int batchSize)
{
	switch (batchSize)
	{
		case 1:
			return calculateRHSNFromVectors<256, 1>(x, y, phi, vx, vy, rhsPhi, L, rho, kappa, depth);
		case 2:
			return calculateRHSNFromVectors<256, 2>(x, y, phi, vx, vy, rhsPhi, L, rho, kappa, depth);
		case 4:
			return calculateRHSNFromVectors<256, 4>(x, y, phi, vx, vy, rhsPhi, L, rho, kappa, depth);
		case 8:
			return calculateRHSNFromVectors<256, 8>(x, y, phi, vx, vy, rhsPhi, L, rho, kappa, depth);
		case 16:
			return calculateRHSNFromVectors<256, 16>(x, y, phi, vx, vy, rhsPhi, L, rho, kappa, depth);
		case 32:
			return calculateRHSNFromVectors<256, 32>(x, y, phi, vx, vy, rhsPhi, L, rho, kappa, depth);
		case 64:
			return calculateRHSNFromVectors<256, 64>(x, y, phi, vx, vy, rhsPhi, L, rho, kappa, depth);
		case 128:
			return calculateRHSNFromVectors<256, 128>(x, y, phi, vx, vy, rhsPhi, L, rho, kappa, depth);
		case 256:
			return calculateRHSNFromVectors<256, 256>(x, y, phi, vx, vy, rhsPhi, L, rho, kappa, depth);
		case 512:
			return calculateRHSNFromVectors<256, 512>(x, y, phi, vx, vy, rhsPhi, L, rho, kappa, depth);
		case 768:
			return calculateRHSNFromVectors<256, 768>(x, y, phi, vx, vy, rhsPhi, L, rho, kappa, depth);
		case 1024:
			return calculateRHSNFromVectors<256, 1024>(x, y, phi, vx, vy, rhsPhi, L, rho, kappa, depth);
		case 1536:
			return calculateRHSNFromVectors<256, 1536>(x, y, phi, vx, vy, rhsPhi, L, rho, kappa, depth);
		case 2048:
			return calculateRHSNFromVectors<256, 2048>(x, y, phi, vx, vy, rhsPhi, L, rho, kappa, depth);
	default:
		std::cerr << "Error: Unsupported batch size " << batchSize << std::endl;
		std::cerr << "Supported batch sizes are: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768, 1024, 1536, 2048" << std::endl;
		return -1;
	}
}

int calculateRHSFromVectors(const double* x, const double* y, const double* phi, double* vx, double* vy, double* rhsPhi, double L, double rho, double kappa, double depth, size_t N)
{
	switch (N)
	{
	case 1:
		return calculateRHSNFromVectors<1, 1>(x, y, phi, vx, vy, rhsPhi, L, rho, kappa, depth);
	case 2:
		return calculateRHSNFromVectors<2, 1>(x, y, phi, vx, vy, rhsPhi, L, rho, kappa, depth);
	case 4:
		return calculateRHSNFromVectors<4, 1>(x, y, phi, vx, vy, rhsPhi, L, rho, kappa, depth);
	case 8:
		return calculateRHSNFromVectors<8, 1>(x, y, phi, vx, vy, rhsPhi, L, rho, kappa, depth);
	case 16:
		return calculateRHSNFromVectors<16, 1>(x, y, phi, vx, vy, rhsPhi, L, rho, kappa, depth);
	case 32:
		return calculateRHSNFromVectors<32, 1>(x, y, phi, vx, vy, rhsPhi, L, rho, kappa, depth);
	case 64:
		return calculateRHSNFromVectors<64, 1>(x, y, phi, vx, vy, rhsPhi, L, rho, kappa, depth);
	case 128:
		return calculateRHSNFromVectors<128, 1>(x, y, phi, vx, vy, rhsPhi, L, rho, kappa, depth);
	case 256:
		return calculateRHSNFromVectors<256, 1>(x, y, phi, vx, vy, rhsPhi, L, rho, kappa, depth);
	case 512:
		return calculateRHSNFromVectors<512, 1>(x, y, phi, vx, vy, rhsPhi, L, rho, kappa, depth);
	case 768:
		return calculateRHSNFromVectors<768, 1>(x, y, phi, vx, vy, rhsPhi, L, rho, kappa, depth);
	case 1024:
		return calculateRHSNFromVectors<1024, 1>(x, y, phi, vx, vy, rhsPhi, L, rho, kappa, depth);
	case 1536:
		return calculateRHSNFromVectors<1536, 1>(x, y, phi, vx, vy, rhsPhi, L, rho, kappa, depth);
	case 2048:
		return calculateRHSNFromVectors<2048, 1>(x, y, phi, vx, vy, rhsPhi, L, rho, kappa, depth);
	case 4096:
		return calculateRHSNFromVectors<4096, 1>(x, y, phi, vx, vy, rhsPhi, L, rho, kappa, depth);
	default:
		std::cerr << "Error: Unsupported particle number " << N << std::endl;
		std::cerr << "Supported particle numbers are: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768, 1024, 1536, 2048, 4096" << std::endl;
		return -1;
	}
}

int calculateVorticities256FromVectors(const c_double* Z, const c_double* phi, double* a, c_double* Zp, c_double* Zpp, double L, double rho, double kappa, double depth)
{
	try {
		ProblemProperties properties;
		properties.rho = rho;
		properties.kappa = kappa;
		properties.depth = depth;
		properties.L = L;

		// adimensionalize properties
		properties = adimensionalizeProperties(properties);

		HeliumBoundaryProblem<256, 1> heliumProblem(properties);
		BaseBoundaryIntegralCalculator<256, 1> calculator(properties, heliumProblem);

		DeviceParticleData deviceData;

		// set device
		checkCuda(setDevice());
		// copy data to device
		loadDataToDevice(Z, phi, deviceData, 256);

		// calculate RHS
		calculator.calculateVorticities(deviceData.devZ);

		checkCuda(cudaMemcpy(a, calculator.getDevA(), sizeof(double) * 256, cudaMemcpyDeviceToHost));
		if(Zp != nullptr) {
			checkCuda(cudaMemcpy(Zp, calculator.getDevZp(), sizeof(c_double) * 256, cudaMemcpyDeviceToHost));
		}
		if(Zpp != nullptr) {
			checkCuda(cudaMemcpy(Zpp, calculator.getDevZpp(), sizeof(c_double) * 256, cudaMemcpyDeviceToHost));
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

		CHECK_CUDA(cudaMemcpy(dev_r.get(), r, sizeof(double) * N, cudaMemcpyHostToDevice));
		if (z != nullptr) {
			CHECK_CUDA(cudaMemcpy(dev_z.get(), z, sizeof(double) * N, cudaMemcpyHostToDevice));
		}
		else {
			CHECK_CUDA(cudaMemset(dev_z.get(), 0, sizeof(double) * N));
		}
		CHECK_CUDA(cudaMemcpy(dev_phi.get(), phi, sizeof(double) * N, cudaMemcpyHostToDevice));

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

		const RadialProperties radialProperties{
			.R = 1.0,
			.depth = simProperties->depth / simProperties->L
		};

		std::cout << "Calculating velocities for N = " << N << ", bessel_mode_count = " << nModes << std::endl;
		std::cout << "Properties: R = " << radialProperties.R << ", depth = " << radialProperties.depth << std::endl;

		RadialVelocityCalculator radialVelocityCalculator(N, nModes);
		radialVelocityCalculator.initialize(radialProperties);
		radialVelocityCalculator.calculateVelocities(dev_vr.get(), dev_vz.get(), surface, radialProperties);

		CHECK_CUDA(cudaMemcpy(vr, dev_vr.get(), sizeof(double) * N, cudaMemcpyDeviceToHost));
		CHECK_CUDA(cudaMemcpy(vz, dev_vz.get(), sizeof(double) * N, cudaMemcpyDeviceToHost));
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
		CHECK_CUDA(setDevice());
		DeviceBuffer<std_complex> devInput(256);
		DeviceBuffer<std_complex> devOutput(256);

		CHECK_CUDA(cudaMemcpy(devInput.get(), input, devInput.bytes(), cudaMemcpyHostToDevice));

		FftDerivative<256, 1> fftDerivative;
		fftDerivative.initialize();
		fftDerivative.exec(devInput.get(), devOutput.get(), false);
		checkCuda(cudaDeviceSynchronize());
		// copy back to host
		CHECK_CUDA(cudaMemcpy(output, devOutput.get(), devOutput.bytes(), cudaMemcpyDeviceToHost));
	}
	catch (const std::exception& ex)
	{
		// Handle exceptions
		std::cerr << "Error: " << ex.what() << std::endl;
		return -1;
	}
	return 0;
}
