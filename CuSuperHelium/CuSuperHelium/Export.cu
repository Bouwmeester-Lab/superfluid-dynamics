#include "Export.cuh"
//#include "SimulationFunctions.cuh"

int dispertionTest256(double wavelength, double simulationTime, double rho, double kappa, double depth, int steps)
{
	ProblemProperties properties;
	properties.rho = rho;
	properties.kappa = kappa;
	properties.depth = depth;

	return dispersionTest<256>(wavelength, simulationTime, properties, steps);
}

int calculateRHS256FromFile(const char* inputFile, const char* outputFile, double L, double rho, double kappa, double depth)
{
	try {
		ProblemProperties properties;
		properties.rho = rho;
		properties.kappa = kappa;
		properties.depth = depth;

		// adimensionalize properties
		properties = adimensionalizeProperties(properties, L);

		HeliumBoundaryProblem<256> heliumProblem(properties);
		BoundaryIntegralCalculator<256> calculator(properties, heliumProblem);

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

		std_complex* devRhs;

		// set device
		checkCuda(setDevice());
		// copy data to device
		loadDataToDevice(particle, deviceData, 256);
		// allocate memory for RHS
		checkCuda(cudaMalloc(&devRhs, 2 * sizeof(std_complex) * 256));

		// calculate RHS
		calculator.run(deviceData.devZ, devRhs);

		// copy result back to host
		std::vector<std::complex<double>> rhs(2 * 256);

		std::vector<std::complex<double>> rhsPos(256);
		std::vector<double> phiRhs(256);

		checkCuda(cudaMemcpy(rhs.data(), devRhs, 2 * sizeof(std_complex) * 256, cudaMemcpyDeviceToHost));

		// free device memory
		checkCuda(cudaFree(devRhs));
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

		// adimensionalize properties
		properties = adimensionalizeProperties(properties, L);

		HeliumBoundaryProblem<256> heliumProblem(properties);
		BoundaryIntegralCalculator<256> calculator(properties, heliumProblem);

		//std::vector<std::complex<double>> Z;
		//std::vector<double> Phi;

		///*loadStateFile(std::string(inputFile), Z, Phi, 256, 2.0 * PI_d / L);*/

		//auto [min_it, max_it] = std::minmax_element(
		//	Z.begin(), Z.end(),
		//	[](const auto& a, const auto& b) {
		//		return a.imag() < b.imag();
		//	}
		//);

		//properties.initial_amplitude = (max_it->imag() - min_it->imag()) / 2.0;

		//ParticleData particle(Z, Phi);
		DeviceParticleData deviceData;

		std_complex* devRhs;

		// set device
		checkCuda(setDevice());
		// copy data to device
		loadDataToDevice(x, y, phi, deviceData, 256);
		// allocate memory for RHS
		checkCuda(cudaMalloc(&devRhs, 2 * sizeof(std_complex) * 256));

		// calculate RHS
		calculator.run(deviceData.devZ, devRhs);

		// copy result back to host
		std::vector<std::complex<double>> rhs(2 * 256);

		//std::vector<std::complex<double>> rhsPos(256);
		//std::vector<double> phiRhs(256);

		checkCuda(cudaMemcpy(rhs.data(), devRhs, 2 * sizeof(std_complex) * 256, cudaMemcpyDeviceToHost));

		// free device memory
		checkCuda(cudaFree(devRhs));
		freeDeviceData(deviceData);

		// prepare data into two vectors, one for position RHS and one for potential RHS
		for (int i = 0; i < 256; i++)
		{
			vx[i] = rhs[i].real();
			vy[i] = rhs[i].imag();
			rhsPhi[i] = rhs[i + 256].real(); // imaginary part should be zero
		}
		// save to file

		//saveStateFile(std::string(outputFile), rhsPos, phiRhs, 256, properties, true);
	}
	catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return -1;
	}

	return 0;
}

ProblemProperties adimensionalizeProperties(ProblemProperties props, double L, double rhoHelium)
{
	double L0 = L / (2.0 * PI_d); // characteristic length
	double g = 3 * 2.6e-24 / std::pow(props.depth, 4);
	double _t0 = std::sqrt(L0 / g);
	double surfaceTensionFactor = rhoHelium * L0 * L0 * L0 / (_t0 * _t0);

	props.kappa = props.kappa / surfaceTensionFactor;
	props.depth = 2.0 * CUDART_PI * props.depth / L;
	props.rho /= rhoHelium; 

	return props;
}
