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

		HeliumBoundaryProblem<256, 1> heliumProblem(properties);
		BoundaryIntegralCalculator<256, 1> calculator(properties, heliumProblem);

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

		HeliumBoundaryProblem<256, 1> heliumProblem(properties);
		BoundaryIntegralCalculator<256, 1> calculator(properties, heliumProblem);

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




template<size_t N, size_t batchSize>
int calculateRHSNFromVectors(const double* x, const double* y, const double* phi, double* vx, double* vy, double* rhsPhi, double L, double rho, double kappa, double depth) 
{
	try {
		ProblemProperties properties;
		properties.rho = rho;
		properties.kappa = kappa;
		properties.depth = depth;

		// adimensionalize properties
		properties = adimensionalizeProperties(properties, L);
		//std::cout << "Adimensionalized properties. " << std::endl;
		HeliumBoundaryProblem<N, batchSize> heliumProblem(properties);
		//std::cout << "Created HeliumBoundaryProblem. " << std::endl;
		BoundaryIntegralCalculator<N, batchSize> calculator(properties, heliumProblem);
		//std::cout << "Created BoundaryIntegralCalculator. " << std::endl;
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
		loadDataToDevice(x, y, phi, deviceData, N, batchSize);
		// allocate memory for RHS
		checkCuda(cudaMalloc(&devRhs, 2 * sizeof(std_complex) * N * batchSize));

		// calculate RHS
		calculator.run(deviceData.devZ, devRhs);

		// copy result back to host
		std::vector<std::complex<double>> rhs(2 * N * batchSize);

		//std::vector<std::complex<double>> rhsPos(256);
		//std::vector<double> phiRhs(256);

		checkCuda(cudaMemcpy(rhs.data(), devRhs, 2 * sizeof(std_complex) * N * batchSize, cudaMemcpyDeviceToHost));

		// free device memory
		checkCuda(cudaFree(devRhs));
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

int calculateVorticities256FromVectors(const c_double* Z, const c_double* phi, double* a, c_double* Zp, c_double* Zpp, double L, double rho, double kappa, double depth)
{
	try {
		ProblemProperties properties;
		properties.rho = rho;
		properties.kappa = kappa;
		properties.depth = depth;

		// adimensionalize properties
		properties = adimensionalizeProperties(properties, L);

		HeliumBoundaryProblem<256, 1> heliumProblem(properties);
		BoundaryIntegralCalculator<256, 1> calculator(properties, heliumProblem);

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

		
		DeviceParticleData deviceData;

		// set device
		checkCuda(setDevice());
		// copy data to device
		loadDataToDevice(Z, phi, deviceData, 256);

		//for (size_t i = 0; i < 256; i++) {
		//	std::cout << phi[i].re << " + " << phi[i].im << "i, " << std::endl;
		//}

		//std::cout << "\n\n\n\n";
		//for (size_t i = 0; i < 256; i++) {
		//	std::cout << Z[i].re << " + " << Z[i].im << "i, " << std::endl;
		//}

		//	

		// calculate RHS
		calculator.calculateVorticities(deviceData.devZ);
		//cudaDeviceSynchronize();
		//std::vector<std::complex<double>> rhsPos(256);
		//std::vector<double> phiRhs(256);

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

int calculateDerivativeFFT256(const c_double* input, c_double* output)
{
	try
	{
		// copy to device
		std_complex* devInput;
		std_complex* devOutput;

		checkCuda(setDevice());
		checkCuda(cudaMalloc(&devInput, sizeof(std_complex) * 256));
		checkCuda(cudaMalloc(&devOutput, sizeof(std_complex) * 256));

		checkCuda(cudaMemcpy(devInput, input, sizeof(std_complex) * 256, cudaMemcpyHostToDevice));



		FftDerivative<256, 1> fftDerivative;
		fftDerivative.initialize();
		fftDerivative.exec(devInput, devOutput, false);
		checkCuda(cudaDeviceSynchronize());
		// copy back to host
		checkCuda(cudaMemcpy(output, devOutput, sizeof(std_complex) * 256, cudaMemcpyDeviceToHost));

		// free device memory
		checkCuda(cudaFree(devInput));
		checkCuda(cudaFree(devOutput));
	}
	catch (const std::exception& ex)
	{
		// Handle exceptions
		std::cerr << "Error: " << ex.what() << std::endl;
	}
	return 0;
}

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

		// adimensionalize properties
		properties = adimensionalizeProperties(properties, L);

		

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
			std::unique_ptr<AutonomousProblem<std_complex, 6 * N * N>> boundaryIntegralCalculatorPtr = std::make_unique<BoundaryIntegralCalculator<N, 3 * N>>(properties, heliumProblem);


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
		std::cerr << "Supported batch sizes are: 1, 2, 4, 8, 16, 32, 64, 128, 256" << std::endl;
		break;
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
		std::cerr << "Error: Unsupported particle number" << N << std::endl;
		std::cerr << "Supported batch sizes are: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768, 1024, 1536, 2048, 4096" << std::endl;
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

		//std::vector<std::complex<double>> hostZPhiBatched(6 * N * N);
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

template <size_t N>
int integrateSimulationGL2_N(const double* initialState, double** statesOut, size_t* statesCount,
	double** timesOut, size_t* timesCount,
	SimProperties* simProperties, GaussLegendreOptions* glCOptions)
{
	try {
		ProblemProperties properties;
		properties.rho = simProperties->rho;
		properties.kappa = simProperties->kappa;
		properties.depth = simProperties->depth;
		// adimensionalize properties
		properties = adimensionalizeProperties(properties, simProperties->L);
		// create the options for GL
		GaussLegendre2Options glOptions = createOptionsFromCOptions(*glCOptions);


		// calculators for the f(y)
		HeliumBoundaryProblem<N, 1> heliumProblem(properties);
		BoundaryIntegralCalculator<N, 1> calculator(properties, heliumProblem);
		RealBoundaryItegralCalculator<N> realCalculator(calculator);
		// calculators for the jacobian
		HeliumBoundaryProblem<N, 3 * N> heliumJacProblem(properties);
		std::unique_ptr<AutonomousProblem<std_complex, 6 * N * N>> jacBoundaryIntegralCalculatorPtr = std::make_unique<BoundaryIntegralCalculator<N, 3 * N>>(properties, heliumJacProblem);
		JacobianCalculator<N> jacobianCalculator(std::move(jacBoundaryIntegralCalculatorPtr));


		GaussLegendre2<N> integrator(realCalculator, jacobianCalculator, glOptions);
		integrator.setStream(cudaStreamPerThread);

		integrator.initialize(initialState, false);
		integrator.runEvolution(glCOptions->t0, glCOptions->t1);

		// copy results to output
		cudaStreamSynchronize(cudaStreamPerThread);
		integrator.copyTimesToHost(timesOut, timesCount);
		integrator.copyStatesToHost(statesOut, statesCount);

		//integrator.integrate(deviceData.devZ, simProperties->t0, simProperties->tFinal, statesOut, timesOut, N);
		//freeDeviceData(deviceData);
	}
	catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return -1;
	}
	return 0;
}


int integrateSimulationGL2(const double* initialState, double** statesOut, size_t* statesCount,
	double** timesOut, size_t* timesCount,
	SimProperties* simProperties, GaussLegendreOptions* glCOptions, size_t N) 
{
	switch (N)
	{
		case 32:
			return integrateSimulationGL2_N<32>(initialState, statesOut, statesCount, timesOut, timesCount, simProperties, glCOptions);
		case 64:
			return integrateSimulationGL2_N<64>(initialState, statesOut, statesCount, timesOut, timesCount, simProperties, glCOptions);
		case 128:
			return integrateSimulationGL2_N<128>(initialState, statesOut, statesCount, timesOut, timesCount, simProperties, glCOptions);
		case 256:
			return integrateSimulationGL2_N<256>(initialState, statesOut, statesCount, timesOut, timesCount, simProperties, glCOptions);
		case 512:
			return integrateSimulationGL2_N<512>(initialState, statesOut, statesCount, timesOut, timesCount, simProperties, glCOptions);
		case 1024:
			return integrateSimulationGL2_N<1024>(initialState, statesOut, statesCount, timesOut, timesCount, simProperties, glCOptions);
	default:
		std::cerr << "Error: Unsupported N size " << N << std::endl;
		std::cerr << "Supported N sizes are: 32, 64, 128 , 256, 512, 1024" << std::endl;
		break;
	}
}


int integrateSimulationGL2_freeMemory(double* statesOut, double* timesOut)
{
	std::free(statesOut);
	std::free(timesOut);
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
