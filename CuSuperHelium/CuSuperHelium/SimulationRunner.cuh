#pragma once


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "ProblemProperties.hpp"
#include "WaterBoundaryIntegralCalculator.cuh"
#include "SimpleEuler.cuh"
#include "AutonomousRungeKuttaStepper.cuh"
#include "ValueLogger.h"
#include "matplotlibcpp.h"
#include "VideoMaking.h"
#include <highfive/H5File.hpp>
#include "RK45.cuh"
#include "SimulationOptions.h"
#include "ExportTypes.cuh"

namespace plt = matplotlibcpp;


template <int N, size_t batchSize>
struct KineticEnergyFunctor {
private:
    BoundaryProblem<N, batchSize>& boundaryProblem;
public:
    KineticEnergyFunctor(BoundaryProblem<N, batchSize>& boundaryProblem) : boundaryProblem(boundaryProblem)
    {

    }

    double operator()() 
    {
        return boundaryProblem.energyContainer.kineticEnergy->getEnergy();
    }
};

template <int N, size_t batchSize>
struct PotentialEnergyFunctor {
private:
    BoundaryProblem<N, batchSize>& boundaryProblem;
public:
    PotentialEnergyFunctor(BoundaryProblem<N, batchSize>& boundaryProblem) : boundaryProblem(boundaryProblem)
    {

    }

    double operator()()
    {
        return boundaryProblem.energyContainer.potentialEnergy->getEnergy();
    }
};

template <int N, size_t batchSize>
struct VolumeFluxFunctor {
private:
    BoundaryIntegralCalculator<N, batchSize>& integralCalculator;
public:
    VolumeFluxFunctor(BoundaryIntegralCalculator<N, batchSize>& integralCalculator) : integralCalculator(integralCalculator)
    {

    }

    double operator()()
    {
        return integralCalculator.volumeFlux.getEnergy();
    }
};

struct TotalEnergyFunctor 
{
private:
    std::shared_ptr<ValueLogger> kineticLogger;
    std::shared_ptr<ValueLogger> potentialLogger;
public:
    TotalEnergyFunctor(std::shared_ptr<ValueLogger> kineticLogger, std::shared_ptr<ValueLogger> potentialLogger) : kineticLogger(kineticLogger), potentialLogger(potentialLogger)
    {

    }

    double operator()() 
    {
        return kineticLogger->getLastLoggedValue() + potentialLogger->getLastLoggedValue();
    }
};

/// <summary>
/// Loads the data in an H5 file containing two datasets with 2 columns each (x, y) pairs representing complex numbers.
/// The first dataset must be "interface" representing the interface shape.
/// The second dataset must be "potential" representing the potential (only the first column is used).
/// </summary>
/// <param name="path"></param>
/// <param name="Z"></param>
/// <param name="phi"></param>
/// <returns></returns>
int loadStateFile(const std::string path, std::vector<std::complex<double>>& Z, std::vector<double>& phi, const size_t N, const double scaleY = 1.0) 
{
    
    HighFive::File file(path, HighFive::File::ReadOnly);

    auto interfaceDataSet = file.getDataSet("interface");
    auto potential = file.getDataSet("potential");

    // check dimentsions and make sure its N x 2
    auto dims = interfaceDataSet.getSpace().getDimensions();
    if (dims.size() != 2 || dims[1] != 2 || dims[0] != N) {
        if (dims.size() == 1) {
            throw std::runtime_error(std::format("Interface must have shape ({}, 2), instead it has ({},)", N, dims[0]));
        }
        else if (dims.size() == 2){
            throw std::runtime_error(std::format("Interface must have shape ({}, 2), instead it has ({}, {})", N, dims[0], dims[1]));
        }
        else {
            throw std::runtime_error(std::format("Interface must have shape ({}, 2), instead it has a higher dimensionality than 2 ({})!", N, dims.size()));
        }
        
    }

    // check dimensions for potential
    auto potDims = potential.getSpace().getDimensions();
    if (potDims.size() == 0) {
        throw std::runtime_error("potential dataset is empty.");
    }
    std::vector<std::array<double, 2>> dataInterface;
	//std::vector<double> dataPotential;

    phi.clear();
	potential.read(phi);

	// read the interface data
    interfaceDataSet.read(dataInterface);

	// convert to complex numbers
    Z.clear();
    Z.resize(dataInterface.size());
    for (size_t i = 0; i < dataInterface.size(); ++i) {
        Z[i] = std::complex<double>(dataInterface[i][0], scaleY * dataInterface[i][1]);
    }
	return 0;
}
void createPropertiesAttributes(HighFive::File& file, const ProblemProperties& properties) 
{
    // add properties to the file
    file.createAttribute<double>("rho", properties.rho);
    file.createAttribute<double>("kappa", properties.kappa);
    file.createAttribute<double>("U", properties.U);
    file.createAttribute<double>("depth", properties.depth);
	file.createAttribute<double>("initial_amplitude", properties.initial_amplitude);
}

int saveStateFile(const std::string path, std::vector<std::complex<double>>& Z, std::vector<double>& phi, const size_t N, ProblemProperties properties, bool rhs = false)
{
	HighFive::File file(path, HighFive::File::Overwrite);

	// add properties to the file
	createPropertiesAttributes(file, properties);
	file.createAttribute<bool>("rhs", rhs);

	auto dataset = file.createDataSet<double>("interface", HighFive::DataSpace({ N, 2 }));
	auto potential = file.createDataSet<double>("potential", HighFive::DataSpace({ N, }));

    std::vector<std::array<double, 2>> dataInterface(N);
    for (size_t i = 0; i < Z.size(); ++i) {
        dataInterface[i] = { Z[i].real(), Z[i].imag() };
    }

	// write the interface data into the dataset
    dataset.write(dataInterface);
	// write the potential data into the dataset
	potential.write(phi);

	file.flush();

	return 0;
}

int loadDataToDevice(const double* x, const double* y, const double* phi, DeviceParticleData& deviceData, const size_t N, const size_t batchSize = 1, cudaStream_t stream = cudaStreamPerThread)
{
    cudaError_t cudaStatus;
    // Allocate GPU buffers for three vectors (two input, one output)    .
	//std::cout << "Allocating device memory..." << std::endl;
    cudaStatus = cudaMallocAsync(&deviceData.devZ, 2 * N * batchSize * sizeof(std_complex), stream); // we need space for both positions and potentials
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return cudaStatus;
    }
	//std::cout << "Device memory allocated." << std::endl;
    // Copy input vectors from host memory to GPU buffers.
    std::vector<std_complex> ZHost(batchSize * N);
	std::vector<std_complex> PhiHost(batchSize * N);

	//std::cout << "Loading data to host complex vectors..." << std::endl;
    for (size_t i = 0; i < batchSize * N; ++i)
    {
        ZHost[i] = std_complex(x[i], y[i]);
		PhiHost[i] = std_complex(phi[i], 0.0);
    }

    cudaStatus = cudaMemcpyAsync(deviceData.devZ, ZHost.data(), batchSize * N * sizeof(std_complex), cudaMemcpyHostToDevice, stream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return cudaStatus;
    }

    // copy to device
    cudaStatus = cudaMemcpyAsync(deviceData.devZ + batchSize * N, PhiHost.data(), batchSize * N * sizeof(std_complex), cudaMemcpyHostToDevice, stream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return cudaStatus;
    }
	//std::cout << "Data loaded to device." << std::endl;
    //cudaDeviceSynchronize();
    //std::vector<std_complex> checkZ(2 * N);
    //cudaStatus = cudaMemcpy(checkZ.data(), deviceData.devZ, 2 * N * sizeof(std_complex), cudaMemcpyDeviceToHost);
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaMemcpy failed!");
    //    return cudaStatus;
    //}

    //std::vector<double> X(N);
    //std::vector<double> Y(N);
    //std::vector<double> Phi(N);
    //std::vector<double> PhiImag(N);

    //for (size_t i = 0; i < N; ++i)
    //{
    //    X[i] = checkZ[i].real();
    //    Y[i] = checkZ[i].imag();
    //    Phi[i] = checkZ[i + N].real();
    //    PhiImag[i] = checkZ[i + N].imag(); // should be 0
    //}
    /*plt::figure();
    plt::plot(X, Y);
    plt::plot(X, Phi);
    plt::plot(X, PhiImag);
    plt::title("Initial Condition");
    plt::show();*/
    return 0;
}

int loadDataToDevice(const c_double* Z, const c_double* phi, DeviceParticleData& deviceData, size_t N, cudaStream_t stream = cudaStreamPerThread)
{
    cudaError_t cudaStatus;
    // Allocate GPU buffers for three vectors (two input, one output)    .
    checkCuda(cudaMallocAsync(&deviceData.devZ, 2 * N * sizeof(std_complex), stream)); // we need space for both positions and potentials
    checkCuda(cudaMemcpyAsync(deviceData.devZ, Z, N * sizeof(std_complex), cudaMemcpyHostToDevice, stream));

    checkCuda(cudaMemcpyAsync(deviceData.devZ + N, phi, N * sizeof(std_complex), cudaMemcpyHostToDevice, stream));
	
    /*cudaDeviceSynchronize();
	std::vector<c_double> ZHost(2 * N);
	checkCuda(cudaMemcpy(ZHost.data(), deviceData.devZ, 2 * N * sizeof(c_double), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        if (Z[i].re != ZHost[i].re || Z[i].im != ZHost[i].im)
        {
			std::cout << "Mismatch at " << i << ": " << Z[i].re << " != " << ZHost[i].re << std::endl;
			std::cout << "Mismatch at " << i << ": " << Z[i].im << " != " << ZHost[i].im << std::endl;
        }
        if (phi[i].re != ZHost[i + N].re || phi[i].im != ZHost[i + N].im)
        {
			std::cout << "Mismatch at " << i << ": " << phi[i].re << " != " << ZHost[i + N].re << std::endl;
			std::cout << "Mismatch at " << i << ": " << phi[i].im << " != " << ZHost[i + N].im << std::endl;
        }
    }*/

	return 0;
}

int loadDataToDevice(ParticleData& data, DeviceParticleData& deviceData, const size_t N) 
{
    cudaError_t cudaStatus;
    // Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc(&deviceData.devZ, 2 * N * sizeof(std_complex)); // we need space for both positions and potentials
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return cudaStatus;
    }
    // Copy input vectors from host memory to GPU buffers.
	std::vector<std_complex> ZHost(N);
    for (size_t i = 0; i < N; ++i)
    {
		ZHost[i] = std_complex(data.Z[i].real(), data.Z[i].imag());
    }

    cudaStatus = cudaMemcpy(deviceData.devZ,  ZHost.data(), N * sizeof(std_complex), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return cudaStatus;
    }

	// copy the potentials
	std::vector<std_complex> potentialComplex(N);
    for (size_t i = 0; i < N; ++i) 
    {
		potentialComplex[i] = std_complex(data.Potential[i], 0.0);
    }
	// copy to device
	cudaStatus = cudaMemcpy(deviceData.devZ + N, potentialComplex.data(), N * sizeof(std_complex), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return cudaStatus;
    }
	cudaDeviceSynchronize();
	std::vector<std_complex> checkZ(2*N);
	cudaStatus = cudaMemcpy(checkZ.data(), deviceData.devZ, 2 * N * sizeof(std_complex), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return cudaStatus;
	}

	//std::vector<double> X(N);
	//std::vector<double> Y(N);
	//std::vector<double> Phi(N);
	//std::vector<double> PhiImag(N);

 //   for (size_t i = 0; i < N; ++i)
 //   {
	//	X[i] = checkZ[i].real();
	//	Y[i] = checkZ[i].imag();
	//	Phi[i] = checkZ[i + N].real();
	//	PhiImag[i] = checkZ[i + N].imag(); // should be 0
 //   }
	/*plt::figure();
	plt::plot(X, Y);
	plt::plot(X, Phi);
	plt::plot(X, PhiImag);
	plt::title("Initial Condition");
	plt::show();*/
    return 0;
}


int freeDeviceData(DeviceParticleData& deviceData) 
{
    cudaError_t cudaStatus;
    cudaStatus = cudaFree(deviceData.devZ);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaFree failed!");
        return cudaStatus;
    }
    return 0;
}

/// <summary>
/// Sets the device data for the simulation. This function assumes that the device hasn't already been set using setDevice().
/// </summary>
template<int numParticles>
int setDeviceData(ParticleData data, DeviceParticleData& deviceData)
{
    cudaError_t cudaStatus;
    cudaStatus = setDevice();
    if (cudaStatus != cudaSuccess) {
        return cudaStatus;
    }
    loadDataToDevice(data, deviceData, numParticles);
}

template <int numParticles>
RK45_std_complex<2 * numParticles> createRK45Solver() 
{

}

template <int numParticles, size_t batchSize, typename TSolver, typename TOptions>
requires std::derived_from<std::remove_cvref_t<TSolver>, OdeSolver>
int runSimulation(BoundaryProblem<numParticles, batchSize>& boundaryProblem, const int numSteps, ProblemProperties& properties, ParticleData data, TOptions solverOptions, SimulationOptions simOptions, const int loggingPeriod = -1, const bool plot = true, const bool show = true, double t0 = 1.0, int fps = 10, const bool saveH5 = true)
{ 
    cudaError_t cudaStatus;
    cudaStatus = setDevice();
    if (cudaStatus != cudaSuccess) {
        return cudaStatus;
    }
    // create the data in the device.
	DeviceParticleData deviceData;
    loadDataToDevice(data, deviceData, numParticles);


    double dt = solverOptions.initial_timestep;

    const int loggingSteps = loggingPeriod <= 0 ? numSteps / 100 : loggingPeriod;
   // std::vector<double> loggedSteps(numSteps / loggingSteps + 1, 0);

    /*HeliumBoundaryProblem<numParticles> boundaryProblem(properties);*/
    BoundaryIntegralCalculator<numParticles, batchSize> boundaryIntegrator(properties, boundaryProblem);

    // energy, constant functors
    KineticEnergyFunctor<numParticles, batchSize> kineticEnergy(boundaryProblem);
    PotentialEnergyFunctor<numParticles, batchSize> potentialEnergy(boundaryProblem);
    VolumeFluxFunctor<numParticles, batchSize> volumeFluxFunctor(boundaryIntegrator);


    std::shared_ptr<ValueLogger> kineticEnergyLogger = std::make_shared<ValueCallableLogger<KineticEnergyFunctor<numParticles, batchSize>>>(kineticEnergy, loggingSteps, numSteps / loggingSteps + 1);
    std::shared_ptr<ValueLogger> potentialEnergyLogger = std::make_shared<ValueCallableLogger<PotentialEnergyFunctor<numParticles, batchSize>>>(potentialEnergy, loggingSteps, numSteps / loggingSteps + 1);
    std::shared_ptr<ValueLogger> volumeFluxLogger = std::make_shared<ValueCallableLogger<VolumeFluxFunctor<numParticles, batchSize>>>(volumeFluxFunctor, loggingSteps, numSteps / loggingSteps + 1);
    
    
    TotalEnergyFunctor totalEnergyFunctor(kineticEnergyLogger, potentialEnergyLogger);
    
    std::shared_ptr<ValueLogger> totalEnergyLogger = std::make_shared<ValueCallableLogger<TotalEnergyFunctor>>(totalEnergyFunctor, loggingSteps, numSteps / loggingSteps + 1);





    // boundaryIntegrator.initialize_device(data.Z, data.Potential);

    DataLogger<std_complex, 2 * numParticles> stateLogger;
    stateLogger.setSize(numSteps / loggingSteps + 1);
    stateLogger.setStep(loggingSteps);

    TSolver solver(boundaryIntegrator, stateLogger, {kineticEnergyLogger, potentialEnergyLogger, volumeFluxLogger, totalEnergyLogger}, dt);
	solver.setOptions(solverOptions);
    solver.initialize(deviceData.devZ, true);
    //rungeKunta.setTimeStep(dt);
  //  for (;;) {
		//// inifinite loop to test if we still get the growing memory consumption with RK45
  //  }
    
    auto result = solver.runEvolution(0.0, dt * numSteps);
    if (result == OdeSolverResult::StiffnessDetected) {
        throw std::runtime_error("Stiff problem detected");
    }
    /*for (int i = 0; i < numSteps; ++i) {
        rungeKunta.runStep(i);

        if (kineticEnergyLogger.shouldLog(i)) {
            kineticEnergyLogger.logValue();
        }
        if (potentialEnergyLogger.shouldLog(i)) {
            potentialEnergyLogger.logValue();
        }
        if (volumeFluxLogger.shouldLog(i)) {
            volumeFluxLogger.logValue();
        }
        if (totalEnergyLogger.shouldLog(i)) {
            totalEnergyLogger.logValue();
        }
        if (i % loggingSteps == 0) {
            loggedSteps[i / loggingSteps] = i;
        }
    }*/
    
    

    if (simOptions.saveHDF5) {
        std::vector<std::vector<std_complex>> timeStepData = stateLogger.getAllData();
        std::vector<double> times = stateLogger.getTimes();

        if(timeStepData.size() == 0) 
        {
			return -1; // nothing to save!
		}
	    size_t vector_size = timeStepData[0].size();
        //    // Create HDF5 file
        HighFive::File file( "temp/" + simOptions.outputFilename, HighFive::File::Overwrite);
        //HighFive::DataSpace dataspace = HighFive::DataSpace({ (size_t)loggingSteps,  vector_size });

		// add properties to the file
		file.createAttribute<double>("rho", properties.rho);
		file.createAttribute<double>("kappa", properties.kappa);
		file.createAttribute<double>("U", properties.U);
		file.createAttribute<double>("depth", properties.depth);
		/*file.createAttribute<double>("y_min", properties.y_min);
		file.createAttribute<double>("y_max", properties.y_max);*/
		file.createAttribute<double>("dt0", dt);
		file.createAttribute<double>("initial_amplitude", properties.initial_amplitude);
		file.createAttribute<int>("numParticles", numParticles);
		file.createAttribute<int>("numSteps", numSteps);
		file.createAttribute<int>("loggingSteps", loggingSteps);

		// create user provided attributes
        for (const auto& [key, value] : simOptions.attributes) 
        {
			file.createAttribute<double>(key, value);
        }

        std::vector<std::array<double, 2>> row_tmp(vector_size);
        //row_tmp.reserve(2 * vector_size);

        for (size_t i = 0; i < timeStepData.size(); ++i) {
            for (size_t j = 0; j < timeStepData[i].size(); ++j) {
                row_tmp[j] = { timeStepData[i][j].real(), timeStepData[i][j].imag() };
            }

            HighFive::DataSpace space({ vector_size, 2 });
            HighFive::DataSet dataset = file.createDataSet<double>(std::format("i = {}", std::to_string(i * loggingSteps)), space);

            dataset.write(row_tmp);
	    }

        // writes the time information. This is a vector of doubles representing the time of each step. This is essential to have with RK45 since it's variable step!
        HighFive::DataSpace timeSpace({ times.size() });
        HighFive::DataSet timeDataSet = file.createDataSet<double>("times", timeSpace);
        timeDataSet.write(times);

        //dataset.write(reinterpret_cast<std::vector<std::vector<std::complex<double>>&>(timeStepData));
		// Create datasets for energy logs
        HighFive::DataSpace energySpace({ kineticEnergyLogger->getLoggedValuesCount() });
        HighFive::DataSet kineticEnergyDataset = file.createDataSet<double>("KineticEnergy", energySpace);
        kineticEnergyDataset.write(kineticEnergyLogger->getLoggedValues());

        
        HighFive::DataSet potentialEnergyDataset = file.createDataSet<double>("PotentialEnergy", energySpace);
        potentialEnergyDataset.write(potentialEnergyLogger->getLoggedValues());

        HighFive::DataSet totalEnergyDataset = file.createDataSet<double>("TotalEnergy", energySpace);
        totalEnergyDataset.write(totalEnergyLogger->getLoggedValues());

        HighFive::DataSet volumeFluxDataset = file.createDataSet<double>("VolumeFlux", energySpace);
		volumeFluxDataset.write(volumeFluxLogger->getLoggedValues());
		file.flush();
    }
    

    
    if (simOptions.createVideo && !simOptions.outputFilename.empty()) {
        std::vector<std::vector<std_complex>> timeStepData = stateLogger.getAllData();
        std::vector<double> times = stateLogger.getTimes();

        std::vector<std::string> paths;
        std::vector<std::string> pathsPotential;
        createFrames<numParticles>(timeStepData, times, loggingSteps, paths, 640, 480, properties.y_min, properties.y_max);
        createPotentialFrames<numParticles>(timeStepData, times, loggingSteps, pathsPotential, 640, 480);
        createVideo(std::format("temp/frames/{}.avi", simOptions.videoFilename), 640, 480, paths, fps);
        createVideo(std::format("temp/frames/{}_pot.avi", simOptions.videoFilename), 640, 480, pathsPotential, fps);
    }

    if (plot)
    {
        std::vector<std::vector<std_complex>> timeStepData = stateLogger.getAllData();
        std::vector<double> times = stateLogger.getTimes();

        plt::figure();
        auto title = "Interface And Potential";
        plt::title(title);

        for (int i = 0; i < timeStepData.size(); i++) {
            auto& stepData = timeStepData[i];
            std::vector<double> x_step(numParticles, 0);
            std::vector<double> y_step(numParticles, 0);
            for (int j = 0; j < numParticles; j++) {
                x_step[j] = stepData[j].real();
                y_step[j] = stepData[j].imag();
            }
            plt::plot(x_step, y_step, { {"label", "Interface at t=" + std::to_string(i * loggingSteps * dt)} });
        }

        plt::legend();

        plt::figure();
        plt::title("Kinetic, Potential and Total Energy");
        plt::plot(times, kineticEnergyLogger->getLoggedValues(), { {"label", "Kinetic Energy"} });
        plt::plot(times, potentialEnergyLogger->getLoggedValues(), { {"label", "Potential Energy"} });
        plt::plot(times, totalEnergyLogger->getLoggedValues(), { {"label", "Total Energy"} });

        //plt::xlabel("Time Steps");
        //plt::ylabel("Energy");

        plt::figure();
        plt::title("Volume Flux");
        plt::plot(times, volumeFluxLogger->getLoggedValues(), { {"label", "Volume Flux"} });
        plt::xlabel("Time (a.u.)");
        plt::ylabel("Volume Flux");
        plt::legend();
        if (show)
            plt::show();
    }
    if (plot || simOptions.createVideo) {
        plt::close();
        plt::cla();
    }
	// free device memory
    freeDeviceData(deviceData);
    
	return 0;
}

template<int numParticles, size_t batchSize, typename TSolver, typename TOptions>
requires std::derived_from<std::remove_cvref_t<TSolver>, OdeSolver>
int runSimulationHelium(const int numSteps, ProblemProperties& properties, ParticleData data, TOptions rk45Options, SimulationOptions simOptions,  const int loggingPeriod = -1, const bool plot = true, const bool show = true, double t0 = 1.0, int fps = 10) {
    
    HeliumBoundaryProblem<numParticles, batchSize> boundaryProblem(properties);
    return runSimulation<numParticles, batchSize, TSolver, TOptions>(boundaryProblem, numSteps, properties, data, rk45Options, simOptions, loggingPeriod, plot, show, t0, fps);
}

template <int numParticles, size_t batchSize>
int runSimulationWater(const int numSteps, ProblemProperties& properties, ParticleData data, RK45_Options rk45Options, SimulationOptions simOptions, const int loggingPeriod = -1, const bool plot = true, const bool show = true, double t0 = 1.0, int fps = 10) {
    
    WaterBoundaryProblem<numParticles, batchSize> boundaryProblem(properties);
    return runSimulation<numParticles, batchSize>(boundaryProblem, numSteps, properties, data, rk45Options, simOptions, loggingPeriod, plot, show, t0, fps);
}

template <int numParticles, size_t batchSize>
int runSimulationHeliumInfiniteDepth(const int numSteps, ProblemProperties& properties, ParticleData data, RK45_Options rk45Options, SimulationOptions simOptions, const int loggingPeriod = -1, const bool plot = true, const bool show = true, double t0 = 1.0, int fps = 10) {
    
    HeliumInfiniteDepthBoundaryProblem<numParticles, batchSize> boundaryProblem(properties);
    return runSimulation<numParticles, batchSize>(boundaryProblem, numSteps, properties, data, rk45Options, simOptions, loggingPeriod, plot, show, t0, fps);
}