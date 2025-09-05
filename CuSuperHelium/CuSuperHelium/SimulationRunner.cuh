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

namespace plt = matplotlibcpp;

cudaError_t setDevice()
{

    cudaError_t cudaStatus;
    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return cudaStatus;
    }

    return cudaStatus;
}

struct ParticleData {
    std::vector<std::complex<double>>& Z; // vector of particle positions
    std::vector<double>& Potential; // vector of particle potentials
	ParticleData(std::vector<std::complex<double>>& positions, std::vector<double>& potential) : Z(positions), Potential(potential) {}
};

struct DeviceParticleData 
{
	std_complex* devZ; // device pointer to particle positions and potentials
};

template <int N>
struct KineticEnergyFunctor {
private:
    BoundaryProblem<N>& boundaryProblem;
public:
    KineticEnergyFunctor(BoundaryProblem<N>& boundaryProblem) : boundaryProblem(boundaryProblem)
    {

    }

    double operator()() 
    {
        return boundaryProblem.energyContainer.kineticEnergy->getEnergy();
    }
};

template <int N>
struct PotentialEnergyFunctor {
private:
    BoundaryProblem<N>& boundaryProblem;
public:
    PotentialEnergyFunctor(BoundaryProblem<N>& boundaryProblem) : boundaryProblem(boundaryProblem)
    {

    }

    double operator()()
    {
        return boundaryProblem.energyContainer.potentialEnergy->getEnergy();
    }
};

template <int N>
struct VolumeFluxFunctor {
private:
    BoundaryIntegralCalculator<N>& integralCalculator;
public:
    VolumeFluxFunctor(BoundaryIntegralCalculator<N>& integralCalculator) : integralCalculator(integralCalculator)
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

    auto dataset = file.getDataSet("interface");
    auto potential = file.getDataSet("potential");

    // check dimentsions and make sure its N x 2
    auto dims = dataset.getSpace().getDimensions();
    if (dims.size() != 2 || dims[1] != 2 || dims[0] != N) {
        throw std::runtime_error(std::format("Interface must have shape ({}, 2)", N));
    }

    // check dimensions for potential
    auto potDims = dataset.getSpace().getDimensions();
    if (dims.size() == 0) {
        throw std::runtime_error("potential dataset is empty.");
    }
    std::vector<std::array<double, 2>> dataInterface;
	//std::vector<double> dataPotential;

    phi.clear();
	potential.read(phi);

	// read the interface data
    dataset.read(dataInterface);

	// convert to complex numbers
    Z.clear();
    Z.resize(dataInterface.size());
    for (size_t i = 0; i < dataInterface.size(); ++i) {
        Z[i] = std::complex<double>(dataInterface[i][0], scaleY * dataInterface[i][1]);
    }
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

	std::vector<double> X(N);
	std::vector<double> Y(N);
	std::vector<double> Phi(N);
	std::vector<double> PhiImag(N);

    for (size_t i = 0; i < N; ++i)
    {
		X[i] = checkZ[i].real();
		Y[i] = checkZ[i].imag();
		Phi[i] = checkZ[i + N].real();
		PhiImag[i] = checkZ[i + N].imag(); // should be 0
    }
	/*plt::figure();
	plt::plot(X, Y);
	plt::plot(X, Phi);
	plt::plot(X, PhiImag);
	plt::title("Initial Condition");
	plt::show();*/
    return 0;
}

template <int numParticles>
int runSimulation(BoundaryProblem<numParticles>& boundaryProblem, const int numSteps, ProblemProperties& properties, ParticleData data, RK45_Options rk45Options, SimulationOptions simOptions, const int loggingPeriod = -1, const bool plot = true, const bool show = true, double t0 = 1.0, int fps = 10, const bool saveH5 = true)
{ 
    cudaError_t cudaStatus;
    cudaStatus = setDevice();
    if (cudaStatus != cudaSuccess) {
        return cudaStatus;
    }
    // create the data in the device.
	DeviceParticleData deviceData;
    loadDataToDevice(data, deviceData, numParticles);


    double dt = rk45Options.initial_timestep;

    const int loggingSteps = loggingPeriod < 0 ? numSteps / 20 : loggingPeriod;
   // std::vector<double> loggedSteps(numSteps / loggingSteps + 1, 0);

    /*HeliumBoundaryProblem<numParticles> boundaryProblem(properties);*/
    BoundaryIntegralCalculator<numParticles> boundaryIntegrator(properties, boundaryProblem);

    // energy, constant functors
    KineticEnergyFunctor<numParticles> kineticEnergy(boundaryProblem);
    PotentialEnergyFunctor<numParticles> potentialEnergy(boundaryProblem);
    VolumeFluxFunctor<numParticles> volumeFluxFunctor(boundaryIntegrator);


    std::shared_ptr<ValueLogger> kineticEnergyLogger = std::make_shared<ValueCallableLogger<KineticEnergyFunctor<numParticles>>>(kineticEnergy, loggingSteps, numSteps / loggingSteps + 1);
    std::shared_ptr<ValueLogger> potentialEnergyLogger = std::make_shared<ValueCallableLogger<PotentialEnergyFunctor<numParticles>>>(potentialEnergy, loggingSteps, numSteps / loggingSteps + 1);
    std::shared_ptr<ValueLogger> volumeFluxLogger = std::make_shared<ValueCallableLogger<VolumeFluxFunctor<numParticles>>>(volumeFluxFunctor, loggingSteps, numSteps / loggingSteps + 1);
    
    
    TotalEnergyFunctor totalEnergyFunctor(kineticEnergyLogger, potentialEnergyLogger);
    
    std::shared_ptr<ValueLogger> totalEnergyLogger = std::make_shared<ValueCallableLogger<TotalEnergyFunctor>>(totalEnergyFunctor, loggingSteps, numSteps / loggingSteps + 1);





    // boundaryIntegrator.initialize_device(data.Z, data.Potential);

    DataLogger<std_complex, 2 * numParticles> stateLogger;
    stateLogger.setSize(numSteps / loggingSteps + 1);
    stateLogger.setStep(loggingSteps);

    RK45_std_complex<2 * numParticles> rungeKutta(boundaryIntegrator, stateLogger, {kineticEnergyLogger, potentialEnergyLogger, volumeFluxLogger, totalEnergyLogger}, dt);
    rungeKutta.setTolerance(rk45Options.atol, rk45Options.rtol);
    rungeKutta.initialize(deviceData.devZ, true);
    rungeKutta.setMaxTimeStep(rk45Options.h_max);
	rungeKutta.setMinTimeStep(rk45Options.h_min);
    //rungeKunta.setTimeStep(dt);
  //  for (;;) {
		//// inifinite loop to test if we still get the growing memory consumption with RK45
  //  }

    auto result = rungeKutta.runEvolution(numSteps, dt * numSteps);
    if (result == RK45Result::StiffnessDetected) {
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

    std::vector<std::vector<std_complex>>& timeStepData = stateLogger.getAllData();
    std::vector<double>& times = stateLogger.getTimes();

    if (saveH5) {
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
    }
    

    
    if (simOptions.createVideo && !simOptions.outputFilename.empty()) {
        std::vector<std::string> paths;
        std::vector<std::string> pathsPotential;
        createFrames<numParticles>(timeStepData, times, loggingSteps, paths, 640, 480, properties.y_min, properties.y_max);
        createPotentialFrames<numParticles>(timeStepData, times, loggingSteps, pathsPotential, 640, 480);
        createVideo(std::format("temp/frames/{}.avi", simOptions.videoFilename), 640, 480, paths, fps);
        createVideo(std::format("temp/frames/{}_pot.avi", simOptions.videoFilename), 640, 480, pathsPotential, fps);
    }

    if (plot)
    {
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
	return 0;
}

template<int numParticles>
int runSimulationHelium(const int numSteps, ProblemProperties& properties, ParticleData data, RK45_Options rk45Options, SimulationOptions simOptions,  const int loggingPeriod = -1, const bool plot = true, const bool show = true, double t0 = 1.0, int fps = 10) {
    
    HeliumBoundaryProblem<numParticles> boundaryProblem(properties);
    return runSimulation<numParticles>(boundaryProblem, numSteps, properties, data, rk45Options, simOptions, loggingPeriod, plot, show, t0, fps);
}

template <int numParticles>
int runSimulationWater(const int numSteps, ProblemProperties& properties, ParticleData data, RK45_Options rk45Options, SimulationOptions simOptions, const int loggingPeriod = -1, const bool plot = true, const bool show = true, double t0 = 1.0, int fps = 10) {
    
    WaterBoundaryProblem<numParticles> boundaryProblem(properties);
    return runSimulation<numParticles>(boundaryProblem, numSteps, properties, data, rk45Options, simOptions, loggingPeriod, plot, show, t0, fps);
}

template <int numParticles>
int runSimulationHeliumInfiniteDepth(const int numSteps, ProblemProperties& properties, ParticleData data, RK45_Options rk45Options, SimulationOptions simOptions, const int loggingPeriod = -1, const bool plot = true, const bool show = true, double t0 = 1.0, int fps = 10) {
    
    HeliumInfiniteDepthBoundaryProblem<numParticles> boundaryProblem(properties);
    return runSimulation<numParticles>(boundaryProblem, numSteps, properties, data, rk45Options, simOptions, loggingPeriod, plot, show, t0, fps);
}