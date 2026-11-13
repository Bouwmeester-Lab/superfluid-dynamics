#pragma once


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "ProblemProperties.hpp"
#include "cufft.h"
#include "constants.cuh"
#include "Derivatives.cuh"
#include "WaterVelocities.cuh"
#include "MatrixSolver.cuh"
#include "AutonomousProblem.h"
#include "Energies.cuh"
#include "DelayedIntensityTerm.cuh"
#include "BoundaryProblem.cuh"
#include "BaseBoundaryIntegrator.cuh"
#include "TimedBoundaryIntegrator.cuh"
#include "SimpleEuler.cuh"
#include "AutonomousRungeKuttaStepper.cuh"
#include "ValueLogger.h"
#include "matplotlibcpp.h"
#include "VideoMaking.h"
#include <highfive/H5File.hpp>
#include "StateFile.h"
#include "RK45.cuh"
#include "SimulationOptions.h"
#include "ExportTypes.cuh"
#include "DeviceParticleLoading.cuh"
#include "HeliumBoundaryProblem.cuh"
#include "WaterBoundaryProblem.cuh"
//#include "H"

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
    BaseBoundaryIntegralCalculator<N, batchSize>& integralCalculator;
public:
    VolumeFluxFunctor(BaseBoundaryIntegralCalculator<N, batchSize>& integralCalculator) : integralCalculator(integralCalculator)
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
    BaseBoundaryIntegralCalculator<numParticles, batchSize> boundaryIntegrator(properties, boundaryProblem);

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
     //TODO: value loggers to update: { kineticEnergyLogger, potentialEnergyLogger, volumeFluxLogger, totalEnergyLogger }
    TSolver solver(boundaryIntegrator, dt);
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
