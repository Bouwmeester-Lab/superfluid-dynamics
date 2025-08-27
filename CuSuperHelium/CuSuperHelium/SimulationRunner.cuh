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
    std_complex* Z; // Array of particle positions
    std_complex* Potential; // Array of particle potentials
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

template <int numParticles>
int runSimulation(BoundaryProblem<numParticles>& boundaryProblem, const int numSteps, double dt, ProblemProperties& properties, ParticleData data, const int loggingPeriod = -1, const bool plot = true, const bool show = true, double t0 = 1.0, int fps = 10, const bool saveH5 = true)
{ 
    cudaError_t cudaStatus;
    cudaStatus = setDevice();
    if (cudaStatus != cudaSuccess) {
        return cudaStatus;
    }


    const int loggingSteps = loggingPeriod < 0 ? numSteps / 20 : loggingPeriod;
    std::vector<double> loggedSteps(numSteps / loggingSteps + 1, 0);

    /*HeliumBoundaryProblem<numParticles> boundaryProblem(properties);*/
    BoundaryIntegralCalculator<numParticles> boundaryIntegrator(properties, boundaryProblem);

    // energy, constant functors
    KineticEnergyFunctor<numParticles> kineticEnergy(boundaryProblem);
    PotentialEnergyFunctor<numParticles> potentialEnergy(boundaryProblem);
    VolumeFluxFunctor<numParticles> volumeFluxFunctor(boundaryIntegrator);


    std::shared_ptr<ValueLogger> kineticEnergyLogger = std::make_shared<ValueCallableLogger<KineticEnergyFunctor<numParticles>>>(kineticEnergy, loggingSteps);
    std::shared_ptr<ValueLogger> potentialEnergyLogger = std::make_shared<ValueCallableLogger<PotentialEnergyFunctor<numParticles>>>(potentialEnergy, loggingSteps);
    std::shared_ptr<ValueLogger> volumeFluxLogger = std::make_shared<ValueCallableLogger<VolumeFluxFunctor<numParticles>>>(volumeFluxFunctor, loggingSteps);
    
    
    TotalEnergyFunctor totalEnergyFunctor(kineticEnergyLogger, potentialEnergyLogger);
    
    std::shared_ptr<ValueLogger> totalEnergyLogger = std::make_shared<ValueCallableLogger<TotalEnergyFunctor>>(totalEnergyFunctor, loggingSteps);





    boundaryIntegrator.initialize_device(data.Z, data.Potential);

    DataLogger<std_complex, 2 * numParticles> stateLogger;
    stateLogger.setSize(numSteps / loggingSteps);
    stateLogger.setStep(loggingSteps);

    RK45_std_complex<2 * numParticles> rungeKunta(boundaryIntegrator, stateLogger, {kineticEnergyLogger, potentialEnergyLogger, volumeFluxLogger, totalEnergyLogger}, dt);

    rungeKunta.initialize(boundaryIntegrator.getY0());
    //rungeKunta.setTimeStep(dt);
    rungeKunta.runEvolution(numSteps, dt * numSteps);

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

    if (saveH5) {
	    size_t vector_size = timeStepData[0].size();
        //    // Create HDF5 file
        HighFive::File file("temp/data.h5", HighFive::File::Overwrite);
        //HighFive::DataSpace dataspace = HighFive::DataSpace({ (size_t)loggingSteps,  vector_size });

		// add properties to the file
		file.createAttribute<double>("rho", properties.rho);
		file.createAttribute<double>("kappa", properties.kappa);
		file.createAttribute<double>("U", properties.U);
		file.createAttribute<double>("depth", properties.depth);
		/*file.createAttribute<double>("y_min", properties.y_min);
		file.createAttribute<double>("y_max", properties.y_max);*/
		file.createAttribute<double>("dt", dt);
		file.createAttribute<double>("initial_amplitude", properties.initial_amplitude);
		file.createAttribute<int>("numParticles", numParticles);
		file.createAttribute<int>("numSteps", numSteps);
		file.createAttribute<int>("loggingSteps", loggingSteps);

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
    

    if (plot)
    {
        std::vector<std::string> paths;
        std::vector<std::string> pathsPotential;
        createFrames<numParticles>(timeStepData, dt * t0, loggingSteps, paths, 640, 480, properties.y_min, properties.y_max);
		createPotentialFrames<numParticles>(timeStepData, dt * t0, loggingSteps, pathsPotential, 640, 480);
        createVideo("temp/frames/video.avi", 640, 480, paths, fps);
        createVideo("temp/frames/video_pot.avi", 640, 480, pathsPotential, fps);
        //plt::figure();
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
        plt::plot(loggedSteps, kineticEnergyLogger->getLoggedValues(), { {"label", "Kinetic Energy"} });
        plt::plot(loggedSteps, potentialEnergyLogger->getLoggedValues(), { {"label", "Potential Energy"} });
        plt::plot(loggedSteps, totalEnergyLogger->getLoggedValues(), { {"label", "Total Energy"} });

        //plt::xlabel("Time Steps");
        //plt::ylabel("Energy");

        plt::figure();
        plt::title("Volume Flux");
        plt::plot(loggedSteps, volumeFluxLogger->getLoggedValues(), { {"label", "Volume Flux"} });
        plt::xlabel("Time Steps");
        plt::ylabel("Volume Flux");
        plt::legend();
        if (show)
            plt::show();
    }
	return 0;
}

template<int numParticles>
int runSimulationHelium(const int numSteps, double dt, ProblemProperties& properties, ParticleData data, const int loggingPeriod = -1, const bool plot = true, const bool show = true, double t0 = 1.0, int fps = 10) {
    
    HeliumBoundaryProblem<numParticles> boundaryProblem(properties);
    return runSimulation<numParticles>(boundaryProblem, numSteps, dt, properties, data, loggingPeriod, plot, show, t0, fps);
}

template <int numParticles>
int runSimulationWater(const int numSteps, double dt, ProblemProperties& properties, ParticleData data, const int loggingPeriod = -1, const bool plot = true, const bool show = true, double t0 = 1.0, int fps = 10) {
    
    WaterBoundaryProblem<numParticles> boundaryProblem(properties);
    return runSimulation<numParticles>(boundaryProblem, numSteps, dt, properties, data, loggingPeriod, plot, show, t0, fps);
}

template <int numParticles>
int runSimulationHeliumInfiniteDepth(const int numSteps, double dt, ProblemProperties& properties, ParticleData data, const int loggingPeriod = -1, const bool plot = true, const bool show = true, double t0 = 1.0, int fps = 10) {
    
    HeliumInfiniteDepthBoundaryProblem<numParticles> boundaryProblem(properties);
    return runSimulation<numParticles>(boundaryProblem, numSteps, dt, properties, data, loggingPeriod, plot, show, t0, fps);
}