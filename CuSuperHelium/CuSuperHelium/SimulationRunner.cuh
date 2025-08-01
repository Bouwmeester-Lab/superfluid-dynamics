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
    BoundaryIntegralCalculator<numParticles> timeStepManager(properties, boundaryProblem);

    ValueLogger kineticEnergyLogger(loggingSteps);
    ValueLogger potentialEnergyLogger(loggingSteps);
    ValueLogger surfaceEnergyLogger(loggingSteps);
    ValueLogger totalEnergyLogger(loggingSteps);
    ValueLogger volumeFluxLogger(loggingSteps);



    timeStepManager.initialize_device(data.Z, data.Potential);

    DataLogger<std_complex, 2 * numParticles> stateLogger;
    stateLogger.setSize(numSteps / loggingSteps);
    stateLogger.setStep(loggingSteps);

    AutonomousRungeKuttaStepper<std_complex, 2 * numParticles> rungeKunta(timeStepManager, stateLogger, dt);

    rungeKunta.initialize(timeStepManager.getY0());
    rungeKunta.setTimeStep(dt);

    for (int i = 0; i < numSteps; ++i) {
        rungeKunta.runStep(i);

        if (kineticEnergyLogger.shouldLog(i)) {
            kineticEnergyLogger.logValue(boundaryProblem.energyContainer.kineticEnergy->getEnergy());
        }
        if (potentialEnergyLogger.shouldLog(i)) {
            potentialEnergyLogger.logValue(boundaryProblem.energyContainer.potentialEnergy->getEnergy());
        }
        if (volumeFluxLogger.shouldLog(i)) {
            volumeFluxLogger.logValue(timeStepManager.volumeFlux.getEnergy());
        }
        if (totalEnergyLogger.shouldLog(i)) {
            totalEnergyLogger.logValue(kineticEnergyLogger.getLastLoggedValue() + potentialEnergyLogger.getLastLoggedValue());
        }
        if (i % loggingSteps == 0) {
            loggedSteps[i / loggingSteps] = i;
        }
    }

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
        plt::plot(loggedSteps, kineticEnergyLogger.getLoggedValues(), { {"label", "Kinetic Energy"} });
        plt::plot(loggedSteps, potentialEnergyLogger.getLoggedValues(), { {"label", "Potential Energy"} });
        plt::plot(loggedSteps, totalEnergyLogger.getLoggedValues(), { {"label", "Total Energy"} });

        //plt::xlabel("Time Steps");
        //plt::ylabel("Energy");

        plt::figure();
        plt::title("Volume Flux");
        plt::plot(loggedSteps, volumeFluxLogger.getLoggedValues(), { {"label", "Volume Flux"} });
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