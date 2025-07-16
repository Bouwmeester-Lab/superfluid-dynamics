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

template<int numParticles>
int runSimulationHelium(const int numSteps, double dt, ProblemProperties& properties, ParticleData data, const int loggingPeriod = -1, const bool plot = true, const bool show = true) {
    cudaError_t cudaStatus;
    cudaStatus = setDevice();
    if (cudaStatus != cudaSuccess) {
        return cudaStatus;
    }

    
	const int loggingSteps = loggingPeriod < 0 ? numSteps / 10 : loggingPeriod;
    std::vector<double> loggedSteps(numSteps / loggingSteps + 1, 0);

    HeliumBoundaryProblem<numParticles> boundaryProblem(properties);
    BoundaryIntegralCalculator<numParticles> timeStepManager(properties, boundaryProblem);

    ValueLogger kineticEnergyLogger(loggingSteps);
    ValueLogger potentialEnergyLogger(loggingSteps);
    ValueLogger surfaceEnergyLogger(loggingSteps);
    ValueLogger totalEnergyLogger(loggingSteps);
    ValueLogger volumeFluxLogger(loggingSteps);



	timeStepManager.initialize_device(data.Z, data.Potential);

    DataLogger<std_complex, 2 * numParticles> stateLogger;
    stateLogger.setSize(numSteps / loggingSteps + 1);
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

    auto& timeStepData = stateLogger.getAllData();

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
            plt::plot(x_step, y_step, { {"label", "Interface at t=" + std::to_string(i * dt)} });
        }
        plt::legend();

        plt::figure();
        plt::title("Kinetic, Potential and Total Energy");
        plt::plot(loggedSteps, kineticEnergyLogger.getLoggedValues(), { {"label", "Kinetic Energy"} });
        plt::plot(loggedSteps, potentialEnergyLogger.getLoggedValues(), { {"label", "Potential Energy"} });
        plt::plot(loggedSteps, totalEnergyLogger.getLoggedValues(), { {"label", "Total Energy"} });

        plt::xlabel("Time Steps");
        plt::ylabel("Energy");

        plt::figure();
        plt::title("Volume Flux");
        plt::plot(loggedSteps, volumeFluxLogger.getLoggedValues(), { {"label", "Volume Flux"} });
        plt::xlabel("Time Steps");
        plt::ylabel("Volume Flux");
        plt::legend();
        if(show)
			plt::show();
    }
}