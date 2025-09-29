#pragma once
#include <string>
#include "constants.cuh"

struct SimulationOptions
{
	std::string outputFilename = "output.h5";
	/// <summary>
	/// Video filename without extension.
	/// </summary>
	std::string videoFilename = "output";
	std::map<std::string, double> attributes;
	bool createVideo = false;
};

struct ParticleData {
	std::vector<std::complex<double>>& Z; // vector of particle positions
	std::vector<double>& Potential; // vector of particle potentials
	ParticleData(std::vector<std::complex<double>>& positions, std::vector<double>& potential) : Z(positions), Potential(potential) {}
};

struct DeviceParticleData
{
	std_complex* devZ; // device pointer to particle positions and potentials
};