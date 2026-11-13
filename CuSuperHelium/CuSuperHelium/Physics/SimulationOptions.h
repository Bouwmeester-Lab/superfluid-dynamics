#pragma once
#include <complex>
#include <map>
#include <string>
#include <vector>
#include "Core/constants.cuh"
#include "Core/DeviceBuffer.cuh"

struct SimulationOptions
{
	std::string outputFilename = "output.h5";
	/// <summary>
	/// Video filename without extension.
	/// </summary>
	std::string videoFilename = "output";
	std::map<std::string, double> attributes;
	bool createVideo = false;
	bool saveHDF5 = true;
};

struct ParticleData {
	std::vector<std::complex<double>>& Z; // vector of particle positions
	std::vector<double>& Potential; // vector of particle potentials
	ParticleData(std::vector<std::complex<double>>& positions, std::vector<double>& potential) : Z(positions), Potential(potential) {}
};

/// <summary>
/// Device copy of the particle state, filled by loadDataToDevice: positions, then potentials.
/// Owns its memory, which is freed when the object goes out of scope.
/// </summary>
struct DeviceParticleData
{
	DeviceBuffer<std_complex> buffer;
	std_complex* devZ() const noexcept { return buffer.get(); }
};