#pragma once
#include <string>

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
