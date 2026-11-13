#pragma once
#ifndef STATE_FILE_H
#define STATE_FILE_H

#include <array>
#include <complex>
#include <format>
#include <stdexcept>
#include <string>
#include <vector>

#include <highfive/H5File.hpp>
#include "Physics/ProblemProperties.hpp"

/// <summary>
/// Loads the data in an H5 file containing two datasets with 2 columns each (x, y) pairs representing complex numbers.
/// The first dataset must be "interface" representing the interface shape.
/// The second dataset must be "potential" representing the potential (only the first column is used).
/// </summary>
/// <param name="path"></param>
/// <param name="Z"></param>
/// <param name="phi"></param>
/// <returns></returns>
inline int loadStateFile(const std::string path, std::vector<std::complex<double>>& Z, std::vector<double>& phi, const size_t N, const double scaleY = 1.0) 
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
inline void createPropertiesAttributes(HighFive::File& file, const ProblemProperties& properties) 
{
    // add properties to the file
    file.createAttribute<double>("rho", properties.rho);
    file.createAttribute<double>("kappa", properties.kappa);
    file.createAttribute<double>("U", properties.U);
    file.createAttribute<double>("depth", properties.depth);
	file.createAttribute<double>("initial_amplitude", properties.initial_amplitude);
}

inline int saveStateFile(const std::string path, std::vector<std::complex<double>>& Z, std::vector<double>& phi, const size_t N, ProblemProperties properties, bool rhs = false)
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

#endif // STATE_FILE_H
