#pragma once  

#include <opencv2/opencv.hpp>  
#include <filesystem>  
#include <vector>
#include <iostream>
#include <string>

namespace fs = std::filesystem;  

void createVideo(const std::string name, int width, int height, std::vector<std::string>& paths_pngs, int fps = 30) {
	cv::VideoWriter writer(name, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
		fps,  
		cv::Size(width, height));  
	if (!writer.isOpened()) {
		throw std::runtime_error("Could not open video writer\n");
	}

	for (auto& path : paths_pngs) 
	{
		cv::Mat img = cv::imread(path);
		if (img.empty()) {
			throw std::runtime_error("Could not read image: " + path + "\n");
		}

		// resize the image to the desired width and height
		cv::resize(img, img, cv::Size(width, height));
		writer.write(img);
	}
	writer.release();
}
template <size_t N>
void createFrames(std::vector<std::vector<std_complex>>& data, double dt, int periodLogging, std::vector<std::string>& paths, int width = 640, int height = 480, double ylim1 = -0.1, double ylim2 = 0.1) 
{
	
	paths.clear();
	int i = 0;
	std::cout <<  fs::current_path().string();
	fs::create_directories("temp/frames");
	plt::figure();
	plt::figure_size(width, height);

	for(auto& frame : data) 
	{
		std::vector<double> x(N, 0);
		std::vector<double> y(N, 0);
		
		for(int i = 0; i < N; ++i) 
		{
			x[i] = (frame[i].real());
			y[i] = (frame[i].imag());
		}
		plt::clf();  // Clear the current figure

		plt::plot(x, y, { {"label", "Interface"} });
		plt::ylim(ylim1, ylim2);
		plt::title( std::format("Interface at t={:.10e}" , i * periodLogging * dt));

		paths.push_back(std::format("temp/frames/frame_{}.png", i));

		plt::save(paths.back());

		i++;
	}
	plt::clf();
}

template <size_t N>
void createPotentialFrames(std::vector<std::vector<std_complex>>& data, double dt, int periodLogging, std::vector<std::string>& paths, int width = 640, int height = 480, double ylim1 = -0.1, double ylim2 = 0.1) 
{
	paths.clear();
	int i = 0;
	std::cout << fs::current_path().string();
	fs::create_directories("temp/frames");
	plt::figure();
	plt::figure_size(width, height);

	for (auto& frame : data)
	{
		std::vector<double> x(N, 0);
		std::vector<double> y(N, 0);

		for (int i = 0; i < N; ++i)
		{
			x[i] = (frame[i].real());
			y[i] = (frame[i+N].real());
		}
		plt::clf();  // Clear the current figure

		plt::plot(x, y, { {"label", "Potential"} });
		// plt::ylim(ylim1, ylim2);
		plt::title(std::format("Potential at t={:.10e}", i * periodLogging * dt));

		paths.push_back(std::format("temp/frames/frame_pot_{}.png", i));

		plt::save(paths.back());

		i++;
	}
	plt::clf();
}


