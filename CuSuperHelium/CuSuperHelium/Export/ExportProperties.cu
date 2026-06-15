#include "ExportInternal.cuh"

#include <cmath>

void copyProperties(SimProperties& simProperties, ProblemProperties& properties)
{
	properties.L = simProperties.L;
	properties.rho = simProperties.rho;
	properties.kappa = simProperties.kappa;
	properties.depth = simProperties.depth;

	properties.expansion_order = simProperties.expansion_order;
	properties.use_expansions = simProperties.use_expansions;

	properties.infinite_depth = simProperties.infinite_depth;
}

void printOptomechanicalVariables(OptomechanicalVariables& variables)
{
	std::cout << "detuning: " << variables.detuning << std::endl;
	std::cout << "max_intensity: " << variables.max_intensity << std::endl;
	std::cout << "gamma: " << variables.gamma << std::endl;

	std::cout << "G: " << variables.G << std::endl;
	std::cout << "tau: " << variables.Tau << std::endl;
	std::cout << "x0 mode: " << variables.location_x0_mode << std::endl;
	std::cout << "sigma optical mode: " << variables.sigma_optical_mode << std::endl;
	std::cout << "sigma thermal mode: " << variables.sigma_thermal_mode << std::endl;

	std::cout << "beta: " << variables.Beta << std::endl;
	std::cout << "damping: " << variables.DampingStrength << std::endl;
}

void copyProperties(COptomechanicalVariables& c_optomechanicalVariables, OptomechanicalVariables& opto_variables) 
{
	opto_variables.detuning = c_optomechanicalVariables.detuning;
	opto_variables.gamma = c_optomechanicalVariables.gamma;
	opto_variables.G = c_optomechanicalVariables.G;
	opto_variables.Tau = c_optomechanicalVariables.tau;
	opto_variables.max_intensity = c_optomechanicalVariables.max_intensity;
	opto_variables.initial_time = c_optomechanicalVariables.initial_time;
	opto_variables.location_x0_mode = c_optomechanicalVariables.location_x0_mode;
	opto_variables.sigma_optical_mode = c_optomechanicalVariables.sigma_optical_mode;
	opto_variables.sigma_thermal_mode = c_optomechanicalVariables.sigma_thermal_mode;
	opto_variables.Beta = c_optomechanicalVariables.beta;
	opto_variables.DampingStrength = c_optomechanicalVariables.damping_strength;

	opto_variables.ramp_intensity = c_optomechanicalVariables.ramp_intensity;
	opto_variables.ramp_rate = c_optomechanicalVariables.ramp_rate; // (intensity / seconds)
}

RK4SolverOptions adimensionalizeRK4SolverOptions(RK4SolverOptions options, ProblemProperties properties)
{
	options.t0 /= properties.base_time;
	options.t1 /= properties.base_time;
	options.timeStep /= properties.base_time;

	return options;
}

ProblemProperties adimensionalizeProperties(ProblemProperties props, double rhoHelium)
{
	// calculate base units
	props.base_length = props.L / (2.0 * PI_d); // characteristic length
	props.base_acceleration = 3 * alpha_hamaker_d / std::pow(props.depth, 4);
	props.base_time = std::sqrt(props.base_length / props.base_acceleration);
	props.base_energy = 3.0 * rhoHelium * alpha_hamaker_d * std::pow(props.base_length, 4) / std::pow(props.depth, 4);

	double surfaceTensionFactor = rhoHelium * props.base_length * props.base_length * props.base_length / (props.base_time * props.base_time);

	// adimensionalize properties
	props.kappa = props.kappa / surfaceTensionFactor;
	props.depth = props.depth / props.base_length;

	props.rho /= rhoHelium; 

	std::cout << "Adimensionalized properties: " << std::endl;
	std::cout << "rho: " << props.rho << std::endl;
	std::cout << "kappa: " << props.kappa << std::endl;
	std::cout << "depth: " << props.depth << std::endl;

	return props;
}

OptomechanicalVariables adimensionalizeOptomechanicalVariables(OptomechanicalVariables optomechanicalVariables, ProblemProperties properties, double rhoHelium)
{
	// adimensionalize optomechanical variables
	// gamma is a frequency , so it gets multiplied by time
	optomechanicalVariables.gamma *= properties.base_time;
	// detuning is also a frequency
	optomechanicalVariables.detuning *= properties.base_time;
	// G is a coupling strength (Hz / m), so it gets multiplied by time and base_length
	optomechanicalVariables.G *= properties.base_time * properties.base_length;
	
	// Tau is a time delay, so it gets divided by base_time
	optomechanicalVariables.Tau /= properties.base_time;

	// location_x0_mode is a length, so it gets divided by base_length
	optomechanicalVariables.location_x0_mode /= properties.base_length;
	// sigma_optical_mode is also a length, so it gets divided by base_length
	optomechanicalVariables.sigma_optical_mode /= properties.base_length;
	optomechanicalVariables.sigma_thermal_mode /= properties.base_length;

	// ramp rate is intensity per second, so it gets multiplied by base_time:
	optomechanicalVariables.ramp_rate *= properties.base_time;

	// TODO: deal with max_intensity and Beta

	return optomechanicalVariables;
}
