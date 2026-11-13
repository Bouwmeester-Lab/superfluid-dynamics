#include "Export/ExportInternal.cuh"

#include <cmath>

PhysicalParameters toPhysicalParameters(const SimProperties& simProperties)
{
	PhysicalParameters p;
	p.L = simProperties.L;
	p.depth = simProperties.depth;
	p.rho = simProperties.rho;
	p.kappa = simProperties.kappa;
	p.hamakerConstant = simProperties.hamaker_constant;
	p.filmDensity = simProperties.film_density;
	return p;
}

DimensionlessParameters toDimensionlessParameters(const SimProperties& simProperties)
{
	DimensionlessParameters properties = NonDimensionalization::nonDimensionalize(toPhysicalParameters(simProperties));
	properties.expansion_order = simProperties.expansion_order;
	properties.use_expansions = simProperties.use_expansions;
	properties.infinite_depth = simProperties.infinite_depth;
	return properties;
}

RK4SolverOptions toDimensionless(RK4SolverOptions options, const Scales& scales)
{
	options.t0 = NonDimensionalization::toDimensionlessTime(options.t0, scales);
	options.t1 = NonDimensionalization::toDimensionlessTime(options.t1, scales);
	options.timeStep = NonDimensionalization::toDimensionlessTime(options.timeStep, scales);
	return options;
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
	switch (c_optomechanicalVariables.drive_type) {
	case CDRIVE_TYPE_Constant:
		opto_variables.drive_type = DriveType::Constant;
		break;
	case CDRIVE_TYPE_Ramped:
		opto_variables.drive_type = DriveType::Ramped;
		break;
	case CDRIVE_TYPE_SineWave:
		opto_variables.drive_type = DriveType::Sine;
		break;
	default:
		throw std::invalid_argument("Invalid drive type in COptomechanicalVariables: " + std::to_string(static_cast<int>(c_optomechanicalVariables.drive_type)));
	}
	opto_variables.ramp_rate = c_optomechanicalVariables.ramp_rate; // (intensity / seconds)
	opto_variables.omega_drive = c_optomechanicalVariables.omega_drive; // (radians / second) 
}

