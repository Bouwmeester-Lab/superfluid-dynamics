#pragma once
#ifndef RK4OPTIONS_H
#define RK4OPTIONS_H

struct RK4Options {
	double initial_timestep = 1e-2; ///< Time step for the Runge-Kutta method
	bool returnTrajectory = true;
};

#endif // !RK4OPTIONS_H