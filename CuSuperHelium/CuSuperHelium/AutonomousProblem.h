#pragma once
/// <summary>
/// Class representing a generic autonomous problem (i.e. independent of time) that can be solved with a generic ODE solver.
/// This class will automatically allocate the device memory for the right-hand side of the problem, which is expected to be filled by the run function.
/// The destructor will free the device memory which must be virtual to allow for proper cleanup in derived classes.
/// </summary>
/// <typeparam name="T">The type of the RHS of the problem, i.e. double or complex?</typeparam>
/// <typeparam name="N">The number of variables to account for</typeparam>
template<typename T, int N>
class AutonomousProblem
{
public:
	T* devTimeEvolutionRhs; ///< Device pointer to the right-hand side of the time evolution problem
	virtual T* getY0() = 0; ///< Function to get the initial state of the problem, to be implemented in derived classes
	AutonomousProblem()
	{
		cudaMalloc(&devTimeEvolutionRhs, N * sizeof(T)); // Allocate device memory for the right-hand side of the time evolution problem
	}
	virtual ~AutonomousProblem()
	{
		cudaFree(devTimeEvolutionRhs); // Free the device memory for the right-hand side of the time evolution problem
	}

	/// <summary>
	/// Function to run the time evolution problem, at the end the device pointer devTimeEvolutionRhs should be filled with the right-hand side of the problem
	/// </summary>
	/// <param name="initialState"></param>
	virtual void run(T* initialState) = 0;
};