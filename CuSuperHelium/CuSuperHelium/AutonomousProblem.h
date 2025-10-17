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
	// virtual T* getY0() = 0; ///< Function to get the initial state of the problem, to be implemented in derived classes
	AutonomousProblem()
	{
	}
	virtual ~AutonomousProblem()
	{
	}

	/// <summary>
	/// Function to run the time evolution problem, at the end the device pointer devTimeEvolutionRhs should be filled with the right-hand side of the problem
	/// </summary>
	/// <param name="initialState"></param>
	/// <param name="rhs">The rhs vector where to place the result</param>
	virtual void run(T* initialState, T* rhs) = 0;
	virtual void setStream(cudaStream_t stream) = 0;
};

template<typename T, int N, size_t batchSize>
class BatchedAutonomousProblem
{
public:
	// virtual T* getY0() = 0; ///< Function to get the initial state of the problem, to be implemented in derived classes
	BatchedAutonomousProblem()
	{
	}
	virtual ~BatchedAutonomousProblem()
	{
	}

	/// <summary>
	/// Function to run the time evolution problem, at the end the device pointer devTimeEvolutionRhs should be filled with the right-hand side of the problem
	/// </summary>
	/// <param name="initialState">An array pointer to a N*batchSize sized array holding batchSize initial states for the autonomous problem</param>
	/// <param name="rhs">The rhs vector where to place the result for each initial state. It's sized as (NxbatchSize) in 1D.</param>
	virtual void runBatch(T* initialState, T* rhs) = 0;
	virtual void setStream(cudaStream_t stream) = 0;
};