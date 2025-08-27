#pragma once

#include <vector>

class ValueLogger
{
public:
	ValueLogger(int stepInterval);
	~ValueLogger();
	
	void logValue(double value);
	virtual void logValue() = 0;
	bool shouldLog(const int currentStep) const 
	{ 
		return currentStep % stepInterval == 0; 
	}
	double getLoggedValue(int index) const 
	{ 
		return loggedValues.at(index); 
	}
	double getLastLoggedValue() const 
	{ 
		return loggedValues.empty() ? 0.0 : loggedValues.back(); 
	}
	std::vector<double>& getLoggedValues() 
	{ 
		return loggedValues; 
	}
	size_t getLoggedValuesCount() const 
	{ 
		return loggedValues.size(); 
	}
private:
	int stepInterval = 1000; // Log every 1000 steps
	std::vector<double> loggedValues; // Store logged values for later use
};

ValueLogger::ValueLogger(int stepInterval) : stepInterval(stepInterval)
{
}

ValueLogger::~ValueLogger()
{
}

void ValueLogger::logValue(double value)
{
	loggedValues.push_back(value);
}

template <typename T>
class ValueCallableLogger : public ValueLogger
{
private:
	T& functor;
public:
	ValueCallableLogger(T& functor, int intervelStep) : ValueLogger::ValueLogger(intervelStep), functor(functor)
	{
	}
	virtual void logValue() override {
		ValueLogger::logValue(functor());
	}
};