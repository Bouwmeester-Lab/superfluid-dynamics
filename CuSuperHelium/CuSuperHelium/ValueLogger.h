#pragma once

#include <vector>

class ValueLogger
{
public:
	ValueLogger(int stepInterval, size_t size = 0);
	~ValueLogger();
	
	void logValue(double value);
	void setSize(size_t size)
	{
		this->size = size;
		loggedValues.reserve(size);
	}
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
	size_t size = 0;
};

ValueLogger::ValueLogger(int stepInterval, size_t size) : stepInterval(stepInterval), size(size)
{
	loggedValues.clear();
	loggedValues.reserve(size);
}

ValueLogger::~ValueLogger()
{
}

void ValueLogger::logValue(double value)
{
	if(std::isnan(value) || std::isinf(value))
		throw std::runtime_error("ValueLogger::logValue: Attempting to log NaN or Inf value.");
	loggedValues.push_back(value);
}

template <typename T>
class ValueCallableLogger : public ValueLogger
{
private:
	T& functor;
public:
	ValueCallableLogger(T& functor, int intervelStep, size_t size = 0) : ValueLogger::ValueLogger(intervelStep, size), functor(functor)
	{
	}
	virtual void logValue() override {
		ValueLogger::logValue(functor());
	}
};