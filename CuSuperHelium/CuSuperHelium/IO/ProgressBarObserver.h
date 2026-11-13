#pragma once
#ifndef PROGRESS_BAR_OBSERVER_H
#define PROGRESS_BAR_OBSERVER_H

#include <string>
#include "IO/StepObserver.h"
#include "indicators/progress_bar.hpp"

/// <summary>
/// Shows a console progress bar for an integration from t0 to t1. Opt-in: attach it with OdeSolver::addObserver.
/// </summary>
template <typename T>
class ProgressBarObserver final : public StepObserver<T>
{
public:
	ProgressBarObserver(double t0, double t1, const std::string& label = "Running integration") : t0(t0), t1(t1),
		bar{
			indicators::option::BarWidth{50},
			indicators::option::Start{"["},
			indicators::option::Fill{"="},
			indicators::option::Lead{">"},
			indicators::option::Remainder{" "},
			indicators::option::End{"]"},
			indicators::option::PostfixText{label},
			indicators::option::ForegroundColor{indicators::Color::green},
			indicators::option::ShowPercentage{true},
			indicators::option::FontStyles{std::vector<indicators::FontStyle>{indicators::FontStyle::bold}},
			indicators::option::ShowElapsedTime{true},
			indicators::option::ShowRemainingTime{true},
		}
	{
	}

	void onStart(double, const T*, size_t, cudaStream_t) override
	{
		bar.set_progress(0);
	}

	void onAcceptedStep(double t, const T*, size_t, cudaStream_t) override
	{
		if (t1 == t0)
		{
			return;
		}
		const double fraction = (t - t0) / (t1 - t0);
		const size_t percent = fraction <= 0.0 ? 0 : (fraction >= 1.0 ? 100 : static_cast<size_t>(fraction * 100.0));
		if (percent > bar.current())
		{
			bar.set_progress(percent);
		}
	}

private:
	double t0;
	double t1;
	indicators::ProgressBar bar;
};

#endif // !PROGRESS_BAR_OBSERVER_H
