#include "Derivatives.hpp"


double filterIndexTanh(int m, int N)
{
	return 0.5 * (1 - tanh(40 * (static_cast<double>(m) / N - 0.25)));
}

