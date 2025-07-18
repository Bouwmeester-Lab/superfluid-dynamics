import numpy as np
from scipy.integrate import quad

# Fix the syntax error: missing multiplication symbol in sin argument
def compute_fourier_series(f, N, a=0, b=2 * np.pi):
    """
    Compute the Fourier series coefficients for a 2π-periodic function f(x)
    over the interval [a, b] using N terms.

    Returns:
        a0: constant term
        an: list of cosine coefficients
        bn: list of sine coefficients
    """
    L = (b - a) / 2

    a0 = (1 / (2 * L)) * quad(f, a, b)[0]
    an = []
    bn = []

    for n in range(1, N + 1):
        an_n = (1 / L) * quad(lambda x: f(x) * np.cos(n * np.pi * (x - a) / L), a, b)[0]
        bn_n = (1 / L) * quad(lambda x: f(x) * np.sin(n * np.pi * (x - a) / L), a, b)[0]
        an.append(an_n)
        bn.append(bn_n)

    return a0, an, bn

if __name__ == "__main__":
    # Example usage: compute Fourier coefficients of a Gaussian centered at π
    def gaussian(x, sigma=0.4):
        return np.exp(-((x - 0.75 * np.pi) ** 2) / (2 * sigma ** 2))

    # Compute coefficients
    N = 20
    a0_ex, an_ex, bn_ex = compute_fourier_series(lambda x: gaussian(x, sigma=0.4), N)

    print(f"a0 = {a0_ex:.16e}")
    for a in an_ex:
        print(f"{a:.16e}, ", end = "")
    print("\n")
    
    for b in bn_ex:
        print(f"{b:.16e}, ", end = "")