import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

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

def fourier_series(x, a0, an, bn):
    """Evaluate Fourier series at x, given coefficients a0, an, bn"""
    result = a0
    for n in range(1, len(an) + 1):
        result += an[n - 1] * np.cos(n * x) + bn[n - 1] * np.sin(n * x)
    return result

if __name__ == "__main__":
    # Example usage: compute Fourier coefficients of a Gaussian centered at π
    def gaussian(x, x0 = 0.75*np.pi, sigma=0.4):
        return np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))
    def bimodal(x, x0, x1, sigma = 0.4):
        return gaussian(x, x0, sigma) + gaussian(x, x1, sigma)


    # Compute coefficients
    N = 200
    a0_ex, an_ex, bn_ex = compute_fourier_series(lambda x: 9.4247779608e-06 * gaussian(x, np.pi,  sigma=0.2), N)

    print(f"a0 = {a0_ex:.16e}")
    for a in an_ex:
        print(f"{a:.16e}, ", end = "")
    print("\n")
    
    for b in bn_ex:
        print(f"{b:.16e}, ", end = "")

    x_vals = np.linspace(0, 2 * np.pi, 1000)
    y_vals = [fourier_series(x, a0_ex, an_ex, bn_ex) for x in x_vals]

    plt.figure(figsize=(8, 4))
    plt.plot(x_vals, y_vals, label='Fourier Approximation')
    plt.title('Fourier Series Approximation')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)
    plt.legend()
    plt.show()