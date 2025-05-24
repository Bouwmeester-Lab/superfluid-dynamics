import numpy as np
import scipy.signal
import scipy.fft as fft

class DifferentiatorFFT:
    def __init__(self, N):
        self.N = N
        self.indx = fft.fttshift(np.arange(-N/2, N/2))
    def __call__(self, X, *args, **kwds):
        coeffs = fft.fft(X)
        coeffs *= 1j * self.indx
        return fft.ifft(coeffs)
    
class DifferentiatorLinearPeriodic(DifferentiatorFFT):
    def __init__(self, N, linear_part_expression):
        super().__init__(N)
        self.linear = linear_part_expression(np.arange(N))
    def __call__(self, X, *args, **kwds):
        Xperiodic = X - self.linear 
        return self.linear + super().__call__(Xperiodic, *args, **kwds)
