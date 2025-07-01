import numpy as np
import scipy.fft as fft

class FftDerivative:
    
    @staticmethod
    def exec(Xperiodic, scale = 1.0):
        Xperiodic = Xperiodic.astype(np.complex128)
        N = len(Xperiodic)
        indx = fft.fftshift(np.arange(-N/2, N/2))
        coeffs = fft.fft(Xperiodic)
        
        c = coeffs[N//2]
        
        coeffs = 1j * indx * coeffs
        coeffs[N//2 + 1] = 0
        coeffs[N//2] = -np.pi * np.real(c)
        return fft.ifft(coeffs)*scale
    @staticmethod
    def secondDerivative(Xperiodic, scale = 1.0):
        Xperiodic = Xperiodic.astype(np.complex128)
        N = len(Xperiodic)
        indx = np.arange(-N/2, N/2)
        indx = np.power(1j*fft.fftshift(indx), 2)

        coeffs = fft.fft(Xperiodic) * indx
        return fft.ifft(coeffs) * scale

class XDerivative:
    @staticmethod
    def firstDerivative(X):
        N = len(X)
        j = np.arange(N)
        Xlin = np.pi * 2.0 * j /N
        Xper = X - Xlin
        return FftDerivative.exec(Xper, 2.0*np.pi/N) + 2.0 * np.pi / N
    @staticmethod
    def secondDerivative(X):
        N = len(X)
        j = np.arange(N)
        Xlin = np.pi * 2.0 * j / N
        Xper =X - Xlin
        return FftDerivative.secondDerivative(Xper, 4.0 * np.pi * np.pi / N**2)

class YDerivative:
    @staticmethod
    def firstDerivative(Y):
        return FftDerivative.exec(Y, 2.0 * np.pi / len(Y))
    @staticmethod
    def secondDerivative(Y):
        return FftDerivative.secondDerivative(Y, 4.0 * np.pi * np.pi / len(Y)**2)


class PhiDerivative:
    """
    Assumes that U = 0, so that Phi is purely periodic.
    """
    @staticmethod
    def firstDerivative(Phi):
        return FftDerivative.exec(Phi, 2.0 * np.pi / len(Phi))
    @staticmethod 
    def secondDerivative(Phi):
        return FftDerivative.secondDerivative(Phi, 2.0 * np.pi / len(Phi))
    