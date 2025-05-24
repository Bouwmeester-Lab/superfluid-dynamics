import numpy as np
import derivatives
import scipy.linalg
def cotangent(z):
    return np.power(np.tan(z), -1)

class PhysicalProperties:
    def __init__(self, rho, kappa, U):
        self.rho = rho
        self.kappa = kappa
        self.U = U

class BoundaryIntegrationProblem:
    def __init__(self, N, physicalProperties : PhysicalProperties):
        self.N = N
        self.properties = physicalProperties

        self.dZ = derivatives.DifferentiatorLinearPeriodic(self.N, lambda j: 2*np.pi*j/N)
        # self.dy = derivatives.DifferentiatorFFT(self.N)
        self.dPhi = derivatives.DifferentiatorFFT(self.N)
        self.da = derivatives.DifferentiatorFFT(self.N)

        self.M = np.zeros((N,N))

    def setInitialProfile(self, X, Y, Phi):
        self.X = X
        self.Y = Y
        self.Phi = Phi

    def __calculateZ(self):
        self.Z = self.X + 1j*self.Y
        self.ZPrime = self.dZ(self.Z)
        self.ZDoublePrime = self.dZ(self.ZPrime)
    def __formM(self):
        try:
            rho = self.properties.rho
            
            for k in range(self.N):
                for j in range(self.N):
                    if k == j:
                        self.M[k, k] = (1+rho)*0.5 + (1-rho)*0.25/np.pi * np.imag(self.ZDoublePrime[k] / self.ZPrime[k])
                    else:
                        self.M[k, j] = (1-rho)/(4*np.pi)*np.imag(self.ZPrime[k] * cotangent((self.Z[k] - self.Z[j])/2.0))
            return True
        except Exception as e:
            print(f"Failed to form the matrix M with exception: {e}")
            return False
        
    def __one_over_curvature_radius(self):
        Xprime, Yprime = np.real(self.ZPrime), np.imag(self.ZPrime)
        Xdprime, Ydprime = np.real(self.ZDoublePrime), np.imag(self.ZDoublePrime)
        return (Xprime*Ydprime-Yprime*Xdprime)/np.power(Xprime**2 + Yprime**2, 3/2)
    
    def __calculatePhiPrime(self):
        rho = self.properties.rho
        kappa = self.properties.kappa


    def __calculatea(self):
        self.a = scipy.linalg.solve(self.M, self.PhiPrime)
        self.aprime = self.da(self.a)
    def __calculateVelocities(self, upper = False):

    