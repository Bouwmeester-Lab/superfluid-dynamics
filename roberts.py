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
        self.V = np.zeros((self.N,), dtype=np.complex128)

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
    
    def __calculatePhiPrimeRHS(self):
        rho = self.properties.rho
        kappa = self.properties.kappa
        
        if kappa != 0:
            rs = self.__one_over_curvature_radius()
        else:
            rs = 0
        # u1 = 
        return -1*(1+rho)*np.imag(self.Z)+0.5*()


    def __calculatea(self):
        self.PhiPrime = self.dPhi(self.Phi)
        self.a = scipy.linalg.solve(self.M, self.PhiPrime)
        self.aprime = self.da(self.a)
    def __calculateVelocities(self, upper = False):
        if upper:
            sign = -1
        else:
            sign = 1
        
        for k in range(self.N):
            for j in range(self.N):
                if k == j:
                    self.V[k] += sign*self.a[k]/(2*self.ZPrime[k]) - 1j/(4*np.pi) (self.ZDoublePrime[k]/(self.ZPrime[k]**2)*self.a[k]-2*self.aprime[k]/self.ZPrime[k])
                else:
                    self.V[k] += -1j/(4*np.pi) * self.a[j] * cotangent((self.Z[k] - self.Z[j])/2.0) # cotangent_substraction(Z[k]/2, Z[j]/2, k, j, "vu") # eq. 3.2 from Roberts 1983
        return True
    