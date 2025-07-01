import numpy as np
import numpy.typing as npt
from Derivatives import *
def cot(z):
    return np.cos(z)/np.sin(z)

class WaterIntegralCalculator:
    def __init__(self):
        self.derivatives = FftDerivative()
        pass
    def setInitialState(self, X0, Y0, Phi0):
        self.X0 = X0
        self.Y0 = Y0
        self.Phi0 = Phi0
        pass
    def createCombinedState(self):
        return np.hstack((self.X0, self.Y0, self.Phi0))
    @staticmethod
    def __createMMatrix(X, Y, Xp, Yp, Xpp, Ypp) -> npt.ArrayLike:
        N = len(X)
        M = np.empty((N, N), np.float64)
        Z = X + 1j*Y
        Zp = Xp +1j*Yp
        Zpp = Xpp + 1j * Ypp
        for k in range(N):
            for j in range(N):
                if k == j:
                    M[k, j] = 0.5 + 0.25/np.pi * np.imag(Zpp[k]/Zp[k]) 
                else:
                    M[k, j] = 0.25/np.pi * np.imag(Zp[k] / np.tan(0.5 * (Z[k] - Z[j])))
        return M
    @staticmethod
    def __createVelocityMatrices(X, Y, Xp, Yp, Xpp, Ypp) -> tuple[npt.ArrayLike, npt.ArrayLike]:
        N = len(X)
        V1 = np.empty((N, N), np.complex128)
        V2 = np.empty((N,), np.complex128)
        Z = X + 1j*Y
        Zp = Xp +1j*Yp
        Zpp = Xpp + 1j * Ypp
        for k in range(N):
            for j in range(N):
                if k == j:
                    V1[k, j] = 0.5/Zp[k] - 0.25j/np.pi*Zpp[k]/np.power(Zp[k], 2.0)
                    V2[k] = 0.5j/(np.pi*Zp[k])
                else:
                    V1[k, j] = -0.25j/np.pi*cot(0.5*(Z[k] - Z[j]))
        return V1, V2
    @staticmethod
    def __createRhsPhi(Y, vx, vy):
        return 0.5*(np.power(vx, 2.0) + np.power(vy, 2.0)) - Y

    def calculateEvolution(self):
        self.Xp = XDerivative.firstDerivative(self.X0)
        self.Xpp = XDerivative.secondDerivative(self.X0)

        self.Yp = YDerivative.firstDerivative(self.Y0)
        self.Ypp = YDerivative.secondDerivative(self.Y0)

        self.Phip = PhiDerivative.firstDerivative(self.Phi0)

        self.M = self.__createMMatrix(self.X0, self.Y0, self.Xp, self.Yp, self.Xpp, self.Ypp)

        self.a = np.linalg.solve(self.M, self.Phip)

        self.ap = FftDerivative.exec(self.a, 2.0*np.pi/len(self.a))

        V1, V2 = self.__createVelocityMatrices(self.X0, self.Y0, self.Xp, self.Yp, self.Xpp, self.Ypp)

        velocities = V1 @ self.a + np.multiply(V2, self.ap)

        self.vx = np.real(velocities)
        self.vy = -np.imag(velocities)
        self.dPhi =self.__createRhsPhi(self.Y0, self.vx, self.vy)
    
    def runTimeStep(self, t, initialState) -> npt.ArrayLike:
        N = len(initialState) // 3
        X = initialState[:N]
        Y = initialState[N:2*N]
        Phi = initialState[2*N:]

        self.setInitialState(X, Y, Phi)
        self.calculateEvolution()

        return np.hstack((self.vx, self.vy, self.dPhi))
