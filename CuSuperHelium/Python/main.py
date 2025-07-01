from WaterIntegralCalculator import WaterIntegralCalculator
from scipy.integrate import solve_ivp
import StokeWaves
import numpy as np
import matplotlib.pyplot as plt

def main(N, h, omega):
    j = 2.0 * np.pi * np.arange(N) / N
    X0 = StokeWaves.X(j, 0, h, omega)
    Y0 = StokeWaves.Y(j, 0, h, omega)
    Phi0 = StokeWaves.Phi(j, 0, h, omega)

    initialState = np.hstack((X0, Y0, Phi0))

    calculator = WaterIntegralCalculator()

    calculator.setInitialState(X0, Y0, Phi0)
    calculator.calculateEvolution()

    plt.figure()
    plt.plot(X0, Y0)
    plt.plot(X0, calculator.Xp, label = "xp")
    plt.plot(X0, calculator.Xpp, label = "xpp")
    plt.legend()
    plt.figure()
    plt.plot(X0, calculator.Yp, label = "yp")
    plt.plot(X0, calculator.Ypp, label ="Ypp")
    plt.legend()
    plt.figure()
    plt.plot(X0, Phi0, label = "Phi")
    plt.plot(X0, calculator.Phip, label = "PhiPrime")
    plt.legend()

    plt.figure()

    plt.title("RHS of starting values in Python")
    plt.plot(X0, calculator.vx, label = "vx")
    plt.plot(X0, calculator.vy, label = "vy")
    plt.plot(X0, calculator.dPhi, label = "dPhi")
    plt.legend()
    plt.show()
    sol = solve_ivp(calculator.runTimeStep, (0, 1.5), initialState, max_step = 0.5e-3)
    if sol.success:
        plt.figure()
        plt.plot(X0, Y0, label = "initial")
        plt.plot(X0, Phi0, label="Potential")
        Xfin = sol.y[:N]
        Yfin = sol.y[N:2*N]
        Phifin = sol.y[2*N:]
        plt.plot(Xfin[:,-1], Yfin[:,-1])

        plt.legend()
        plt.show()
    else:
        print(sol.message)

if __name__ == "__main__":
    main(128, 0.1, 1)