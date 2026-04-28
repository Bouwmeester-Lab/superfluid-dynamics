import integration.rhs as rhs
import integration.gauss_legendre as gl
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.special as sp
import FourierSeries as fs
from scipy.integrate import solve_ivp
import os



if __name__ != "__main__":
    exit(0)

simManager = rhs.SimulationManager(r"D:\repos\superfluid-dynamics\CuSuperHelium\x64\Release\CuSuperHelium.dll")

alpha_hammaker = 2.6e-24


N = 256
def rhs_func(t, y, L = 1e-6, depth = 15e-9):
        local_x = y[:N]
        local_ampl = y[N:2*N]
        local_pot = y[2*N:3*N]
        res, vx, vy, dphi = simManager.calculate_rhs_from_vectors(local_x, local_ampl, local_pot, L, 145, 0, depth)
        if res != 0:
            raise Exception("Error in calculation")
        return np.concatenate((vx, vy, dphi))

def mode(t, r, r0, R, omega, zeta, A, phase_space=0, phase_time=0):
        return A * np.cos(zeta * (r - r0) / R + phase_space) * np.cos(omega * t + phase_time)
def zeta(n):
        return sp.jn_zeros(1, n)
modes = 1
zetas = zeta(modes)
def c3(d):
        return np.sqrt(3 * alpha_hammaker / d**3)
def angular_freq(zeta, _c3, R):
    return zeta * _c3 / R

L = 1e-3
depth = 50e-9

def f(y, L):
    return rhs_func(0, y, L, depth)

def J(y, L):
    local_x = y[:N]
    local_ampl = y[N:2*N]
    local_pot = y[2*N:3*N]
    res, jacobian = simManager.calculate_jacobian(local_x, local_ampl, local_pot, L, 145, 0, depth)
    if jacobian is None or res != 0:
        raise Exception("Error in Jacobian calculation")
    return jacobian

r = np.array([2.0*np.pi/N*x for x in range(N)])
initial_amplitude = 0.001*depth
L0 = L / (2.0 * np.pi)
g = 3*2.6e-24 / depth**4
_t0 = np.sqrt(L0 / g)

speed = c3(depth)
omegas = angular_freq(zetas, speed, 2*np.pi)
phase_spaces = np.zeros_like(zetas)  # np.random.uniform(0, 0.1*np.pi, modes)

def gaussian(x, x0 = 0.75*np.pi, sigma=0.4):
        return np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))
def soliton_sech2(x, x0 = 0.75*np.pi, width=0.4):
        return 1.0 / np.cosh((x - x0) / width) ** 2

def bimodal(x, x0, x1, sigma1 = 0.4, sigma2 = 0.4, a1 = 1.0, a2 = 1.0):
    return a1 * gaussian(x, x0, sigma1) + a2 * soliton_sech2(x, x1, sigma2)



amplitude = []
# potential = []
# for i in range(modes):
#     m = lambda x: mode(0, x, np.pi, 2.0*np.pi, omegas[i], zetas[i], initial_amplitude / L0, 0, phase_spaces[i])
#     a0_ex, an_ex, bn_ex = fs.compute_fourier_series(lambda x: bimodal(x, 0.5*np.pi, 1.25*np.pi, 0.4, 0.4, initial_amplitude / L0, initial_amplitude / L0), 10) # fs.compute_fourier_series(lambda x: soliton_sech2(x, np.pi, 0.5), 12)
#     m_ampl = [fs.fourier_series(x, a0_ex, an_ex, bn_ex) for x in r]
#     amplitude.append(m_ampl)
#     # amplitude.append(np.cos((i + 1) * r) * initial_amplitude / L0)
#     # potential.append(np.sin((i + 1) * r) * initial_amplitude / L0)

# sum = np.sum(amplitude, axis=0)
# ampl = sum / np.max(np.abs(sum)) * (initial_amplitude / L0)

# pot = np.sum(potential, axis=0)
# pot = pot / np.max(np.abs(pot)) * (initial_amplitude / L0)
def setInitialY0(r, L, initial_amplitude):
    L0 = L / (2.0 * np.pi)
    pot = np.zeros_like(r)
    ampl = 0.0 * np.sin(r) * initial_amplitude / L0
    pot = initial_amplitude/L0 * np.sin(r)
    return np.concatenate((r, ampl, pot))

def saveSimulationFile(filename, T, Y, Y0, L, depth, t0, t1, initial_amplitude, g):
    with h5py.File(filename, "w") as file:
        file.create_dataset("T", data=T)
        file.create_dataset("Y", data=Y)
        # file.create_dataset("x0", data=r)
        file.create_dataset("Y0", data=Y0)
        # file.create_dataset("pot0", data=pot)
        file.attrs["depth"] = depth
        file.attrs["L"] = L
        file.attrs["N"] = N
        file.attrs["t0"] = t0
        file.attrs["t1"] = t1
        file.attrs["initial_amplitude"] = initial_amplitude
        file.attrs["g"] = g
def loadSimulationFile(filename):
    ## check that file exists first
    if not os.path.exists(filename):
         return None
    with h5py.File(filename, "r") as file:
        T = np.array(file["T"])
        Y = np.array(file["Y"])
        Y0 = np.array(file["Y0"])
        depth = file.attrs["depth"]
        L = file.attrs["L"]
        N = file.attrs["N"]
        t0 = file.attrs["t0"]
        t1 = file.attrs["t1"]
        initial_amplitude = file.attrs["initial_amplitude"]
        g = file.attrs["g"]
        return T, Y, Y0, L, depth, N, t0, t1, initial_amplitude, g
    

print(f"Reference Time is {_t0:.2e} s")

t0 = 0.0
t1 = 5
h = 3
print(f"Integrating from t={t0:.2e} to t={t1:.3f} (nondim) with step h={h} (nondim))")
print(f"c3 is {speed:.2e} m/s")

Y0 = setInitialY0(r, L, initial_amplitude)

sim_props = rhs.CSimulationProperties(
        L = 1e-6,
        depth = 15e-9,
        rho = 150,
        kappa = 0,
        use_expansions = False,
        infinite_depth = False
    )
L0 = sim_props.L / (2.0 * np.pi)
_t0 = np.sqrt(L0 / g)
optomechanical_props = rhs.COptomechanicalProperties(
    detuning = 0.1,
    gamma = 0.01,
    G = 0.01,
    tau = 0.2,
    max_intensity = 0.1,
    initial_time = 0.0,
    location_x0_mode = np.pi * 1.0,
    sigma_optical_mode = 0.8, # 20e-9 / L0,
    beta = 1.0,
    damping_strength = 0.01
)
rk4_props = rhs.CRK4Options(
    timeStep = 0.01,
    t0 = 0.0,
    t1 = 5,
    returnTrajectory = True,
)

g = 3 * 2.6e-24 / np.power(sim_props.depth, 4)
L0 = sim_props.L / (2.0 * np.pi)


print(f"Reference Time is {_t0:.2e} s")
print(f"Reference Length is {L0:.2e} m")
# np.exp(-rk4_props.timeStep / 0.01)
print(f"Depth in non-dim is {sim_props.depth / L0:.7e}")

res, T_new, Y_new = simManager.integrate_optomechanical_problem(Y0, sim_props, optomechanical_props, rk4_props)

if res != 0:
    print("Error in integration")
    exit(1)

import matplotlib.pyplot as plt

for i, t in enumerate(T_new):
    plt.plot(Y_new[i, :N], Y_new[i, N:2*N], label=f"t={t:.2f}")
# plt.legend()
plt.show()

from scipy.interpolate import interp1d



def interpolate(x, y, x0):
    f = interp1d(x, y, fill_value="extrapolate")
    return f(x0)

x0 = 1.0*np.pi

values = np.zeros_like(T_new)

for i, t in enumerate(T_new):
    values[i] = L0 * interpolate(Y_new[i, :N], Y_new[i, N:2*N], x0)

plt.plot(T_new*_t0 *1e6, values)
plt.xlabel("Time (µs)")
plt.ylabel(f"Amplitude at x={x0:.2f}")
plt.show()