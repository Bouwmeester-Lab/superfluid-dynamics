
import os
import integration.rhs as rhs
import integration.gauss_legendre as gl
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.special as sp
import FourierSeries as fs
from scipy.integrate import solve_ivp

simManager = rhs.SimulationManager(r"D:\repos\superfluid-dynamics\CuSuperHelium\x64\Release\CuSuperHelium.dll")
alpha_hammaker = 2.6e-24


N = 32
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
    ampl = np.cos(r) * initial_amplitude / L0
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
t1 = 25 * 500
h = 3
print(f"Integrating from t={t0:.2e} to t={t1:.3f} (nondim) with step h={h} (nondim))")
print(f"c3 is {speed:.2e} m/s")

# t0, t1, h = 0.0, 547, 0.5
# from tqdm import tqdm
max_wave_length = 1e-4
min_wave_length = 1e-6


waveNumbers = np.linspace(1e3, 200000, 20)
waveLengths = 1 / waveNumbers
periods = 250
h = 10

initial_amplitudes = np.linspace(1e-6 * depth, 1e-2 * depth, 20)
wl = waveLengths[5] # 2*np.pi / (1.89e5)

# for wl in waveLengths[5:]: # next one to start with 1.87e-05
for amplitude in initial_amplitudes[16:]:
    y0 = setInitialY0(r, wl, amplitude)
    t1 = wl / speed * periods
    L0 = wl / (2.0 * np.pi)
    _t0 = np.sqrt(L0 / g)
    h = 0.005 * wl / speed / _t0
    print(f"Simulating wave length {wl:.2e} m for time {t1:.2e} s, equivalent to {t1/_t0:.2e} nondim time, with a max step size of {h:.2f} nondim. Initial amplitude {amplitude:.2e} m")
    simProperties = rhs.CSimulationProperties(
        L = wl,
        depth = depth,
        rho = 145,
        kappa = 0,
    )
    gaussLegendreOptions = rhs.CGaussLegendreProperties(
        t0 = 0,
        t1 = t1 / _t0,
        stepSize = h,
        newtonTolerance = 1e-10,
        maxNewtonIterations = 12,
        allowSimplifiedFallback = True,
        returnTrajectory = True,
        armijo_c = 1e-4,
        backtrack = 0.5,
        minAlpha = 1e-6,
        maxStepsHalves = 20
    )

    if os.path.exists(f"data/fixed/simulation_ia_{amplitude:.2e}m.h5"):
        T, Y, Y0, L_loaded, depth_loaded, N_loaded, t0_loaded, t1_loaded, initial_amplitude_loaded, g_loaded = loadSimulationFile(f"data/fixed/simulation_ia_{amplitude:.2e}m.h5")
        if(t1 / _t0 <= t1_loaded):
            print(f"  Simulation already exists with sufficient time {t1_loaded:.2e} s, skipping.")
            continue
        else:
            print(f"  Simulation exists but insufficient time {t1_loaded:.2e} s, extending to {t1:.2e} s.")
            gaussLegendreOptions.t0 = t1_loaded
            res, T_new, Y_new = simManager.integrate(Y[-1, :], simProperties, gaussLegendreOptions)
            if res != 0:
                print(f"  Error during integration extension.")
                continue
            # T_new, Y_new = gl.integrate_gl2(lambda y: f(y, wl),  lambda y: J(y, wl), Y[-1, :], t1_loaded, t1 / _t0, h, show_progress = True, max_step_halves=20)
            Tcon = np.concatenate((T, T_new))
            Ycon = np.concatenate((Y, Y_new))
            saveSimulationFile(f"data/fixed/simulation_ia_{amplitude:.2e}m.h5", Tcon, Ycon, y0, wl, depth, t0, t1 / _t0, amplitude, g)
    else:
        res, T, Y = simManager.integrate(y0, simProperties, gaussLegendreOptions)
        if res != 0:
            print(f"  Error during integration.")
            continue
        # T, Y = gl.integrate_gl2(lambda y: f(y, wl),  lambda y: J(y, wl), y0, t0, t1 / _t0, h, show_progress = True, max_step_halves=20)
        saveSimulationFile(f"data/fixed/simulation_ia_{amplitude:.2e}m.h5", T, Y, y0, wl, depth, t0, t1 / _t0, amplitude, g)