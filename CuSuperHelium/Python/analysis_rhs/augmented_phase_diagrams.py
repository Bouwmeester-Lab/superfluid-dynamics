import sys
from pathlib import Path
python_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(python_dir))

import integration.rhs as rhs
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.special as sp
import FourierSeries as fs
from scipy.integrate import solve_ivp
import os
from matplotlib.widgets import Slider
from scipy.linalg import null_space
from scipy.interpolate import interp1d

simManager = rhs.SimulationManager(r"D:\repos\superfluid-dynamics\CuSuperHelium\x64\Release\CuSuperHelium.dll")

alpha_hammaker = 2.6e-24

N = 64
t0 = 0.0
t1 = 60000

sim_props = rhs.CSimulationProperties(
        L = 1e-3,
        depth = 15e-9,
        rho = 150,
        kappa = 0,
        use_expansions = False,
        infinite_depth = False
    )
L0 = sim_props.L / (2.0 * np.pi)
optomechanical_props = rhs.COptomechanicalProperties(
    detuning = 0.1,
    gamma = 0.01,
    G = -0.5,
    tau = 1.0,
    max_intensity = 0.01,
    initial_time = 0.0,
    location_x0_mode = np.pi * 1.0,
    sigma_optical_mode =  20e-6 / L0,
    beta =  0.5,
    damping_strength = 0.0
)

rk4_props = rhs.CRK4Options(
    timeStep = 0.8,
    t0 = t0,
    t1 = t1,
    returnTrajectory = True,
)



r = np.array([2.0*np.pi/N*x for x in range(N)])


def gaussian(x, x0 = 0.75*np.pi, sigma=0.8):
    return np.exp(-(x - x0)**2 / (2 * sigma**2))

def soliton_sech2(x, x0, sigma):
    return 1 / np.cosh((x - x0) / sigma)**2

def setInitialY0(r):
    pot = np.zeros_like(r)
    ampl = np.zeros_like(r)#np.ones_like(r) * 0.01 * sim_props.depth / L0 * soliton_sech2(r, x0=np.pi, sigma=0.8)
    pot = np.zeros_like(r) # np.ones_like(r) *  0.01 * soliton_sech2(r, x0=np.pi, sigma=0.8)
    delayed = np.zeros_like(r)
    return np.concatenate((r, ampl, pot, delayed))

Y0 = setInitialY0(r)
g = 3*2.6e-24 / sim_props.depth**4
_t0 = np.sqrt(L0 / g)

print(f"Reference Time is {_t0:.2e} s")
print(f"Reference Length is {L0:.2e} m")
# np.exp(-rk4_props.timeStep / 0.01)
print(f"Depth in non-dim is {sim_props.depth / L0:.7e}")

def f(y):
    return simManager.calculate_augmented_rhs(y, sim_props, optomechanical_props)


dampings = np.array([0.0, 0.1, 0.2, 2])*-1e-2
def interpolate(x, y, x0):
    f = interp1d(x, y, fill_value="extrapolate")
    return f(x0)

def plot_phase_diagram(value, change_func, ax_pd, ax_ts, label="Detuning", n_arrows=5, lw=0.8, arrow_step=5):
    change_func(value)
    res, T_new, Y_new = simManager.integrate_augmented_optomechanical_problem(Y0, sim_props, optomechanical_props, rk4_props)
    if res != 0:
        print("Error in integration")
        exit(1)
    x0 = 1.0*np.pi

    L0 = sim_props.L / (2.0 * np.pi)
    g = 3*2.6e-24 / sim_props.depth**4
    _t0 = np.sqrt(L0 / g)

    values_y = np.zeros_like(T_new)
    values_phi = np.zeros_like(T_new)
    values_d = np.zeros_like(T_new)
    # plt.figure()
    for i, t in enumerate(T_new):
        values_y[i] = L0 * interpolate(Y_new[i, :N], Y_new[i, N:2*N], x0)
        values_phi[i] = interpolate(Y_new[i, :N], Y_new[i, 2*N:3*N], x0)
        values_d[i] = interpolate(Y_new[i, :N], Y_new[i, 3*N:4*N], x0)

    ydot = np.gradient(values_y, T_new * _t0)
    line, = ax_pd.plot(values_y, ydot, label=f"{label}={value:.3e}", lw=0.8)
    color = line.get_color()
    idx = np.linspace(0, len(values_y) - arrow_step - 1, n_arrows, dtype=int)
    for i in idx:
        ax_pd.annotate(
            "",
            xy=(values_y[i + arrow_step], ydot[i + arrow_step]),
            xytext=(values_y[i], ydot[i]),
            arrowprops=dict(
                arrowstyle="->",
                color=color,
                lw=lw,
                shrinkA=0,
                shrinkB=0,
            ),
        )
    
    ax_ts.plot(T_new*_t0 *1e6, values_y, label=f"{label}={value:.3e}", lw=lw)

# xdot = np.gradient(values, T_new)
## make a subplot with the 3 phase diagrams for (y, ydot), (phi, phi_dot), (d, d_dot)
fig, axes = plt.subplots(2, 1, figsize=(15, 5))

change_detuning = lambda detuning: setattr(optomechanical_props, "detuning", detuning)
change_optical_sigma = lambda sigma: setattr(optomechanical_props, "sigma_optical_mode", sigma)
change_gamma = lambda gamma: setattr(optomechanical_props, "gamma", gamma)
change_damping = lambda damping: setattr(optomechanical_props, "damping_strength", damping)

change_depth = lambda depth: setattr(sim_props, "depth", depth)

depths = np.array([ 10, 20 ])*1e-9
for depth in depths:
    plot_phase_diagram(depth, change_depth, axes[0], axes[1], label="Depth")

axes[0].set_xlabel(r"$y(x_0 = \pi)$ m")
axes[0].set_ylabel(r"$\dot{y}(x_0 = \pi)$ m/s")
axes[0].set_title("Phase Diagram for y at x0 = pi")
axes[0].legend()

axes[1].set_xlabel("Time (µs)")
axes[1].set_ylabel(r"$y(x_0 = \pi)$ m")
axes[1].set_title("Time Series for y at x0 = pi")
axes[1].legend()
# axes[1].scatter(values_phi, phi_dot)
# axes[1].set_xlabel(r"$\phi(x_0 = \pi)$")
# axes[1].set_ylabel(r"$\dot{\phi}(x_0 = \pi)$")
# axes[1].set_title("Phase Diagram for phi at x0 = pi")
# axes[2].scatter(values_d, d_dot)
# axes[2].set_xlabel(r"$d(x_0 = \pi)$")
# axes[2].set_ylabel(r"$\dot{d}(x_0 = \pi)$")
# axes[2].set_title("Phase Diagram for d at x0 = pi")

plt.show()