import sys
from pathlib import Path

from matplotlib.animation import FuncAnimation
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

def interpolate(x, y, x0):
    f = interp1d(x, y, fill_value="extrapolate")
    return f(x0)

### everything in SI units, all conversion to non-dimensional units is done in the C++ code, so we can just use real units here and not worry about it.
detuning = 0.5e6
gamma = 1e6
sigma = 50e-6
G = -20e6 * 1e9
tau = 1/18e3
x0 = 0.5e-3
L = 1e-3
depth = 5e-9
alpha_hamaker = 3.5e-24 # 6.3 https://arxiv.org/html/2504.13001v1#S5

simManager = rhs.SimulationManager(r"D:\repos\superfluid-dynamics\CuSuperHelium\x64\Release\CuSuperHelium.dll")

N = 1024//2
t0 = 0.0
t1 = 8000e-6 # in us #~ 1 au of time is about 1 us in for this system (L ~ 1 mm, depth 20 nm)
timeStep = 0.1e-6 # in us
beta = 1e6 # adimensional, this is just a ratio.

sim_props = rhs.CSimulationProperties(
        L = L,
        depth = depth,
        rho = 150,
        kappa = 0,
        use_expansions = False,
        infinite_depth = False
    )

L0 = sim_props.L / (2.0 * np.pi)
g = 3*alpha_hamaker / sim_props.depth**4
_t0 = np.sqrt(L0 / g)

optomechanical_props = rhs.COptomechanicalProperties(
    detuning = detuning, # in SI units Hz
    gamma = gamma, # in SI units Hz
    G = G, # in SI units Hz/m
    tau = tau, # in SI units s
    max_intensity = 1.0, # 100*P0 / base_power,
    initial_time = 0.0,
    location_x0_mode = 0.5*sim_props.L, # in SI units m # half of L
    sigma_optical_mode = sigma, # in SI units m
    beta =  beta,
    damping_strength = 0.0
)
rk4_props = rhs.CRK4Options(
    timeStep = timeStep,
    t0 = t0,
    t1 = t1,
    returnTrajectory = True,
)

detunings = np.array([-1, -0.1, 0.1, 1]) * detuning

r = np.array([2.0*np.pi/N*x for x in range(N)])

pot = np.zeros_like(r)
ampl = np.zeros_like(r)#np.ones_like(r) * 0.01 * sim_props.depth / L0 * soliton_sech2(r, x0=np.pi, sigma=0.8)
pot = np.zeros_like(r) # np.ones_like(r) *  0.01 * soliton_sech2(r, x0=np.pi, sigma=0.8)
delayed = np.zeros_like(r)

Y0 = np.concatenate((r, ampl, pot, delayed))

fig, axes = plt.subplots(2, 1, figsize=(15, 5))
ax_pd = axes[0]
ax_time = axes[1]

fig2, ax2 = plt.subplots(1, 1, figsize=(15, 5))

 

def plot_phase_diagram(values_y, T, base_time, ax_pd, ax_ts, label="Detuning", n_arrows=5, lw=0.8, arrow_step=5):
    ydot = np.gradient(values_y, T * base_time)
    line, = ax_pd.plot(values_y, ydot, label=f"{label}", lw=0.8)
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
    ax_ts.plot(T * base_time * 1e6, values_y, label=f"{label}", lw=lw)

for det in detunings:
    optomechanical_props.detuning = det
    print(f"Integrating for detuning={det:.3e} Hz")
    res, T_new, Y_new = simManager.integrate_augmented_optomechanical_problem(Y0, sim_props, optomechanical_props, rk4_props)
    if res != 0:
        print("Error in integration")
        exit(1)

    values_y = np.zeros_like(T_new)
    values_phi = np.zeros_like(T_new)
    values_d = np.zeros_like(T_new)

    for i, t in enumerate(T_new):
        values_y[i] = L0 * interpolate(Y_new[i, :N], Y_new[i, N:2*N], x0)
        values_phi[i] = interpolate(Y_new[i, :N], Y_new[i, 2*N:3*N], x0)
        values_d[i] = interpolate(Y_new[i, :N], Y_new[i, 3*N:4*N], x0)
    plot_phase_diagram(values_y, T_new, _t0, ax_pd, ax_time, label=f"Detuning = {det:.3e} Hz", n_arrows=10, lw=1.5)
    ax2.plot(Y_new[-1, :N], Y_new[-1, N:2*N], label=f"Last Interface - Detuning = {det:.3e} Hz", lw=1.5)


# plot_phase_diagram(values_y, T_new, _t0, ax_pd, ax_time, label=f"Depth = {depth:.3e}", n_arrows=10, lw=1.5)
axes[0].set_xlabel(r"$y(x_0 = \pi)$ m")
axes[0].set_ylabel(r"$\dot{y}(x_0 = \pi)$ m/s")
axes[0].set_title("Phase Diagram for y at x0 = pi")
axes[0].legend()

axes[1].set_xlabel("Time (µs)")
axes[1].set_ylabel(r"$y(x_0 = \pi)$ m")
axes[1].set_title("Time Series for y at x0 = pi")
axes[1].legend()

plt.show()