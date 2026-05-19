import sys
from pathlib import Path
from tokenize import group

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
from scipy.fft import fft, fftfreq, fftshift

def interpolate(x, y, x0):
    f = interp1d(x, y, fill_value="extrapolate")
    return f(x0)

### everything in SI units, all conversion to non-dimensional units is done in the C++ code, so we can just use real units here and not worry about it.
detuning = -1e4
gamma = 1e6

G = 20e6 * 1e9
tau = 1/18e3
x0 = 0.5e-3
L = 1e-3
depth = 15e-9
alpha_hamaker = 3.5e-24 # 6.3 https://arxiv.org/html/2504.13001v1#S5

simManager = rhs.SimulationManager(r"D:\repos\superfluid-dynamics\CuSuperHelium\x64\Release\CuSuperHelium.dll")

N = 2**10
t0 = 0.0
t1 = 100000e-6 # in us #~ 1 au of time is about 1 us in for this system (L ~ 1 mm, depth 20 nm)
timeStep = 0.6e-6 # in us
beta = 1e6 # adimensional, this is just a ratio.

sim_props = rhs.CSimulationProperties(
        L = L,
        depth = depth,
        rho = 90,
        kappa = 0,
        use_expansions = False,
        infinite_depth = False
    )

L0 = sim_props.L / (2.0 * np.pi)
g = 3*alpha_hamaker / sim_props.depth**4
_t0 = np.sqrt(L0 / g)
sigma = 30e-6 # in m, size of the beam waist
print(f"Sigma: {sigma:.3e} m")

optomechanical_props = rhs.COptomechanicalProperties(
    detuning = detuning, # in SI units Hz
    gamma = gamma, # in SI units Hz
    G = G, # in SI units Hz/m
    tau = tau, # in SI units s
    max_intensity = 50, # 100*P0 / base_power,
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

detunings = np.array([ 1]) * detuning

r = np.array([2.0*np.pi/N*x for x in range(N)])

pot = np.zeros_like(r)
ampl = np.zeros_like(r)#np.ones_like(r) * 0.01 * sim_props.depth / L0 * soliton_sech2(r, x0=np.pi, sigma=0.8)
pot = np.zeros_like(r) # np.ones_like(r) *  0.01 * soliton_sech2(r, x0=np.pi, sigma=0.8)
delayed = np.zeros_like(r)

Y0 = np.concatenate((r, ampl, pot, delayed))

fig, axes = plt.subplots(2, 1, figsize=(15, 5))
ax_pd = axes[0]
ax_time = axes[1]

fig2, axes2 = plt.subplots(2, 1, figsize=(15, 5))
ax_spatial_fft = axes2[0]
ax_last_interface = axes2[1]

 

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
def plot_spatial_fft(values_y, ax, label=""):
    k = fftfreq(len(values_y), d=L/N*1e3)
    Y = fft(values_y)

    ## shift the zero frequency component to the center of the spectrum
    k = fftshift(k)
    Y = fftshift(Y)

    ax.plot(k, np.abs(Y), label=label)

def check_bounds_for_time(Told, Tnew):
    """
    Checks if the new time array Tnew extends the old array correctly:
    - Tnew should start where Told left off, i.e. Tnew[0] should be approximately equal to Told[-1]
    """
    if Told[-1] > Tnew[0]:
        raise ValueError("Tnew does not start where Told left off.")
    return True
    
def save_results(filename, T, Y, sim_props, optomechanical_props, rk4_props):
    with h5py.File(filename, "a") as f:
        Yshape = Y.shape
        Yshape = list(Yshape)
        Yshape[0] = None
        Yshape = tuple(Yshape)
        
        print(f"Saving results to {filename} with T shape {T.shape} and Y shape {Yshape}")
        ## check if the dataset exists:
        if "T" in f and "Y" in f:
            print("Datasets already exist, appending data if needed.")
            T_existing = f["T"]
            Y_existing = f["Y"]
            oldN = T_existing.shape[0]
            newN = T.shape[0]
            if check_bounds_for_time(T_existing, T):
                ## append data to existing datasets
                T_existing.resize((T_existing.shape[0] + T.shape[0],))
                T_existing[oldN:oldN+newN] = T
                Y_existing.resize((Y_existing.shape[0] + Y.shape[0], Y_existing.shape[1]))
                Y_existing[oldN:oldN+newN, :] = Y
        else:
            f.create_dataset("T", data=T, maxshape=(None,))
            f.create_dataset("Y", data=Y, maxshape=Yshape)
        ## save the properties as attributes
        ## check if th group exists:
        if "sim_props" in f:
            print("sim_props group already exists, overwriting attributes.")
            group_sim_props = f["sim_props"]
        else:
            group_sim_props = f.create_group("sim_props")
        for name, _ in  sim_props._fields_:
            group_sim_props.attrs[name] = getattr(sim_props, name)
        if "optomechanical_props" in f:
            print("optomechanical_props group already exists, overwriting attributes.")
            group_optomechanical_props = f["optomechanical_props"]
        else:
            group_optomechanical_props = f.create_group("optomechanical_props")
        for name, _ in optomechanical_props._fields_:
            group_optomechanical_props.attrs[name] = getattr(optomechanical_props, name)
        if "rk4_props" in f:
            print("rk4_props group already exists, overwriting attributes.")
            group_rk4_props = f["rk4_props"]
            ## if the time bounds have changed, update them:
            if group_rk4_props.attrs["t1"] < rk4_props.t1:
                group_rk4_props.attrs["t1"] = rk4_props.t1
        else:
            group_rk4_props = f.create_group("rk4_props")
            for name, _ in rk4_props._fields_:
                group_rk4_props.attrs[name] = getattr(rk4_props, name)

def load_results(filename):
    if not os.path.exists(filename):
        return None
    with h5py.File(filename, "r") as f:
        T = np.array(f["T"])
        Y = np.array(f["Y"])
        sim_props = rhs.CSimulationProperties(**f["sim_props"].attrs)
        optomechanical_props = rhs.COptomechanicalProperties(**f["optomechanical_props"].attrs)
        rk4_props = rhs.CRK4Options(**f["rk4_props"].attrs)
        return T, Y, sim_props, optomechanical_props, rk4_props

for det in detunings:
    ### load file if it exists, otherwise run the simulation and save the results
    filename = f"results_detuning_{det:.3e}.h5"
    results = load_results(filename)
    if results is not None:
        T_new, Y_new, sim_props, optomechanical_props, rk4_props = results
        L0 = sim_props.L / (2.0 * np.pi)
        g = 3*alpha_hamaker / sim_props.depth**4
        _t0 = np.sqrt(L0 / g)
        print(f"Loaded results for detuning={det:.3e} Hz from file.")
        Y0 = Y_new[-1, :]

        t0_existing = rk4_props.t0
        t1_existing = rk4_props.t1

        if t1_existing < t1:
            ## extend the simulation from t1_existing to t1
            rk4_props.t0 = t1_existing
            rk4_props.t1 = t1
            print(f"Extending simulation from t={t1_existing:.3e} s to t={t1:.3e} s for detuning={det:.3e} Hz")
        else:
            print(f"Existing simulation already covers the desired time range for detuning={det:.3e} Hz, skipping integration.")

            continue
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
        values_y[i] = L0 * interpolate(Y_new[i, :N], Y_new[i, N:2*N], x0/L0)
        values_phi[i] = interpolate(Y_new[i, :N], Y_new[i, 2*N:3*N], x0/L0)
        values_d[i] = interpolate(Y_new[i, :N], Y_new[i, 3*N:4*N], x0/L0)
    plot_phase_diagram(values_y, T_new, _t0, ax_pd, ax_time, label=f"Detuning = {det:.3e} Hz", n_arrows=10, lw=1.5)
    plot_spatial_fft(Y_new[-1, N:2*N]*L0, ax_spatial_fft, label=f"Spatial FFT - Detuning = {det:.3e} Hz")
    ax_last_interface.plot(Y_new[-1, :N]*L0, Y_new[-1, N:2*N]*L0, label=f"Last Interface - Detuning = {det:.3e} Hz", lw=1.5)
    save_results(f"results_detuning_{det:.3e}.h5", T_new, Y_new, sim_props, optomechanical_props, rk4_props)

# plot_phase_diagram(values_y, T_new, _t0, ax_pd, ax_time, label=f"Depth = {depth:.3e}", n_arrows=10, lw=1.5)
axes[0].set_xlabel(r"$y(x_0 = \pi)$ m")
axes[0].set_ylabel(r"$\dot{y}(x_0 = \pi)$ m/s")
axes[0].set_title("Phase Diagram for y at x0 = pi")
axes[0].legend()

axes[1].set_xlabel("Time (µs)")
axes[1].set_ylabel(r"$y(x_0 = \pi)$ m")
axes[1].set_title("Time Series for y at x0 = pi")
axes[1].legend()

ax_last_interface.set_xlabel(r"$x$ (m)")
ax_last_interface.set_ylabel(r"$y(x)$ (m)")
ax_last_interface.set_title("Last Interface for y at different Detunings")
ax_last_interface.legend()


ax_spatial_fft.set_xlabel("1/$\lambda$ (1/mm)")
ax_spatial_fft.set_ylabel("FFT Amplitude")
ax_spatial_fft.set_title("Spatial FFT of y(x)")
plt.show()