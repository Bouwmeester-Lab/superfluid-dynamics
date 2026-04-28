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

simManager = rhs.SimulationManager(r"D:\repos\superfluid-dynamics\CuSuperHelium\x64\Release\CuSuperHelium.dll")

alpha_hammaker = 2.6e-24

N = 32
t0 = 0.0
t1 = 500

sim_props = rhs.CSimulationProperties(
        L = 1e-6,
        depth = 15e-9,
        rho = 150,
        kappa = 0,
        use_expansions = False,
        infinite_depth = False
    )

optomechanical_props = rhs.COptomechanicalProperties(
    detuning = 0.1,
    gamma = 0.01,
    G = -0.02,
    tau = 1.0,
    max_intensity = 0.1,
    initial_time = 0.0,
    location_x0_mode = np.pi * 1.0,
    sigma_optical_mode = 0.2, # 20e-9 / L0,
    beta = 1.0,
    damping_strength = 0.0
)

rk4_props = rhs.CRK4Options(
    timeStep = 0.05,
    t0 = t0,
    t1 = t1,
    returnTrajectory = True,
)

L0 = sim_props.L / (2.0 * np.pi)

r = np.array([2.0*np.pi/N*x for x in range(N)])
g = 3*2.6e-24 / sim_props.depth**4
_t0 = np.sqrt(L0 / g)

def setInitialY0(r):
    pot = np.zeros_like(r)
    ampl = np.zeros_like(r)
    pot = np.zeros_like(r)
    delayed = np.zeros_like(r)
    return np.concatenate((r, ampl, pot, delayed))

Y0 = setInitialY0(r)

print(f"Reference Time is {_t0:.2e} s")
print(f"Reference Length is {L0:.2e} m")
# np.exp(-rk4_props.timeStep / 0.01)
print(f"Depth in non-dim is {sim_props.depth / L0:.7e}")

def f(y):
    return simManager.calculate_augmented_rhs(y, sim_props, optomechanical_props)



res, T_new, Y_new = simManager.integrate_augmented_optomechanical_problem(Y0, sim_props, optomechanical_props, rk4_props)


from scipy.interpolate import interp1d



def interpolate(x, y, x0):
    f = interp1d(x, y, fill_value="extrapolate")
    return f(x0)

x0 = 1.0*np.pi

values = np.zeros_like(T_new)
plt.figure()
for i, t in enumerate(T_new):
    values[i] = L0 * interpolate(Y_new[i, :N], Y_new[i, N:2*N], x0)

plt.plot(T_new*_t0 *1e6, values)
plt.xlabel("Time (µs)")
plt.ylabel(f"Amplitude at x={x0:.2f}")
plt.show()


### we need to try to understand what the scales of the different d/dt components are in order to properly scale the residuals for the least squares solver.
dxdt = np.gradient(Y_new[:, :N], T_new, axis=0)
dydt = np.gradient(Y_new[:, N:2*N], T_new, axis=0)
dphidt = np.gradient(Y_new[:, 2*N:3*N], T_new, axis=0)
dDdt = np.gradient(Y_new[:, 3*N:4*N], T_new, axis=0)

## Remove physically irrelevant uniform phi drift
dphidt = dphidt - np.mean(dphidt, axis=1, keepdims=True)

### we have how the different components of the state vector evolve in time, we can use this information to scale the residuals for the least squares solver.
quantities = {
    r"$|\dot{x}|$": dxdt,
    r"$|\dot{y}|$": dydt,
    r"$|\dot{\phi}|$": dphidt,
    r"$|\dot{D}|$": dDdt,
}

fig, axes = plt.subplots(2, 2, figsize=(10, 7))
axes = axes.ravel()

for ax, (label, data) in zip(axes, quantities.items()):
    values = np.abs(data).ravel()
    values = values[np.isfinite(values)]

    ax.hist(values, bins=100)
    ax.set_xlabel(label)
    ax.set_ylabel("count")
    ax.set_title(label)

    print(label)
    print("  median:   ", np.median(values))
    print("  90%:      ", np.percentile(values, 90))
    print("  95%:      ", np.percentile(values, 95))
    print("  99%:      ", np.percentile(values, 99))
    print("  max:      ", np.max(values))

plt.tight_layout()
plt.show()

# J = rhs.jacobian_fd(f, Y0, eps=1e-6)

# # calculate the eigenvalues and eigenvectors of the Jacobian
# eigenvalues, eigenvectors = np.linalg.eig(J)

# ## plot the eigenvalues in the complex plane
# plt.figure(figsize=(8, 6))
# plt.scatter(eigenvalues.real, eigenvalues.imag, color='blue', marker='o')
# plt.title('Eigenvalues of the Jacobian in the Complex Plane')
# plt.xlabel('Real Part')
# plt.ylabel('Imaginary Part')
# plt.grid()
# plt.show()

### this is the https://chatgpt.com/s/t_69f122b2137081918128e9c866892cba piece of conversation relevant here.
## the idea is to study the behavior of point k around a known point.

## we begin by studying the full system.
#  i.e. we want to find the fixed points of the full system, where the membrane is stationary.

# this corresponds to finding the roots of the function f(y) = 0, where y is the state vector of the system.

from scipy.optimize import least_squares

def scaled_residual_full(Y_flat, N, scales):
    dY = f(Y_flat).copy()


    dphidt = dY[2*N:3*N]
    # Remove physically irrelevant uniform phi drift
    dphidt = dphidt - np.mean(dphidt)

    sx, sy, sphi, sD = scales

    dY[0:N]       /= sx
    dY[N:2*N]     /= sy
    dY[2*N:3*N]   = dphidt / sphi
    dY[3*N:4*N]   /= sD

    return dY

scales = np.array([5e-5, 2e-5, 2.2531487047672272e-05, 1e3]) # adjust these scales based on the expected magnitudes of the different components




sol = least_squares(
    lambda Y_flat: scaled_residual_full(Y_flat, N, scales),
    Y0,
    method="trf",
    ftol=1e-10,
    xtol=1e-10,
    gtol=1e-10,
    max_nfev=50,
)

Y_star = sol.x
r_scaled = scaled_residual_full(Y_star, N, scales)
r_raw = f(Y_star)

print("cost:", sol.cost)
print("scaled norm:", np.linalg.norm(r_scaled))
print("scaled max:", np.max(np.abs(r_scaled)))

print("raw norm:", np.linalg.norm(r_raw))
print("raw max |dx/dt|:", np.max(np.abs(r_raw[:N])))
print("raw max |dy/dt|:", np.max(np.abs(r_raw[N:2*N])))
print("raw max |dphi/dt|:", np.max(np.abs(r_raw[2*N:3*N])))
print("raw max |dD/dt|:", np.max(np.abs(r_raw[3*N:4*N])))

x_star = Y_star[:N]
y_star = Y_star[N:2*N]
phi_star = Y_star[2*N:3*N]
D_star = Y_star[3*N:4*N]

# plt.figure(figsize=(12, 8))
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
## plot the interface shape
axes[0, 0].plot(x_star, y_star)
axes[0, 0].set_title("Interface Shape at Fixed Point")
axes[0, 0].set_xlabel("x")
axes[0, 0].set_ylabel("y")
## plot the potential
axes[0, 1].plot(x_star, phi_star)
axes[0, 1].set_title("Optical Potential at Fixed Point")
axes[0, 1].set_xlabel("x")
axes[0, 1].set_ylabel("phi")
## plot the delayed optical mode
axes[1, 0].plot(x_star, D_star)
axes[1, 0].set_title("Delayed Optical Mode at Fixed Point")
axes[1, 0].set_xlabel("x")
axes[1, 0].set_ylabel("D")
plt.show()