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


simManager = rhs.SimulationManager(r"D:\repos\superfluid-dynamics\CuSuperHelium\x64\Release\CuSuperHelium.dll")

alpha_hammaker = 2.6e-24

N = 64
t0 = 0.0
t1 = 100

sim_props = rhs.CSimulationProperties(
        L = 1e-6,
        depth = 15e-9,
        rho = 150,
        kappa = 0,
        use_expansions = False,
        infinite_depth = False
    )

optomechanical_props = rhs.COptomechanicalProperties(
    detuning = 0.2,
    gamma = 0.01,
    G = -0.02,
    tau = 1.0,
    max_intensity = 0.1,
    initial_time = 0.0,
    location_x0_mode = np.pi * 1.0,
    sigma_optical_mode = 0.2, # 20e-9 / L0,
    beta =  1.0,
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
# plt.show()


### we need to try to understand what the scales of the different d/dt components are in order to properly scale the residuals for the least squares solver.
dxdt = np.gradient(Y_new[:, :N], T_new, axis=0)
dydt = np.gradient(Y_new[:, N:2*N], T_new, axis=0)
dphidt = np.gradient(Y_new[:, 2*N:3*N], T_new, axis=0)
dDdt = np.gradient(Y_new[:, 3*N:4*N], T_new, axis=0)

## Remove physically irrelevant uniform phi drift
dphidt = dphidt - np.mean(dphidt, axis=1, keepdims=True)
dDdt = dDdt - np.mean(dDdt, axis=1, keepdims=True)

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
# plt.show()

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
Y0 = Y_new[-1] # start from the final state of the trajectory
x_fixed = Y_new[-1, :N].copy()

s_x = 5e-5
s_y = 2e-5
s_phi = 2.2531487047672272e-05
s_D = 5.412250175140798e-05

scales = np.array([s_x, s_y, s_phi, s_D]) # adjust these scales based on the expected magnitudes of the different components

def pack_state_from_reduced(U):
    Y = np.empty(4*N)

    Y[:N] = x_fixed
    Y[N:2*N] = U[:N]
    Y[2*N:3*N] = U[N:2*N]
    Y[3*N:4*N] = U[2*N:3*N]

    return Y

def scaled_residual(Y_flat, N, scales):
    dY = f(Y_flat).copy()


    dphidt = dY[2*N:3*N]
    dDdt = dY[3*N:4*N]
    # Remove physically irrelevant uniform phi drift
    dphidt = dphidt - np.mean(dphidt)
    dDdt = dDdt - np.mean(dDdt)

    sx, sy, sphi, sD = scales

    dY[0:N]       /= sx
    dY[N:2*N]     /= sy
    dY[2*N:3*N]   = dphidt / sphi
    dY[3*N:4*N]   = dDdt / sD

    return dY

def reduced_residual(U):
    Y = pack_state_from_reduced(U)
    return scaled_residual(Y, N, scales)

U0 = np.concatenate([
    Y0[N:2*N],
    Y0[2*N:3*N],
    Y0[3*N:4*N],
])


lower = np.full(3*N, -np.inf)
upper = np.full(3*N,  np.inf)

# # x must stay between 0 and 2pi
# lower[:N] = 0.0
# upper[:N] = 2*np.pi

# D must be nonnegative
lower[2*N:3*N] = 0.0

print("Running least squares solver to find fixed point of the system...")

sol = least_squares(
    reduced_residual,
    U0, # start from the final state of the trajectory
    method="trf",
    ftol=1e-10,
    xtol=1e-10,
    gtol=1e-10,
    max_nfev=100,
    bounds=(lower, upper)
)

Y_star = pack_state_from_reduced(sol.x)
r_scaled = scaled_residual(Y_star, N, scales)
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
## plot the residuals
axes[1, 1].plot(x_star, r_raw[:N], label="dx/dt")
axes[1, 1].plot(x_star, r_raw[N:2*N], label="dy/dt")
axes[1, 1].plot(x_star, r_raw[2*N:3*N], label="dphi/dt")
axes[1, 1].plot(x_star, r_raw[3*N:4*N], label="dD/dt")
axes[1, 1].set_title("Residuals at Fixed Point")
axes[1, 1].set_xlabel("x")
axes[1, 1].set_ylabel("Residual")
axes[1, 1].legend()

### calculate the Jacobian at the fixed point
J = rhs.jacobian_fd(f, Y_star, eps=1e-6)

idx_dyn = np.r_[N:2*N, 2*N:3*N, 3*N:4*N]

J_red = J[np.ix_(idx_dyn, idx_dyn)]

# B has shape (3N, 3N-1)
# Its columns span the gauge-free subspace.
g = np.zeros(3*N)
g[N:2*N] = 1.0
g /= np.linalg.norm(g)
B = null_space(g[None, :])

## project the jacobian onto the gauge-free subspace to remove the zero eigenvalue associated with the gauge invariance of phi:
# J_phys = B.T @ J_red @ B

eigenvalues, eigenvectors = np.linalg.eig(J_red)

# eigenvectors = B @ eigenvectors

# Normalize eigenvectors to unit norm, just to make visibility comparable
def scaled_norm(v, state_scales):
    return np.linalg.norm(v / state_scales)

eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis=0, keepdims=True)

print("Shape of eigenvalues:", eigenvalues.shape)
print("Shape of eigenvectors:", eigenvectors.shape)
## plot the eigenvalues in the complex plane
plt.figure(figsize=(8, 6))
plt.scatter(eigenvalues.real, eigenvalues.imag, color='blue', marker='o')
plt.title('Eigenvalues of the Jacobian at Fixed Point in the Complex Plane')
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.grid()


### we need to understande the visibility of each eigenmode in the point of interest, at x0 = pi

x0_index = np.argmin(np.abs(x_star - np.pi))
Pk_index = np.array([x0_index,]) #, 3*N + x0_index]) # projection onto the point of interest for each component
print(f"Index of point of interest (x0=pi): {x0_index}")
print("Visibility of each eigenmode at x0=pi is given by | P_k v_j | where P_k is the projection onto the point of interest k and v_j is the eigenvector of mode j.")

visibility_point_k = np.zeros(3*N)
point_scales = np.array([s_y, s_phi, s_D])
state_scales = np.empty(3*N)
state_scales[:N]       = s_y
state_scales[N:2*N]    = s_phi
state_scales[2*N:3*N]  = s_D
# state_scales[3*N:4*N]  = s_D

def scaled_vector(v, state_scales):
    vx = v[:N]
    vy = v[N:2*N]
    vphi = v[2*N:3*N]
    vD = v[3*N:4*N]

    # remove drift in phi and D:
    mean_vphi = np.mean(vphi)
    mean_vD = np.mean(vD)

    vphi = vphi - mean_vphi
    vD = vD - mean_vD

    vx_scaled = vx / state_scales[:N]
    vy_scaled = vy / state_scales[N:2*N]
    vphi_scaled = vphi / state_scales[2*N:3*N]
    vD_scaled = vD / state_scales[3*N:4*N]

    return np.concatenate((vx_scaled, vy_scaled, vphi_scaled, vD_scaled))
def red_scaled_vector(v, state_scales):
    vy = v[:N]
    vphi = v[N:2*N]
    vD = v[2*N:3*N]

    # remove drift in phi and D:
    mean_vphi = np.mean(vphi)
    mean_vD = np.mean(vD)

    vphi = vphi - mean_vphi
    vD = vD - mean_vD

    vy_scaled = vy / state_scales[:N]
    vphi_scaled = vphi / state_scales[N:2*N]
    vD_scaled = vD / state_scales[2*N:3*N]
    return np.concatenate((vy_scaled, vphi_scaled, vD_scaled))

for j in range(3*N):
    vj = eigenvectors[:, j]

    local_part = red_scaled_vector(vj, state_scales)[Pk_index]
    full_norm = np.linalg.norm(red_scaled_vector(vj, state_scales))

    visibility_point_k[j] = np.linalg.norm(local_part) / full_norm

# visibilities = np.linalg.norm(eigenvectors[Pk_index, :], axis=0)
print("Visibilities:", visibility_point_k)




gauge_overlap = np.abs(np.conj(g) @ eigenvectors)
plt.figure(figsize=(7, 4))
plt.plot(gauge_overlap, "o")
plt.plot(np.abs(eigenvalues) <= 1e-12, "o")

plt.xlabel("mode index")
plt.ylabel(r"$|g^\dagger v_j|$")
plt.title("Gauge overlap of each eigenvector")
plt.tight_layout()

Jg = J_red @ g
print("||J g|| =", np.linalg.norm(Jg))
print("||J g|| / ||J|| =", np.linalg.norm(Jg) / np.linalg.norm(J_red))
V_sub = eigenvectors[:, [95-N, 96-N]].astype(complex)

# Orthonormal basis for span{v95, v96}
Q, _ = np.linalg.qr(V_sub)

g_complex = g.astype(complex)

g_projection = Q @ (Q.conj().T @ g_complex)

overlap = np.linalg.norm(g_projection) / np.linalg.norm(g_complex)

print("overlap of g with span{v95, v96} =", overlap)

coeffs, *_ = np.linalg.lstsq(V_sub, g_complex, rcond=None)

g_fit = V_sub @ coeffs
g_fit /= np.linalg.norm(g_fit)

print("coeffs:", coeffs)
print("fit overlap:", abs(np.vdot(g_complex, g_fit)))
print("||J g_fit||:", np.linalg.norm(J_red @ g_fit))

j = 13
v13 = eigenvectors[:, j].astype(complex)
v13 /= np.linalg.norm(v13)

overlap_gauge = abs(g @ v13)

print("lambda_13:", eigenvalues[j])
print("gauge overlap mode 13:", overlap_gauge)
print("norm of v13:", np.linalg.norm(v13))
print("||J v13||:", np.linalg.norm(J_red @ v13))

# ## plot the components of v13 to see if it's linked mostly to x, y, phi, or D
# plt.figure(figsize=(12, 8))
# # plt.subplot(2, 2, 1)
# plt.plot(np.abs(v13[:N]), label="x")
# plt.plot(np.abs(v13[N:2*N]), label="y")
# plt.plot(np.abs(v13[2*N:3*N]), label="phi")
# plt.plot(np.abs(v13[3*N:4*N]), label="D")
# plt.legend()

# plt.show()
# plt.show()


## eigenvectors associated with phi's gauge invariance:
vectors_gauge = np.dot(eigenvectors.T, g)

# Make sure these are numpy arrays
eigenvalues = np.asarray(eigenvalues)
visibility = np.asarray(visibility_point_k)

re = eigenvalues.real
im = eigenvalues.imag

# Fixed plot limits so the axes do not jump around
re_pad = 0.05 * (re.max() - re.min() + 1e-30)
im_pad = 0.05 * (im.max() - im.min() + 1e-30)

xlim = (re.min() - re_pad, re.max() + re_pad)
ylim = (im.min() - im_pad, im.max() + im_pad)

# Visibility range
vmin = visibility.min()
vmax = visibility.max()

# Initial threshold
threshold0 = vmin
min_index = 0


mask = visibility >= threshold0
mask_index = np.arange(len(eigenvalues)) >= min_index
mask = mask & mask_index
# Figure and axes
fig, ax = plt.subplots(figsize=(7, 5))
plt.subplots_adjust(bottom=0.22)

# Initial scatter
sc = ax.scatter(
    re[mask],
    im[mask],
    c=visibility[mask],
    vmin=vmin,
    vmax=vmax
)

# Create annotations once
annotations = []

for i, lam in enumerate(eigenvalues):
    ann = ax.annotate(
        str(i),                     # label text
        (lam.real, lam.imag),        # point location
        textcoords="offset points",
        xytext=(4, 4),
        ha="left",
        fontsize=8,
        visible=mask[i] and i >= min_index,            # initial visibility
    )
    annotations.append(ann)

cbar = fig.colorbar(sc, ax=ax)
cbar.set_label(r"visibility on point $k$")

ax.set_xlabel(r"$\mathrm{Re}(\lambda)$")
ax.set_ylabel(r"$\mathrm{Im}(\lambda)$")
ax.axvline(0.0, linestyle="--")
ax.set_xlim(*xlim)
ax.set_ylim(*ylim)

title = ax.set_title(
    f"Visibility ≥ {threshold0:.3e} ({np.sum(mask)} / {len(eigenvalues)} modes)"
)

# Slider axis: [left, bottom, width, height]
ax_slider = fig.add_axes([0.18, 0.08, 0.65, 0.04])
ax_slider_2 = fig.add_axes([0.18, 0.02, 0.65, 0.04])

slider = Slider(
    ax=ax_slider,
    label="min visibility",
    valmin=vmin,
    valmax=vmax,
    valinit=threshold0,
    valfmt="%.3e"
)

slider_min_index = Slider(
    ax=ax_slider_2,
    label="min mode index",
    valmin=0,
    valmax=len(eigenvalues),
    valinit=0,
    valfmt="%.0f"
)

def update(threshold):
    mask = visibility >= threshold

    # Update point positions
    offsets = np.column_stack((re[mask], im[mask]))
    sc.set_offsets(offsets)

    # Update annotation visibility
    for ann, visible in zip(annotations, mask):
        ann.set_visible(visible)

    # Update colors
    sc.set_array(visibility[mask])

    title.set_text(
        f"Visibility ≥ {threshold:.3e} ({np.sum(mask)} / {len(eigenvalues)} modes)"
    )

    fig.canvas.draw_idle()

def update_min_index(min_index):
    mask_index = np.arange(len(eigenvalues)) >= min_index
    mask = visibility >= slider.val
    mask = mask & mask_index

    # Update point positions
    offsets = np.column_stack((re[mask], im[mask]))
    sc.set_offsets(offsets)

    # Update annotation visibility
    for ann, visible in zip(annotations, mask):
        ann.set_visible(visible)

    # Update colors
    sc.set_array(visibility[mask])

    title.set_text(
        f"Visibility ≥ {slider.val:.3e} ({np.sum(mask)} / {len(eigenvalues)} modes)"
    )

    fig.canvas.draw_idle()

slider.on_changed(update)
slider_min_index.on_changed(update_min_index)

# plt.show()


### study the result of a perturbation in the system using the reduced Jacobian to calculate the motion over time in the linearized system

from scipy.sparse.linalg import expm_multiply

def point_indices_reduced(N, k):
    """
    Reduced state ordering:
    U = [y0...yN-1, phi0...phiN-1, D0...DN-1]
    """
    return np.array([k, N + k, 2*N + k])


def linear_response(J_red, delta0, t_eval):
    """
    Computes deltaU(t) = exp(J_red t) delta0.

    t_eval must be uniformly spaced for this expm_multiply call.
    Returns array with shape (len(t_eval), len(delta0)).
    """
    return expm_multiply(
        J_red,
        delta0,
        start=t_eval[0],
        stop=t_eval[-1],
        num=len(t_eval),
        endpoint=True,
    )

# delta0 = np.zeros(3*N-1)
k = 16*2
# perturb y_k
def periodic_distance(x, x0, period=2*np.pi):
    """
    Signed shortest distance on a periodic domain.
    """
    return (x - x0 + period/2) % period - period/2


def gaussian_bump(x_fixed, k, sigma):
    dx = periodic_distance(x_fixed, x_fixed[k])
    bump = np.exp(-0.5 * (dx / sigma)**2)
    bump /= np.linalg.norm(bump)
    return bump


def reduced_perturbation_y_bump(N, k, x_fixed, sigma, amplitude=1.0):
    """
    Reduced ordering:
    U = [y, phi, D]
    """
    delta = np.zeros(3*N)

    bump = gaussian_bump(x_fixed, k, sigma)
    delta[N:2*N] = amplitude * bump

    return delta

# delta0 = reduced_perturbation_y_bump(
#     N=N,
#     k=k,
#     x_fixed=x_fixed,
#     sigma=0.3,
#     amplitude=1.0,
# )

# delta0 = np.zeros(3*N-1)
j = 129
v159 = eigenvectors[:, j]
v159 /= np.linalg.norm(v159)

delta0 = v159


plt.figure()
plt.plot(x_fixed, delta0[:N])
plt.xlabel("x")
plt.ylabel("initial perturbation")
plt.title("Initial Perturbation in y")


# delta0[k-N//2:k+N//2] = 0.2

# choose a time window
t_eval = np.linspace(0, 10, 1000)

delta_t = linear_response(J_red, delta0, t_eval)

idx_k_red = point_indices_reduced(N, k)
z_k_t = delta_t[:, idx_k_red]

plt.figure(figsize=(8, 5))

plt.plot(t_eval, z_k_t[:, 0], label=r"$\delta y_k$")
# plt.plot(t_eval, z_k_t[:, 1], label=r"$\delta \phi_k$")
plt.plot(t_eval, z_k_t[:, 2], label=r"$\delta D_k$")

plt.xlabel("time")
plt.ylabel("linear response")
plt.legend()
plt.tight_layout()
plt.show()
