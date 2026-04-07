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
N = 32

depth = 20e-9 # 20 nanometers
alpha_hammaker = 2.6e-24
r_amplitude = 2e-9 # 2 nm



def gaussian(x, x0, sigma):
    return np.exp(-((x - x0)**2)/(2*sigma**2))
def waveform(x, t, x0, k0, omega0, amplitude, sigma):
    return amplitude * gaussian(x, x0, sigma) * np.cos(k0*x - omega0*t)
def setInitialY0(r, y0, pot):
    return np.concatenate((r, y0, pot))

def saveSimulationFile(filename, T, Y, Y0, L, depth, t0, t1, initial_amplitude, g, k0, sigma):
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
        file.attrs["k0"] = k0
        if sigma is not None:
            file.attrs["sigma"] = sigma
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
    
def c3(d):
    return np.sqrt(3 * alpha_hammaker / d**3)
# Ls = np.linspace(20e-6, 1e-4, 1)
# L = 500e-6
min_L = 0.8e-3
max_L = 1e-3

wavenumbers = np.linspace(2.0*np.pi / max_L, 2.0*np.pi / min_L, 10)
Ls = 2.0 * np.pi / wavenumbers


wl0s = 50e-6 #np.linspace(0.05, 0.4, 10) * L 
# sigmas =   np.linspace(0.05, 0.9, 10) * L # [2.5e-6, ]
amplitudes = np.linspace(0.001, 0.05, 5) # np.linspace(0.001, 0.1, 5) * depth 

for j, L in enumerate(Ls):
    print(f"\n\nStarting simulations for L={L*1e6:.2f} um. {j+1}/{len(Ls)} wavelengths done")
    for i, r_amplitude in enumerate(amplitudes):
        # sigma = sigmas[0]
        wl0 = L
        


        # print(f"Simulating for wl0={wl0*1e6:.2f} um, sigma={sigma*1e6:.2f} um")
        print(f"Simulating for amplitude={r_amplitude*1e9:.2f} nm, wl0={wl0*1e6:.2f} um. {i+1}/{len(amplitudes)}  amplitudes done")
        L0 = wl0 / (2.0 * np.pi)

        
        print(f"  depth in L0 units: {depth:.8f}")

        amplitude = r_amplitude * depth / L0 # / L0
        g = 3*alpha_hammaker / depth**4
        # sigma = 0.09 * L
        r = np.array([2.0*np.pi/N*x for x in range(N)])

        # wl0 = 0.08 * L
        k0 = (2.0*np.pi)/(wl0)
        _t0 = np.sqrt(L0 / g)

        # freq_carrier = mean_phase_velocity * k0 # at 0.8 um wavelength
        speed = c3(depth)
        periods = 40
        t1 =  wl0 / speed * periods / _t0
        h = 1 #0.001 * wl0 / speed / _t0

        # y = waveform(r, 0, np.pi, k0*L0, 0, amplitude, sigma/L0)
        y = np.cos(r) * amplitude
        # envelope = amplitude * gaussian(r, np.pi, sigma/L0)

        # fig2 = plt.figure(figsize=(9,6), dpi=100)
        # ax2 = fig2.add_subplot(1,1,1)
        # ax2.plot(r*L0*1e6, y*L0*1e9)
        # # ax2.plot(r*L0*1e6, envelope*L0*1e9, 'r--', label="Envelope")
        # ax2.set_xlabel("Position (um)")
        # ax2.set_ylabel("Displacement (nm)")
        # ax2.set_title(f"Wavepacket at t={0*1e9:.2f} ns, Carrier Frequency = {speed * k0/ (2*np.pi):.2f} Hz")

        # plt.show()

        y0 = setInitialY0(r, y, amplitude * np.sin(r))

        # plt.plot(r, y)
        # plt.plot(r, amplitude*np.sin(r), 'r--')
        # plt.axhline(-depth/L0, color='k')
        # plt.show()

        simProperties = rhs.CSimulationProperties(
                L = wl0,
                depth = depth,
                rho = 150,
                kappa = 0,
                use_expansions = True,
                expansion_order = 3,
                infinite_depth = False
            )
        gaussLegendreOptions = rhs.CGaussLegendreProperties(
            t0 = 0,
            t1 = t1,
            stepSize = h,
            newtonTolerance = 1e-8,
            maxNewtonIterations = 30,
            allowSimplifiedFallback = False,
            returnTrajectory = True,
            armijo_c = 1e-4,
            backtrack = 0.05,
            minAlpha = 1e-6,
            maxStepsHalves = 40
        )
        base_path = f"data/linear/exp3" # wl_{depth*1e9:.0f}
        t0 = 0.0
        type_sim = "finite" if simProperties.infinite_depth == False else "infinite"
        filename = f"{base_path}/simulation_{type_sim}_exp_{simProperties.expansion_order}_k0_{k0:.2e}m_amp_{amplitude*L0:.2e}_wl0_{wl0:.2e}.h5"
        if os.path.exists(filename):
            T, Y, Y0, L_loaded, depth_loaded, N_loaded, t0_loaded, t1_loaded, initial_amplitude_loaded, g_loaded = loadSimulationFile(filename)
            if(t1  <= t1_loaded):
                print(f"  Simulation already exists with sufficient time {t1_loaded:.2e} s, skipping.")
                continue
                # exit(0)
            else:
                print(f"  Simulation exists but insufficient time {t1_loaded:.2e} s, extending to {t1:.2e} s.")
                gaussLegendreOptions.t0 = t1_loaded
                res, T_new, Y_new = simManager.integrate(Y[-1, :], simProperties, gaussLegendreOptions)
                if res != 0:
                    print(f"  Error during integration extension.")
                    continue
                    # exit(1)
                # T_new, Y_new = gl.integrate_gl2(lambda y: f(y, wl),  lambda y: J(y, wl), Y[-1, :], t1_loaded, t1 / _t0, h, show_progress = True, max_step_halves=20)
                Tcon = np.concatenate((T, T_new))
                Ycon = np.concatenate((Y, Y_new))
                saveSimulationFile(filename, Tcon, Ycon, y0, wl0, depth, t0, t1, amplitude, g, k0, None)
        else:
            res, T, Y = simManager.integrate(y0, simProperties, gaussLegendreOptions)
            if res != 0:
                print(f"Error during integration.")
                continue
                # exit(1)
            # T, Y = gl.integrate_gl2(lambda y: f(y, wl),  lambda y: J(y, wl), y0, t0, t1 / _t0, h, show_progress = True, max_step_halves=20)
            saveSimulationFile(filename, T, Y, y0, wl0, depth, t0, t1, amplitude, g, k0, None)