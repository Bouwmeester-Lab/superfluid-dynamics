import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import scienceplots
import itertools

plt.style.use(['science', 'no-latex'])
plt.rcParams.update({'figure.dpi': '100'})
## variables
path =r"C:\Users\emore\OneDrive\Desktop\data.h5" # "D:\repos\superfluid-dynamics\CuSuperHelium\CuSuperHelium\temp\data.h5"
title = "Gaussian Sin Pulse"
H0 = 15e-9 # 15 nm
g = 3 * 2.6e-24 / np.power(H0, 4)
L0 = 500e-6/(2.0*np.pi)
_t0 = np.sqrt(L0 / g)
x = np.array([0.1, 0.5, 1.0, 1.5, 1.9])*np.pi # from 0 to 2*pi
y_lim_min = -0.35
y_lim_max = 0.35
frame = 0
## code to read the h5 file:
values = {}
dt = 0
with h5py.File(path, 'r') as f:
    # List all groups
    # print("Keys: %s" % f.keys())
    ## get all the keys which contain "i = "
    keys = [key for key in f.keys() if "i = " in key]
    # print("Filtered Keys: %s" % keys)
    # get all the iteration numbers after "i = "
    iteration_numbers = [int(key.split("i = ")[1]) for key in keys]
    # iteration_numbers.sort()
    # print("Iteration Numbers: %s" % iteration_numbers)
    for i in range(len(keys)):
        # print(f"Key: {key}")
        # Get the dataset for each key
        key = keys[i]
        iteration = iteration_numbers[i]
        dataset = f[key]

        values[iteration] = dataset[:]  # Append the data to the values list
        # print(f"Dataset shape: {dataset.shape}")
        # print(f"Dataset data: {dataset[:]}")
    dt = f.attrs["dt"]  # Print attributes of the file
    print(f.attrs.keys())
    print(f"depth: {f.attrs['depth'] * L0 * 1e9:.2f} nm")
    print(f"initial amplitude: {f.attrs['initial_amplitude'] * L0 * 1e9:.2f} nm")
    # Get the first object name
    a_group_key = list(f.keys())[0]
    # Get the object type
    a_dataset = f[a_group_key]
    # print(type(a_dataset))  # <class 'h5py._hl.dataset.Dataset'>
    # Get the data
    data = a_dataset[:]
    # print(data)
keys = sorted(values.keys())
# sort the values by key
values_sorted = [values[k] for k in keys]
times = np.array([k * dt * _t0 for k in keys] )

# values = [val for key, val in values]
# print(values_sorted)
point = 10

data = []

for i in range(len(values_sorted)):
    # x.append(values[i][:])  
    data.append(values_sorted[i][:])  # Assuming the amplitude is in the first column

data = np.array(data)

def extrapolate_amplitude(x_requested, x_data, y_data):
    pchip = interp1d(x_data, y_data)
    return pchip(x_requested)

amplitude = []
for i in range(data.shape[0]):
    y_extrapolated = extrapolate_amplitude(x, data[i, :1024//2, 0], data[i, :1024//2, 1])
    amplitude.append(y_extrapolated)

amplitude = L0 * np.array(amplitude)
# plt.figure(figsize=(10, 6))
fig, ax = plt.subplots(len(x)+1, 1, constrained_layout=True, figsize=(13.33, 7.5))
fig.suptitle(f"Wave Amplitude Over Time at Different Positions - {title}", fontsize=16)

# select the points where time is between 0 and -10.5 ms
mask = times >= -10.5e-3
times = times[mask]
amplitude = amplitude[mask, :]

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
color_iter_plots = itertools.cycle(colors)
# color_iter_vlines = itertools.cycle(colors)
for i in range(len(x)):
    color = next(color_iter_plots)
    ax[i].plot(times * 1e3, amplitude[:, i] * 1e9, label=f"x = {x[i] * L0 * 1e6:.2f} um", color=color)
    ax[i].set_xlabel("Time (ms)")
    ax[i].set_ylabel("Amplitude (nm)")
    ax[i].set_ylim(y_lim_min, y_lim_max)
    # ax[i].set_title(f"Wave amplitude at x = {x[i] * L0 * 1e6:.2f} um")
    ax[i].legend()
    ax[i].grid()
    ax[-1].axvline(x[i]* L0 * 1e6, color=color)

ax[-1].set_xlabel("Simulated Space (um)")    
ax[-1].set_xlim(0, L0 * 2* np.pi * 1e6)
ax[-1].set_ylabel("")
# plt.plot(times*1e3, amplitude * 1e9, label="Amplitude")
# plt.xlabel("Time (ms)")
# plt.ylabel("Amplitude (nm)")
# title = ""
# for i in range(len(x)):
#     title += f"x = {x[i]*L0*1e6:.2f} um, "
## increase the space between subplots
# fig.get_layout_engine().set(w_pad= 0.2)
# fig.get_layout_engine().set(bottom= 0.02)
fig.get_layout_engine().set(rect=(0.01,0.01,0.97,0.97))
fig.savefig(f"figures/wave_amplitude_over_time_{title}.png", dpi=300, bbox_inches='tight')

fig = plt.figure(figsize=(13.33, 7.5))
plt.plot(data[frame, :, 0] * L0 * 1e6, data[frame, :, 1] * L0 * 1e9)
plt.xlabel("Simulated Space (um)")
plt.ylabel("Amplitude (nm)")
plt.title(f"Wave at Frame {frame} - {title}")
fig.savefig(f"figures/wave_at_frame_{frame}_{title}.png", dpi=300, bbox_inches='tight')

plt.show()