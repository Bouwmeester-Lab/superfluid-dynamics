import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

## variables
path = r"C:\Users\emore\OneDrive\Desktop\data.h5"
H0 = 15e-9 # 15 nm
g = 3 * 2.6e-24 / np.power(H0, 4)
L0 = 500e-6/(2.0*np.pi)
_t0 = np.sqrt(L0 / g)
x = 6 # from 0 to 2*pi

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
    print("Iteration Numbers: %s" % iteration_numbers)
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
print(values_sorted)
point = 10
frame = 11
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
plt.figure(figsize=(10, 6))
plt.plot(times*1e3, amplitude * 1e9, label="Amplitude")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude (nm)")
plt.title(f"Wave amplitude at {x*L0*1e6:.2e} um, L = {L0*1e6*2.0*np.pi:.2e} um")
plt.show()