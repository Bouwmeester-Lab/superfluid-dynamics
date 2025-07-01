import numpy as np

def X(j, t, h, omega):
    return j - h*np.sin(j - omega * t)
def Y(j, t, h, omega):
    return h * np.cos(j - omega * t)
def Phi(j, t, h, omega):
    return h * omega * np.sin(j - omega * t)