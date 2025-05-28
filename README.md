# superfluid-dynamics
A repo aimed at simulating the dynamics of fluids, in particular superfluid helium using boundary integral methods.

This repo will use CUDA to accelarate solving the dynamics of superfluid helium using boundary integral methods.

The repo will use the method developped by [Roberts (1983)](https://doi.org/10.1093/imamat/31.1.13) to solve the dynamics of superfluid helium.

First we will implement the method as if it was related to fluids subject to gravity as originally done in the paper, then we will adapt it to the case of superfluid helium.

CUDA will be used throughout the repo to accelerate the computations and hopefully make it possible to simulate many particles in a more reasonable time.