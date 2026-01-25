# %%
import os
import time

import matplotlib
import numpy as np
from pyhgf.model import Network

# %% globals
# parameters from table 1
omega_c = 0.0  # global volatility
omega_a_check = -3.0  # local volatility of state a
omega_b_check = -3.0  # local volatility of state b
omega_a = -2.0  # global volatility of state a
omega_b = -2.0  # global volatility of state b
alpha_c_a = 0.05  # coupling strength: x_c -> x_a (value coupling)
alpha_c_b = 0.05  # coupling strength: x_c -> x_b (value coupling)
kappa_a = 0.5  # coupling strength: x_a -> x_a (volatility coupling)
kappa_b = 0.5  # coupling strength: x_b -> x_b (volatility coupling)
eps_a = 1.0  # observation noise for u_a
eps_b = 1.0  # observation noise for u_b


# %% simulate_gen_model
def simulate_generative_model(n_trials: int = 175, seed: int = 42) -> dict:
    np.random.seed(seed)
    # initial states
    x_c = np.zeros(n_trials)
    x_a_check = np.zeros(n_trials)
    x_b_check = np.zeros(n_trials)
    x_a = np.zeros(n_trials)
    x_b = np.zeros(n_trials)
    u_a = np.zeros(n_trials)
    u_b = np.zeros(n_trials)

    # starting values
    x_c[0] = 0.0
    x_a_check[0] = omega_a_check
    x_b_check[0] = omega_b_check

    # DEFINE GLOBAL VOLATILITY MAP
    x_c_trajectory = np.zeros(n_trials)
    x_c_trajectory[30:50] = np.linspace(
        0.0,
        0.8,
        20,
    )
    x_c_trajectory[50:100] = 0.8
    x_c_trajectory[100:120] = np.linspace(
        0.8,
        -0.3,
        20,
    )
    x_c_trajectory[120:] = -0.3
    x_c = x_c_trajectory

    network = (
        Network()
        .add_nodes(
            n_nodes=2,
            precision=1.0,  # input precision
        )
        # parent of input 0
        .add_nodes(
            value_children=0,
            mean=0.0,
            precision=0.5,
            tonic_volatility=-2.0,  # omega_a
        )
        # parent of input 1
        .add_nodes(
            value_children=1,
            mean=0.0,
            precision=0.5,
            tonic_volatility=-2.0,  # omega_b
        )
        # volatility parent of x_a
        .add_nodes(
            volatility_children=2,
            mean=6.0,
            precision=1.0,
            tonic_volatility=-3.0,  # omega_a
            volatility_coupling_children=(0.5),  # kappa_a
        )
        # parent of x_b
        .add_nodes(
            volatility_children=3,
            mean=4.0,
            precision=1.0,
            tonic_volatility=-3.0,  # omega_b
            volatility_coupling_children=(0.5),  # kappa_b
        )
        # global volatility
        .add_nodes(
            value_children=[4, 5],
            mean=0.0,
            precision=1.0,
            tonic_volatility=0.0,  # omega_c
            value_coupling_children=(0.05, 0.05),  # drift coupling
        )
    )
