# %%
# matplotlib.use("macosx")  # issues in mac; for linux "tkagg"
import copy
import math

import jax.numpy as jnp
import jax.random as random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy import ndarray
from pyhgf.model import Network
from pyhgf.response import total_gaussian_surprise

print(f"Matplotlib backend: {matplotlib.get_backend()}")

# NAMING CONVENTION: KEY ( GenerativeModel, Inference Model)

# global vars as dictionary
# parameters from table 1
_PARAMS = dict(
    omega_c=-2.0,  # tonic volatility of global node
    omega_a_check=-3.0,  # tonic volatility of local vol a
    omega_b_check=-3.0,  # tonic volatility of local vol b
    omega_a=-2.0,  # tonic volatility of state a
    omega_b=-2.0,  # tonic volatility of state b
    alpha_c_a=0.05,  # value coupling x_c -> check_x_a (drift)
    alpha_c_b=0.05,  # value coupling x_c -> check_x_b (drift)
    kappa_a=0.5,  # volatility coupling check_x_a -> x_a
    kappa_b=0.5,  # volatility coupling check_x_b -> x_b
    input_precision=1.0,  # observation precision (1/noise variance)
    x_a_init=0.0,
    x_b_init=2.0,  # offset so branches are distinguishable
    x_a_check_init=2.0,  # moderate: check_x_a starts higher
    x_b_check_init=1.0,
    x_c_init=0.0,
    state_precision=0.5,
    vol_precision=1.0,
)

PARAMS = dict(
    omega_c=-2.0,  # tonic volatility of global node
    omega_a_check=-3.0,  # tonic volatility of local vol a
    omega_b_check=-3.0,  # tonic volatility of local vol b
    omega_a=-2.0,  # tonic volatility of state a
    omega_b=-2.0,  # tonic volatility of state b
    alpha_c_a=0.05,  # value coupling x_c -> check_x_a (drift)
    alpha_c_b=0.05,  # value coupling x_c -> check_x_b (drift)
    kappa_a=0.5,  # volatility coupling check_x_a -> x_a
    kappa_b=0.5,  # volatility coupling check_x_b -> x_b
    input_precision=1.0,  # observation precision (1/noise variance)
    x_a_init=0.0,
    x_b_init=2.0,  # offset so branches are distinguishable
    x_a_check_init=2.0,  # moderate: check_x_a starts higher
    x_b_check_init=1.0,
    x_c_init=0.0,
    state_precision=0.5,
    vol_precision=1.0,
)

N_TRIALS = 80
SEED = 42
labels = ["unified", "separate"]


def create_unified_network(p: dict = PARAMS) -> Network:
    network = (
        Network()
        .add_nodes(n_nodes=2, precision=p["input_precision"])
        # parent of input 0
        .add_nodes(
            value_children=0,
            mean=p["x_a_init"],
            precision=p["state_precision"],
            tonic_volatility=p["omega_a"],
        )
        # parent of input 1
        .add_nodes(
            value_children=1,
            mean=p["x_b_init"],
            precision=p["state_precision"],
            tonic_volatility=p["omega_b"],
        )
        # volatility parent of x_a
        .add_nodes(
            volatility_children=2,
            mean=p["x_a_check_init"],
            precision=p["vol_precision"],
            tonic_volatility=p["omega_a_check"],
            volatility_coupling_children=(p["kappa_a"],),
        )
        # parent of x_b
        .add_nodes(
            volatility_children=3,
            mean=p["x_b_check_init"],
            precision=p["vol_precision"],
            tonic_volatility=p["omega_b_check"],
            volatility_coupling_children=(p["kappa_b"],),
        )
        # global volatility
        .add_nodes(
            value_children=[4, 5],
            mean=p["x_c_init"],
            precision=p["vol_precision"],
            tonic_volatility=p["omega_c"],
            value_coupling_children=(p["alpha_c_a"], p["alpha_c_b"]),
        )
    )
    return network


def create_separate_network(p: dict = PARAMS) -> Network:
    network = (
        Network()
        .add_nodes(n_nodes=2, precision=p["input_precision"])
        .add_nodes(
            value_children=0,
            mean=p["x_a_init"],
            precision=p["state_precision"],
            tonic_volatility=p["omega_a"],
        )
        .add_nodes(
            value_children=1,
            mean=p["x_b_init"],
            precision=p["state_precision"],
            tonic_volatility=p["omega_b"],
        )
        # local volatility coupling on both "branches"
        .add_nodes(
            volatility_children=2,
            mean=p["x_a_check_init"],
            precision=p["vol_precision"],
            tonic_volatility=p["omega_a_check"],
            volatility_coupling_children=(p["kappa_a"],),
        )
        .add_nodes(
            volatility_children=3,
            mean=p["x_b_check_init"],
            precision=p["vol_precision"],
            tonic_volatility=p["omega_b_check"],
            volatility_coupling_children=(p["kappa_b"],),
        )
        #  global volatilities
        .add_nodes(
            value_children=4,
            mean=p["x_c_init"],
            precision=p["vol_precision"],
            tonic_volatility=p["omega_c"],
            value_coupling_children=(p["alpha_c_a"],),
        )
        .add_nodes(
            value_children=5,
            mean=p["x_c_init"],
            precision=p["vol_precision"],
            tonic_volatility=p["omega_c"],
            value_coupling_children=(p["alpha_c_b"],),
        )
    )
    return network


def zero_one_observations():
    # From:https://computationalpsychiatry.github.io/pyhgf/notebooks/Example_3_Multi_armed_bandit.html
    # three levels of contingencies
    high_prob, chance, low_prob = 0.8, 0.5, 0.2

    # create blocks of contingencies
    stable_contingencies = np.array([low_prob, high_prob]).repeat(30)
    volatile_contingencies = np.tile(np.array([low_prob, high_prob]).repeat(10), 4)
    chance_contingencies = np.array(chance).repeat(30)
    # create sequences of blocks for each arm/rewards
    win_arm1 = np.concatenate([stable_contingencies, chance_contingencies, volatile_contingencies])
    loss_arm1 = np.concatenate([volatile_contingencies, chance_contingencies, stable_contingencies])
    # sample trial level outcomes from the sequences
    u_win_arm1 = np.random.binomial(n=1, p=win_arm1)
    u_loss_arm1 = np.random.binomial(n=1, p=loss_arm1)

    input_data = np.array([u_win_arm1, u_loss_arm1]).T
    return input_data.astype(float)


def generate_observatinos(
    label: str,
    create_function,
    n_trials: int = N_TRIALS,
    rng_key=None,
) -> tuple[Network, ndarray]:
    p = PARAMS
    obs_ranges = np.array_split(range(N_TRIALS), 3)
    if rng_key is None:
        rng_key = random.key(SEED)

    network = create_function(p)
    network = network.create_belief_propagation_fn(sampling_fn=True)
    network = network.sample(
        n_predictions=1,
        time_steps=np.ones(len(obs_ranges[0])),
        rng_key=rng_key,
    )
    u_a = network.samples[0]["mean"][0]
    u_b = network.samples[0]["mean"][1]
    observations = np.column_stack([u_a, u_b])

    # increasing volatility
    p["omega_a"] = p["omega_a"] + 0
    p["omega_b"] = p["omega_b"] - 0
    network = create_function(p)
    network = network.create_belief_propagation_fn(sampling_fn=True)
    network = network.sample(
        n_predictions=1,
        time_steps=np.ones(len(obs_ranges[1])),
        rng_key=rng_key,
    )
    u_a = network.samples[0]["mean"][0]
    u_b = network.samples[0]["mean"][1]
    observations = np.concat((observations, np.column_stack([u_a, u_b])))
    # observations = np.concat(observations, np.column_stack([u_a, u_b]))

    # increasing volatility
    p["omega_a"] = p["omega_a"] - 0
    p["omega_b"] = p["omega_b"] + 0
    network = create_function(p)
    network = network.create_belief_propagation_fn(sampling_fn=True)
    network = network.sample(
        n_predictions=1,
        time_steps=np.ones(len(obs_ranges[1])),
        rng_key=rng_key,
    )
    u_a = network.samples[0]["mean"][0]
    u_b = network.samples[0]["mean"][1]

    observations = np.concat((observations, np.column_stack([u_a, u_b])))

    return network, observations


def run_inference(network: Network, observations: np.ndarray) -> Network:
    inference = network.input_data(input_data=observations)
    return inference


if __name__ == "__main__":
    print("*" * 20)
    print("gHGF model comparison: Unified vs separate global volatility")
    print("*" * 20)
    model_factories = {
        "unified": create_unified_network,
        "separate": create_separate_network,
    }

    gen_unified, obs_unified = generate_observatinos(
        "unified",
        create_unified_network,
        n_trials=N_TRIALS,
        rng_key=random.key(seed=SEED),
    )
    gen_separate, obs_separate = generate_observatinos(
        "separate",
        create_separate_network,
        n_trials=N_TRIALS,
        rng_key=random.key(seed=SEED),
    )

    gen_nets = {}  # store generating networks (with .samples for plotting)
    obs_sets = {}  # store observation arrays

    gen_nets["unified"] = gen_unified
    gen_nets["separate"] = gen_separate

    obs_sets["unified"] = obs_unified
    obs_sets["separate"] = obs_separate

    results = {}
    for gen_label in ["unified", "separate"]:
        obs = obs_sets[gen_label]

        for inf_label in ["unified", "separate"]:
            key = (gen_label, inf_label)
            print(">", key)

            net = model_factories[inf_label]()
            net = net.create_belief_propagation_fn(sampling_fn=True)
            net = run_inference(net, obs)
            surprice_per_trial = net.surprise(response_function=total_gaussian_surprise)
            total_surprise = float(np.nansum(surprice_per_trial))

            results[key] = {
                "network": net,
                "total_surprise": total_surprise,
            }
            print(
                f"{gen_label:>8s} obs -> {inf_label:>8s}: surprise = {total_surprise:.3f}",
            )

    # sum up and try to use native plots
    print("\n[3/3] Results Summary")
    print("-" * 60)
    print(f"{'':>20} {'Unified Obs':>15} {'Separate Obs':>15}")
    print("-" * 60)

    for gen_label in labels:
        u = results[(gen_label, "unified")]["total_surprise"]
        s = results[(gen_label, "separate")]["total_surprise"]
        winner = "UNIFIED" if u < s else "SEPARATE"
        print(f"  {gen_label.upper()} obs -> preferred: {winner} (delta = {abs(u - s):.1f})")

    # Comparing Nodewise inferences
    uu = results[("unified", "unified")]["network"].to_pandas()
    us = results[("unified", "separate")]["network"].to_pandas()
    ss = results[("separate", "separate")]["network"].to_pandas()
    su = results[("separate", "unified")]["network"].to_pandas()

    plt.plot(su["x_0_mean"][:], marker="2", linestyle=":", label="su - x0", color="orange")
    plt.plot(uu["x_0_mean"][:], marker="1", linestyle=":", label="uu - x0", color="red")
    plt.plot(us["x_0_mean"][:], marker="2", linestyle=":", label="us - x0", color="green")
    plt.plot(us["x_0_mean"][:], marker="1", linestyle=":", label="us - x0", color="blue")
    plt.plot(ss["x_0_expected_mean"][:], marker="x", linestyle=":", label="uniObser", color="black")
    plt.plot(ss["x_1_expected_mean"][:], marker="x", linestyle=":", label="sepObser", color="gray")
    plt.legend()
    plt.grid()
    plt.show()

    # Boosting observations on right arm
    print("plot against observations boosted against volatility")
    boost_obs_unified = copy.deepcopy(obs_unified)
    boost_obs_unified[math.floor(N_TRIALS / 2) :, 1] += 3

    # ploting Change accordingly
    run_inference(results[("separate", "separate")]["network"], obs_separate).plot_trajectories()
    run_inference(results[("unified", "unified")]["network"], obs_unified).plot_trajectories()
    run_inference(results[("unified", "unified")]["network"], boost_obs_unified).plot_trajectories()
    run_inference(
        results[("unified", "separate")]["network"], boost_obs_unified
    ).plot_trajectories()
    # run_inference(
    #     results[("separate", "separate")]["network"], boost_obs_unified
    # ).plot_trajectories()

    # PLOTTING FOR 0-1 inputs
    print("plot against 0s and 1s")
    run_inference(
        results[("unified", "separate")]["network"], zero_one_observations()
    ).plot_trajectories()


# %%
