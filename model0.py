# %%
import os
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pyhgf.model import Network

##TODO: add order naming convention, 1st level, 2nd level, 3rd level


def simulate_generative_model(n_trials: int = 175, seed: int = 42) -> dict:
    np.random.seed(seed)

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
    x_a_check[0] = 5.0  # higher initial local volatility for state a
    x_b_check[0] = 5.0  # lower initial local volatility for state b
    x_a[0] = 0.0
    x_b[0] = 0.0

    # pre-specify global volatility trajectory
    # two bursts: upward around trial 30, downward around trial 100
    x_c_trajectory = np.zeros(n_trials)
    # first burst (upward drift from trial 30 to 50)
    x_c_trajectory[30:50] = np.linspace(0, 0.8, 20)
    x_c_trajectory[50:100] = 0.8
    # second burst (downward drift from trial 100 to 120)
    x_c_trajectory[100:120] = np.linspace(0.8, -0.3, 20)
    x_c_trajectory[120:] = -0.3
    x_c = x_c_trajectory

    # simulate forward
    for k in range(1, n_trials):
        # local volatility states receive drift from global volatility parent
        # x_a evolves as GRW with drift from x_c
        drift_a_check = alpha_c_a * x_c[k]
        step_size_a_check = np.exp(omega_a_check)
        x_a_check[k] = (
            x_a_check[k - 1] + drift_a_check + np.sqrt(step_size_a_check) * np.random.randn()
        )

        drift_b_check = alpha_c_b * x_c[k]
        step_size_b_check = np.exp(omega_b_check)
        x_b_check[k] = (
            x_b_check[k - 1] + drift_b_check + np.sqrt(step_size_b_check) * np.random.randn()
        )

        # hidden states x_a and x_b evolve with volatility determined by local volatility parents
        step_size_a = np.exp(omega_a + kappa_a * x_a_check[k])
        x_a[k] = x_a[k - 1] + np.sqrt(step_size_a) * np.random.randn()

        step_size_b = np.exp(omega_b + kappa_b * x_b_check[k])
        x_b[k] = x_b[k - 1] + np.sqrt(step_size_b) * np.random.randn()

    # observations with noise
    u_a = x_a + np.sqrt(eps_a) * np.random.randn(n_trials)
    u_b = x_b + np.sqrt(eps_b) * np.random.randn(n_trials)

    return {
        "x_c": x_c,
        "x_a_check": x_a_check,
        "x_b_check": x_b_check,
        "x_a": x_a,
        "x_b": x_b,
        "u_a": u_a,
        "u_b": u_b,
        "n_trials": n_trials,
    }


def create_unified_global_volatility_hgf() -> Network:
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
            tonic_volatility=-2.0,  # omega_a
            volatility_coupling_children=(0.5),  # kappa_a
        )
        # parent of x_b
        .add_nodes(
            volatility_children=3,
            mean=6.0,
            precision=1.0,
            tonic_volatility=-2.0,  # omega_b
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

    return network


def create_separate_global_volatility_hgf() -> Network:
    network = (
        Network()
        .add_nodes(n_nodes=2, precision=1.0)
        .add_nodes(value_children=0, mean=0.0, precision=0.5, tonic_volatility=-2.0)
        .add_nodes(value_children=1, mean=0.0, precision=0.5, tonic_volatility=-2.0)
        # local volatility coupling on both "branches"
        .add_nodes(
            volatility_children=2,
            mean=6.0,
            precision=1.0,
            tonic_volatility=-2.0,
            volatility_coupling_children=(0.5),
        )
        .add_nodes(
            volatility_children=3,
            mean=6.0,
            precision=1.0,
            tonic_volatility=-2.0,
            volatility_coupling_children=(0.5),
        )
        #  global volatilities
        .add_nodes(
            value_children=4,
            mean=0.0,
            precision=1.0,
            tonic_volatility=0.0,
            value_coupling_children=(0.05),
        )
        .add_nodes(
            value_children=5,
            mean=0.0,
            precision=1.0,
            tonic_volatility=0.0,
            value_coupling_children=(0.05),
        )
    )

    return network


# # try gated approach
# def create_gated_local_global_volatility_hgf() -> Network:

#     network = (
#         Network()
#         .add_nodes(
#             n_nodes=2,
#             precision=1.0,  # Input precision (inverse of observation noise)
#         )

#         .add_nodes(
#             value_children=0,
#             mean=0.0,
#             precision=0.5,
#             tonic_volatility=-2.0,  # omega_a
#         )
#         #parent of input 1
#         .add_nodes(
#             value_children=1,
#             mean=0.0,
#             precision=0.5,
#             tonic_volatility=-2.0,  # omega_b
#         )
#         # local volatility state x_a gated by the global volatility
#         .add_nodes(
#             kind="continuous-state",
#             volatility_children=2,
#             mean=6.0,  # higher initial volatility for state a
#             precision=1.0,
#             tonic_volatility=-3.0,  # omega_a
#             volatility_coupling_children=(0.5,),  # kappa_a
#         )
#         # local volatility state x_b as volatility parent of x_b (node 5)

#         .add_nodes(
#             kind="continuous-state",
#             volatility_children=3,
#             mean=6.0,
#             precision=1.0,
#             tonic_volatility=-3.0,  # omega_b
#             volatility_coupling_children=(0.5,),  # kappa_b
#         )
#         #  global volatility state x_c as a gating parent

#         .add_nodes(
#             kind="continuous-state",
#             mean=0.0,
#             precision=1.0,
#             tonic_volatility=0.0,  # omega_c
#         )
#         # gating connections from global to local volatilities
#         .add_edges(
#             kind="coupling",
#             parent_idxs=6,  # Global volatility
#             children_idxs=[4, 5],  # Both local volatility states
#             coupling_strengths=[0.05, 0.05],  # Gating strength
#         )
#     )

#     return network


def run_inference(
    network: Network, observations: np.ndarray, return_details: bool = False
) -> Network | tuple[Network, dict]:
    observations = np.atleast_2d(observations)
    if observations.shape[0] < observations.shape[1]:
        observations = observations.T  # ensure (n_trials, n_inputs)

    n_trials, n_inputs = observations.shape

    # look for empty/wrong values
    n_nan = np.sum(np.isnan(observations))
    n_inf = np.sum(np.isinf(observations))

    if n_inf > 0:
        raise ValueError(f"observations contain {n_inf} infinite values")

    # time inference
    start_time = time.time()
    network = network.input_data(input_data=observations)
    elapsed = time.time() - start_time

    # get trajectories for diagnostics
    trajectories = network.node_trajectories

    if return_details:
        details = {
            "elapsed_time": elapsed,
            "n_trials": n_trials,
            "n_inputs": n_inputs,
            "trajectories": trajectories,
        }
        return network, details

    return network


def run_inference_batch(
    network_factory: callable,
    observations_list: list[np.ndarray],
) -> list[tuple[Network, dict]]:
    results = []
    n_datasets = len(observations_list)

    for i, obs in enumerate(observations_list):
        network = network_factory()
        network, details = run_inference(
            network,
            obs,
            return_details=True,
        )
        details["dataset_idx"] = i
        results.append((network, details))

    return results


# model comparison
def compute_model_metrics(network: Network) -> dict:
    trajectories = network.node_trajectories

    # sum surprise across input nodes (0 and 1)
    total_surprise = 0.0
    for input_idx in [0, 1]:
        surprise = calculate_surprise(trajectories[input_idx])
        # skip nan values
        total_surprise += np.nansum(surprise)

    # negative log-likelihood
    nll = total_surprise

    n_params = count_free_parameters(network)

    n_obs = len(trajectories[0]["mean"])

    aic = 2 * n_params + 2 * nll

    bic = n_params * np.log(n_obs) + 2 * nll

    return {
        "nll": nll,
        "aic": aic,
        "bic": bic,
        "n_params": n_params,
        "n_obs": n_obs,
        "total_surprise": total_surprise,
    }


def count_free_parameters(network: Network) -> int:
    # count state nodes
    n_continuous_nodes = network.n_nodes - 2  # exclude inputs 0, 1

    n_params = n_continuous_nodes

    # count coupling parameters
    edges = network.edges  # tuple of adjacencylists

    for node_edges in edges:
        # count volatility children connections
        if node_edges.volatility_children is not None:
            n_params += len(node_edges.volatility_children)

        # count value children connections (alpha parameters) for non-input parent nodes
        if node_edges.value_children is not None:
            # only count if this is a higher-level node
            # by checking if it has value_parents
            # or if it connects to volatility nodes
            if node_edges.value_parents is None:  # top-level nodes have coupling params
                n_params += len(node_edges.value_children)

    return n_params


def compare_models(observations: np.ndarray) -> dict:
    # create and fit unified model
    unified_network = create_unified_global_volatility_hgf()
    unified_network = run_inference(unified_network, observations)
    unified_metrics = compute_model_metrics(unified_network)

    # create and fit separate model
    separate_network = create_separate_global_volatility_hgf()
    separate_network = run_inference(separate_network, observations)
    separate_metrics = compute_model_metrics(separate_network)

    # compute differences (negative = unified is better)
    delta_aic = unified_metrics["aic"] - separate_metrics["aic"]
    delta_bic = unified_metrics["bic"] - separate_metrics["bic"]

    print("=" * 50)
    print("Model Comparison: Unified vs Separate Global Volatility")
    print("=" * 50)
    print(f"\nUnified Model (shared global volatility):")
    print(f"  Parameters: {unified_metrics['n_params']}")
    print(f"  NLL: {unified_metrics['nll']:.2f}")
    print(f"  AIC: {unified_metrics['aic']:.2f}")
    print(f"  BIC: {unified_metrics['bic']:.2f}")

    print(f"\nSeparate Model (independent global volatilities):")
    print(f"  Parameters: {separate_metrics['n_params']}")
    print(f"  NLL: {separate_metrics['nll']:.2f}")
    print(f"  AIC: {separate_metrics['aic']:.2f}")
    print(f"  BIC: {separate_metrics['bic']:.2f}")

    print(f"\nModel Comparison:")
    print(f"  ΔAIC (unified - separate): {delta_aic:.2f}")
    print(f"  ΔBIC (unified - separate): {delta_bic:.2f}")

    winner_aic = "Unified" if delta_aic < 0 else "Separate"
    winner_bic = "Unified" if delta_bic < 0 else "Separate"
    print(f"\n  Preferred by AIC: {winner_aic}")
    print(f"  Preferred by BIC: {winner_bic}")

    return {
        "unified": unified_metrics,
        "separate": separate_metrics,
        "unified_network": unified_network,
        "separate_network": separate_network,
        "delta_aic": delta_aic,
        "delta_bic": delta_bic,
    }


def plot_model_comparison(
    sim_data: dict, unified_network: Network, separate_network: Network, save_path: str = None
):
    n_trials = sim_data["n_trials"]
    trials = np.arange(n_trials)

    unified_traj = unified_network.node_trajectories
    separate_traj = separate_network.node_trajectories

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    # row 1: global volatility beliefs
    # left: true global volatility
    axes[0, 0].plot(trials, sim_data["x_c"], "k-", linewidth=2.5, label="True $x_c$")
    axes[0, 0].set_ylabel("Global Volatility")
    axes[0, 0].set_title("True State (Generative Model)")
    axes[0, 0].legend()
    axes[0, 0].set_xlim([0, n_trials])

    # right: compare global volatility beliefs
    # unified: single node 6
    axes[0, 1].plot(
        trials, unified_traj[6]["mean"], "purple", linewidth=2, label="Unified $\\mu_c$"
    )
    # separate: nodes 6 and 7
    axes[0, 1].plot(
        trials, separate_traj[6]["mean"], "r--", linewidth=2, label="Separate $\\mu_{c,a}$"
    )
    axes[0, 1].plot(
        trials, separate_traj[7]["mean"], "b--", linewidth=2, label="Separate $\\mu_{c,b}$"
    )
    axes[0, 1].plot(trials, sim_data["x_c"], "k:", linewidth=1.5, alpha=0.5, label="True $x_c$")
    axes[0, 1].set_ylabel("Global Volatility Belief")
    axes[0, 1].set_title("Model Comparison: Global Volatility")
    axes[0, 1].legend(loc="upper right")
    axes[0, 1].set_xlim([0, n_trials])

    # row 2: local volatility beliefs
    # left: true local volatilities
    axes[1, 0].plot(trials, sim_data["x_a_check"], "r-", linewidth=2, label="True $\\check{x}_a$")
    axes[1, 0].plot(trials, sim_data["x_b_check"], "b-", linewidth=2, label="True $\\check{x}_b$")
    axes[1, 0].set_ylabel("Local Volatility")
    axes[1, 0].set_title("True States")
    axes[1, 0].legend()
    axes[1, 0].set_xlim([0, n_trials])

    # right: compare local volatility beliefs
    axes[1, 1].plot(
        trials, unified_traj[4]["mean"], "r-", linewidth=2, label="Unified $\\mu_{\\check{a}}$"
    )
    axes[1, 1].plot(
        trials, unified_traj[5]["mean"], "b-", linewidth=2, label="Unified $\\mu_{\\check{b}}$"
    )
    axes[1, 1].plot(
        trials,
        separate_traj[4]["mean"],
        "r--",
        linewidth=2,
        alpha=0.7,
        label="Separate $\\mu_{\\check{a}}$",
    )
    axes[1, 1].plot(
        trials,
        separate_traj[5]["mean"],
        "b--",
        linewidth=2,
        alpha=0.7,
        label="Separate $\\mu_{\\check{b}}$",
    )
    axes[1, 1].set_ylabel("Local Volatility Belief")
    axes[1, 1].set_title("Model Comparison: Local Volatility")
    axes[1, 1].legend(loc="upper right", fontsize=8)
    axes[1, 1].set_xlim([0, n_trials])

    # row 3: hidden state beliefs with observations
    # left: observations and true states
    axes[2, 0].scatter(trials, sim_data["u_a"], alpha=0.3, s=8, c="red", label="Obs $u_a$")
    axes[2, 0].scatter(trials, sim_data["u_b"], alpha=0.3, s=8, c="blue", label="Obs $u_b$")
    axes[2, 0].plot(trials, sim_data["x_a"], "r-", linewidth=2, label="True $x_a$")
    axes[2, 0].plot(trials, sim_data["x_b"], "b-", linewidth=2, label="True $x_b$")
    axes[2, 0].set_xlabel("Trial")
    axes[2, 0].set_ylabel("Hidden State")
    axes[2, 0].set_title("Observations & True States")
    axes[2, 0].legend(loc="upper right", fontsize=8)
    axes[2, 0].set_xlim([0, n_trials])

    # right: compare hidden state beliefs
    axes[2, 1].scatter(trials, sim_data["u_a"], alpha=0.2, s=5, c="gray")
    axes[2, 1].scatter(trials, sim_data["u_b"], alpha=0.2, s=5, c="gray")
    axes[2, 1].plot(trials, unified_traj[2]["mean"], "r-", linewidth=2, label="Unified $\\mu_a$")
    axes[2, 1].plot(trials, unified_traj[3]["mean"], "b-", linewidth=2, label="Unified $\\mu_b$")
    axes[2, 1].plot(
        trials, separate_traj[2]["mean"], "r--", linewidth=2, alpha=0.7, label="Separate $\\mu_a$"
    )
    axes[2, 1].plot(
        trials, separate_traj[3]["mean"], "b--", linewidth=2, alpha=0.7, label="Separate $\\mu_b$"
    )
    axes[2, 1].set_xlabel("Trial")
    axes[2, 1].set_ylabel("Hidden State Belief")
    axes[2, 1].set_title("Model Comparison: Hidden States")
    axes[2, 1].legend(loc="upper right", fontsize=8)
    axes[2, 1].set_xlim([0, n_trials])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()
    return fig


def plot_global_volatility_focus(
    sim_data: dict, unified_network: Network, separate_network: Network, save_path: str = None
):
    n_trials = sim_data["n_trials"]
    trials = np.arange(n_trials)

    unified_traj = unified_network.node_trajectories
    separate_traj = separate_network.node_trajectories

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # left: unified model - single shared global volatility
    axes[0].fill_between(
        trials,
        sim_data["x_c"] - 0.5,
        sim_data["x_c"] + 0.5,
        alpha=0.2,
        color="gray",
        label="True $x_c$ range",
    )
    axes[0].plot(trials, sim_data["x_c"], "k-", linewidth=2.5, label="True $x_c$")
    axes[0].plot(trials, unified_traj[6]["mean"], "purple", linewidth=2.5, label="Belief $\\mu_c$")
    axes[0].axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    axes[0].set_xlabel("Trial")
    axes[0].set_ylabel("Global Volatility")
    axes[0].set_title("UNIFIED MODEL\n(Shared Global Volatility)")
    axes[0].legend(loc="upper right")
    axes[0].set_xlim([0, n_trials])

    # right: separate model - two independent global volatilities
    axes[1].fill_between(
        trials,
        sim_data["x_c"] - 0.5,
        sim_data["x_c"] + 0.5,
        alpha=0.2,
        color="gray",
        label="True $x_c$ range",
    )
    axes[1].plot(trials, sim_data["x_c"], "k-", linewidth=2.5, label="True $x_c$")
    axes[1].plot(
        trials,
        separate_traj[6]["mean"],
        "r-",
        linewidth=2.5,
        label="Belief $\\mu_{c,a}$ (branch a)",
    )
    axes[1].plot(
        trials,
        separate_traj[7]["mean"],
        "b-",
        linewidth=2.5,
        label="Belief $\\mu_{c,b}$ (branch b)",
    )
    axes[1].axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    axes[1].set_xlabel("Trial")
    axes[1].set_ylabel("Global Volatility")
    axes[1].set_title("SEPARATE MODEL\n(Independent Global Volatilities)")
    axes[1].legend(loc="upper right")
    axes[1].set_xlim([0, n_trials])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()
    return fig


def calculate_surprise(traj: dict) -> np.ndarray:
    if "surprise" in traj:
        return traj["surprise"]

    if "expected_mean" in traj and "expected_precision" in traj and "observed" in traj:
        mu = np.array(traj["expected_mean"])
        pi = np.array(traj["expected_precision"])
        u = np.array(traj["observed"])

        # avoid log(0)
        safe_pi = np.where(pi <= 1e-10, 1e-10, pi)

        # gaussian surprise: -ln P(u)
        # 0.5 * ln(2pi) - 0.5 * ln(pi) + 0.5 * pi * (u - mu)^2
        # note: 0.5 * ln(2pi / pi) = 0.5 * ln(2pi) - 0.5 * ln(pi)

        nll = 0.5 * np.log(2 * np.pi) - 0.5 * np.log(safe_pi) + 0.5 * pi * (u - mu) ** 2
        return nll

    return np.zeros(len(traj["mean"]))


def plot_surprise_comparison(
    unified_network: Network, separate_network: Network, save_path: str = None
):
    unified_traj = unified_network.node_trajectories
    separate_traj = separate_network.node_trajectories

    # get surprise from input nodes
    unified_surprise_a = np.nan_to_num(calculate_surprise(unified_traj[0]))
    unified_surprise_b = np.nan_to_num(calculate_surprise(unified_traj[1]))
    separate_surprise_a = np.nan_to_num(calculate_surprise(separate_traj[0]))
    separate_surprise_b = np.nan_to_num(calculate_surprise(separate_traj[1]))

    n_trials = len(unified_surprise_a)
    trials = np.arange(n_trials)

    # cumulative surprise
    unified_cumsum = np.cumsum(unified_surprise_a + unified_surprise_b)
    separate_cumsum = np.cumsum(separate_surprise_a + separate_surprise_b)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # left: trial-by-trial surprise
    axes[0].plot(
        trials,
        unified_surprise_a + unified_surprise_b,
        "purple",
        linewidth=1.5,
        alpha=0.7,
        label="Unified",
    )
    axes[0].plot(
        trials,
        separate_surprise_a + separate_surprise_b,
        "orange",
        linewidth=1.5,
        alpha=0.7,
        label="Separate",
    )
    axes[0].set_xlabel("Trial")
    axes[0].set_ylabel("Surprise (per trial)")
    axes[0].set_title("Trial-by-Trial Surprise")
    axes[0].legend()
    axes[0].set_xlim([0, n_trials])

    # right: cumulative surprise
    axes[1].plot(trials, unified_cumsum, "purple", linewidth=2.5, label="Unified")
    axes[1].plot(trials, separate_cumsum, "orange", linewidth=2.5, label="Separate")
    axes[1].set_xlabel("Trial")
    axes[1].set_ylabel("Cumulative Surprise (NLL)")
    axes[1].set_title("Cumulative Surprise (Lower = Better Fit)")
    axes[1].legend()
    axes[1].set_xlim([0, n_trials])

    # add final values as text
    axes[1].text(
        0.95,
        0.95,
        f"Unified: {unified_cumsum[-1]:.1f}\nSeparate: {separate_cumsum[-1]:.1f}",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()
    return fig


def plot_network_structure(
    unified_network: Network, separate_network: Network, save_path: str = None
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    models = [
        ("Unified Model\n(Shared Global Volatility)", unified_network, axes[0]),
        ("Separate Model\n(Independent Global Volatilities)", separate_network, axes[1]),
    ]

    for title, net, ax in models:
        ax.set_title(title, fontsize=12, fontweight="bold")

        try:
            G = nx.DiGraph()
            n_nodes = net.n_nodes
            edges_info = net.edges

            # --- Build Graph ---
            for parent_idx, edge_data in enumerate(edges_info):
                # Value children (drift coupling) -> Solid Lines
                if edge_data.value_children is not None:
                    children = edge_data.value_children
                    if isinstance(children, (int, float, np.integer)):
                        children = [int(children)]
                    for child_idx in children:
                        # Solid black line
                        G.add_edge(
                            parent_idx, child_idx, type="value", color="black", style="solid"
                        )

                # Volatility children (variance coupling) -> Dashed Lines
                if edge_data.volatility_children is not None:
                    children = edge_data.volatility_children
                    if isinstance(children, (int, float, np.integer)):
                        children = [int(children)]
                    for child_idx in children:
                        # Dashed black line
                        G.add_edge(
                            parent_idx, child_idx, type="volatility", color="black", style="dashed"
                        )

            # --- Layout ---
            pos = {}
            # Level 0: Inputs (0, 1) - Bottom
            pos[0] = (0.3, 0)
            pos[1] = (0.7, 0)

            # Level 1: Hidden States (2, 3)
            pos[2] = (0.3, 1)
            pos[3] = (0.7, 1)

            # Level 2: Local Volatility (4, 5)
            pos[4] = (0.3, 2)
            pos[5] = (0.7, 2)

            # Level 3: Global Volatility (6...7)
            if n_nodes == 7:  # Unified
                pos[6] = (0.5, 3)
            elif n_nodes >= 8:  # Separate
                pos[6] = (0.3, 3)
                pos[7] = (0.7, 3)
            else:
                pos = nx.spring_layout(G)

            # --- Drawing ---

            # Draw edges
            edges = G.edges()
            if edges:
                edge_styles = [G[u][v]["style"] for u, v in edges]

                nx.draw_networkx_edges(
                    G,
                    pos,
                    ax=ax,
                    edge_color="black",
                    style=edge_styles,
                    arrows=True,
                    arrowsize=20,
                    width=1.5,
                    node_size=600,
                )

            # Draw nodes
            # Style: Black and white
            # Observed nodes (0, 1): Filled gray/black
            # Latent nodes: White with black border
            node_colors = []
            for n in G.nodes():
                if n in [0, 1]:
                    node_colors.append("#808080")  # Filled gray for observed
                else:
                    node_colors.append("white")  # White for latent

            nx.draw_networkx_nodes(
                G,
                pos,
                ax=ax,
                node_color=node_colors,
                node_size=600,
                edgecolors="black",
                linewidths=1.5,
            )

            # Draw labels (Node IDs or Names)
            labels = {
                0: "$u_a$",
                1: "$u_b$",
                2: "$x_a$",
                3: "$x_b$",
                4: "$\\check{x}_a$",
                5: "$\\check{x}_b$",
                6: "$x_c$" if n_nodes == 7 else "$x_{c,a}$",
                7: "$x_{c,b}$",
            }
            labels = {k: v for k, v in labels.items() if k in G.nodes}
            nx.draw_networkx_labels(G, pos, ax=ax, labels=labels, font_size=10, font_color="black")

            # Simple Legend
            from matplotlib.lines import Line2D

            legend_elements = [
                Line2D([0], [0], color="black", lw=2, label="Value Coupling (Mean)"),
                Line2D([0], [0], color="black", lw=2, linestyle="--", label="Vol. Coupling (Var)"),
            ]
            ax.legend(
                handles=legend_elements,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.05),
                fontsize="small",
                ncol=2,
                frameon=False,
            )

            ax.axis("off")

        except Exception as e:
            print(f"Plotting failed for {title}: {e}")
            import traceback

            traceback.print_exc()
            ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()
    return fig


def run_and_plot_comparison(n_trials: int = 175, seed: int = 42, save_dir: str = "outputs"):
    os.makedirs(save_dir, exist_ok=True)

    # simulate data
    print("Simulating generative model...")
    sim_data = simulate_generative_model(n_trials=n_trials, seed=seed)
    observations = np.column_stack([sim_data["u_a"], sim_data["u_b"]])

    # create and fit unified model
    print("Fitting unified model (shared global volatility)...")
    unified_network = create_unified_global_volatility_hgf()
    unified_network = run_inference(unified_network, observations)

    # create and fit separate model
    print("Fitting separate model (independent global volatilities)...")
    separate_network = create_separate_global_volatility_hgf()
    separate_network = run_inference(separate_network, observations)

    # compute metrics
    unified_metrics = compute_model_metrics(unified_network)
    separate_metrics = compute_model_metrics(separate_network)

    # print comparison
    print("\n" + "=" * 50)
    print("MODEL COMPARISON RESULTS")
    print("=" * 50)
    print(
        f"\nUnified (shared):     AIC = {unified_metrics['aic']:.1f}, BIC = {unified_metrics['bic']:.1f}"
    )
    print(
        f"Separate (independent): AIC = {separate_metrics['aic']:.1f}, BIC = {separate_metrics['bic']:.1f}"
    )
    delta_aic = unified_metrics["aic"] - separate_metrics["aic"]
    delta_bic = unified_metrics["bic"] - separate_metrics["bic"]
    winner = "UNIFIED" if delta_aic < 0 else "SEPARATE"
    print(f"\nPreferred model: {winner} (ΔAIC = {delta_aic:.1f})")

    # generate plots
    print("\nGenerating plots...")

    plot_model_comparison(
        sim_data, unified_network, separate_network, save_path=f"{save_dir}/model_comparison.png"
    )

    plot_global_volatility_focus(
        sim_data,
        unified_network,
        separate_network,
        save_path=f"{save_dir}/global_volatility_focus.png",
    )

    plot_surprise_comparison(
        unified_network, separate_network, save_path=f"{save_dir}/surprise_comparison.png"
    )

    print("Generating network structure plot...")
    plot_network_structure(
        unified_network, separate_network, save_path=f"{save_dir}/network_structure.png"
    )

    return {
        "sim_data": sim_data,
        "unified_network": unified_network,
        "separate_network": separate_network,
        "unified_metrics": unified_metrics,
        "separate_metrics": separate_metrics,
    }


def main():
    # run model comparison between unified and separate global volatility models.
    results = run_and_plot_comparison(n_trials=175, seed=42, save_dir="outputs")
    return results


if __name__ == "__main__":
    results = main()

# %%
