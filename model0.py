import matplotlib

matplotlib.use("macosx")  # issues in mac; for linux "tkagg"
import matplotlib.pyplot as plt

from pyhgf.model import Network
from pyhgf.response import total_gaussian_surprise
import jax.numpy as jnp
import jax.random as random

print(f"Matplotlib backend: {matplotlib.get_backend()}")

# global vars as dictionary
# parameters from table 1
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

N_TRIALS = 175
SEED = 42


def create_unified_network(p: dict = PARAMS) -> Network:
    network = (
        Network().add_nodes(n_nodes=2, precision=p["input_precision"])
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


# using both networks to generate observations
def generate_observations(
    network: Network, n_trials: int = N_TRIALS, rng_key=None
) -> tuple[Network, jnp.ndarray]:
    if rng_key is None:
        rng_key = random.key(SEED)

    network = network.create_belief_propagation_fn(sampling_fn=True)
    network = network.sample(
        n_predictions=1,
        time_steps=jnp.ones(n_trials),
        rng_key=rng_key,
    )

    # nodes 0,1 hold the generated observations in samples[idx]['mean']
    u_a = network.samples[0]["mean"][0]
    u_b = network.samples[1]["mean"][0]
    observations = jnp.column_stack([u_a, u_b])

    return network, observations


def run_inference(network: Network, observations: jnp.ndarray) -> Network:
    obs = network.input_data(input_data=observations)

    return obs


# def run_inference_batch( network_factory: callable, observations_list: list[np.ndarray],
# ) -> list[tuple[Network, dict]]:

#    results = []
#    n_datasets = len(observations_list)


#    for i, obs in enumerate(observations_list):

#        network = network_factory()
#        network, details = run_inference(network, obs,  return_details=True)
#        details['dataset_idx'] = i
#        results.append((network, details))

#    return results

# model comparison
# def compute_model_metrics(network: Network) -> dict:


#    trajectories = network.node_trajectories

# sum surprise across input nodes (0 and 1)
#    total_surprise = 0.0
#    for input_idx in [0, 1]:
#        surprise = calculate_surprise(trajectories[input_idx])
# skip nan values
#        total_surprise += np.nansum(surprise)

# negative log-likelihood
#    nll = total_surprise

#    n_params = count_free_parameters(network)


#    n_obs = len(trajectories[0]['mean'])

#    aic = 2 * n_params + 2 * nll


#    bic = n_params * np.log(n_obs) + 2 * nll


#    return {
#        'nll': nll,
#        'aic': aic,
#        'bic': bic,
#        'n_params': n_params,
#        'n_obs': n_obs,
#        'total_surprise': total_surprise
#    }


# def count_free_parameters(network: Network) -> int:

# count state nodes
#    n_continuous_nodes = network.n_nodes - 2  # exclude inputs 0, 1

#    n_params = n_continuous_nodes

# count coupling parameters
#    edges = network.edges  # tuple of adjacencylists

#    for node_edges in edges:
# count volatility children connections
#        if node_edges.volatility_children is not None:
#            n_params += len(node_edges.volatility_children)

# count value children connections (alpha parameters) for non-input parent nodes
#        if node_edges.value_children is not None:
# only count if this is a higher-level node
# by checking if it has value_parents
# or if it connects to volatility nodes
#            if node_edges.value_parents is None:  # top-level nodes have coupling params
#               n_params += len(node_edges.value_children)

#    return n_params


def main():
    print("=" * 70)
    print("gHGF 2x2 Model Comparison: Unified vs Separate Global Volatility")
    print("=" * 70)

    print("\n[1/3] Generating observations from both generative models...")

    gen_nets = {}  # store generating networks (with .samples for plotting)
    obs_sets = {}  # store observation arrays

    unified_gen, obs_unified = generate_observations(
        create_unified_network(), n_trials=N_TRIALS, rng_key=random.key(SEED)
    )
    gen_nets["unified"] = unified_gen
    obs_sets["unified"] = obs_unified
    print(f"  Unified:  obs shape {obs_unified.shape}")

    separate_gen, obs_separate = generate_observations(
        create_separate_network(), n_trials=N_TRIALS, rng_key=random.key(SEED + 1)
    )
    gen_nets["separate"] = separate_gen
    obs_sets["separate"] = obs_separate
    print(f"  Separate: obs shape {obs_separate.shape}")

    print("\n[2/3] Running 2x2 inference...")

    model_factories = {
        "unified": create_unified_network,
        "separate": create_separate_network,
    }

    results = {}
    for gen_label in ["unified", "separate"]:
        obs = obs_sets[gen_label]
        for inf_label in ["unified", "separate"]:
            key = (gen_label, inf_label)

            # must enable sampling_fn=True so  plot_samples can be called later
            net = model_factories[inf_label]()
            net = net.create_belief_propagation_fn(sampling_fn=True)
            net = run_inference(net, obs)

            # calc surprise
            surprise_per_trial = net.surprise(response_function=total_gaussian_surprise)
            total_surprise = float(jnp.nansum(surprise_per_trial))

            results[key] = {
                "network": net,
                "total_surprise": total_surprise,
            }
            print(
                f"  {gen_label:>8s} obs -> {inf_label:>8s} model: "
                f"surprise = {total_surprise:.1f}"
            )

    # sum up and try to use native plots
    print("\n[3/3] Results Summary")
    print("-" * 60)
    print(f"{'':>20} {'Unified Obs':>15} {'Separate Obs':>15}")
    print("-" * 60)
    for inf_label in ["unified", "separate"]:
        u = results[("unified", inf_label)]["total_surprise"]
        s = results[("separate", inf_label)]["total_surprise"]
        print(f"{inf_label + ' inference':>20} {u:>15.1f} {s:>15.1f}")
    print("-" * 60)

    for gen_label in ["unified", "separate"]:
        u = results[(gen_label, "unified")]["total_surprise"]
        s = results[(gen_label, "separate")]["total_surprise"]
        winner = "UNIFIED" if u < s else "SEPARATE"
        print(
            f"  {gen_label.upper()} obs -> preferred: {winner} "
            f"(delta = {abs(u - s):.1f})"
        )

    print("\n--- Generating native pyhgf plots ---")

    # Network  diagrams
    for label, factory in model_factories.items():
        print(f"  Network structure ({label})...")
        try:
            factory().input_data(input_data=jnp.zeros((10, 2))).plot_network(
                backend="networkx"
            )
        except (ModuleNotFoundError, ImportError) as e:
            print(f"    Skipped — missing dependency: {e}")
            print(f"    Fix: pip install pydot")

    # belief trajectories for each of the 4 conditions
    for gen_label in ["unified", "separate"]:
        for inf_label in ["unified", "separate"]:
            key = (gen_label, inf_label)
            net = results[key]["network"]
            print(f"  Trajectories: gen={gen_label}, inf={inf_label}...")
            net.plot_trajectories()

    # forward-sampling predictions from one fitted model
    print("  Forward predictions from unified->unified...")
    net_example = results[("unified", "unified")]["network"]
    net_example = net_example.sample(
        n_predictions=5, time_steps=jnp.ones(50), rng_key=random.key(99)
    )
    net_example.plot_samples()

    print("\nDone.")
    print(f"Number of open figures: {len(plt.get_fignums())}")
    print(f"Figure numbers: {plt.get_fignums()}")
    plt.show(block=True)

    return gen_nets, obs_sets, results


if __name__ == "__main__":
    gen_nets, obs_sets, results = main()
