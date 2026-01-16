import numpy as np
import matplotlib.pyplot as plt
from pyhgf.model import Network


def simulate_generative_model(n_trials: int = 175, seed: int = 42) -> dict:
    
    np.random.seed(seed)
    
    # parameters from Table 1
    omega_c = 0.0      # tonic volatility of global state
    omega_a_check = -3.0  # tonic volatility of local vol state a
    omega_b_check = -3.0  # tonic volatility of local vol state b  
    omega_a = -2.0     # tonic volatility of hidden state a
    omega_b = -2.0     # tonic volatility of hidden state b
    alpha_c_a = 0.05   # coupling strength: x_c -> x_a (value coupling)
    alpha_c_b = 0.05   # coupling strength: x_c -> x_b (value coupling)
    kappa_a = 0.5      # coupling strength: x_a -> x_a (volatility coupling)
    kappa_b = 0.5      # coupling strength: x_b -> x_b (volatility coupling)
    eps_a = 1.0        # observation noise for u_a
    eps_b = 1.0        # observation noise for u_b
    
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
    x_a_check[0] = 6.0  # higher initial local volatility for state a
    x_b_check[0] = 4.0  # lower initial local volatility for state b
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
        x_a_check[k] = x_a_check[k-1] + drift_a_check + np.sqrt(step_size_a_check) * np.random.randn()
        
        drift_b_check = alpha_c_b * x_c[k]
        step_size_b_check = np.exp(omega_b_check)
        x_b_check[k] = x_b_check[k-1] + drift_b_check + np.sqrt(step_size_b_check) * np.random.randn()
        
        # hidden states x_a and x_b evolve with volatility determined by local volatility parents
        step_size_a = np.exp(omega_a + kappa_a * x_a_check[k])
        x_a[k] = x_a[k-1] + np.sqrt(step_size_a) * np.random.randn()
        
        step_size_b = np.exp(omega_b + kappa_b * x_b_check[k])
        x_b[k] = x_b[k-1] + np.sqrt(step_size_b) * np.random.randn()
    
    # observations with noise
    u_a = x_a + np.sqrt(eps_a) * np.random.randn(n_trials)
    u_b = x_b + np.sqrt(eps_b) * np.random.randn(n_trials)
    
    return {
        'x_c': x_c,
        'x_a_check': x_a_check,
        'x_b_check': x_b_check,
        'x_a': x_a,
        'x_b': x_b,
        'u_a': u_a,
        'u_b': u_b,
        'n_trials': n_trials
    }


def create_local_global_volatility_hgf() -> Network:

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
        #parent of input 1 
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
            volatility_coupling_children=(0.5,),  # kappa_a
        )
        # parent of x_b 
        .add_nodes(
            volatility_children=3,
            mean=4.0,  
            precision=1.0,
            tonic_volatility=-3.0,  # omega_b
            volatility_coupling_children=(0.5,),  # kappa_b
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

# try gated approach
def create_gated_local_global_volatility_hgf() -> Network:

    network = (
        Network()
        .add_nodes(
            n_nodes=2,
            precision=1.0,  # Input precision (inverse of observation noise)
        )
        
        .add_nodes(
            value_children=0,
            mean=0.0,
            precision=0.5,
            tonic_volatility=-2.0,  # omega_a
        )
        #parent of input 1 
        .add_nodes(
            value_children=1,
            mean=0.0,
            precision=0.5,
            tonic_volatility=-2.0,  # omega_b
        )
        # local volatility state x_a gated by the global volatility
        .add_nodes(
            kind="continuous-state",
            volatility_children=2,
            mean=6.0,  # higher initial volatility for state a
            precision=1.0,
            tonic_volatility=-3.0,  # omega_a
            volatility_coupling_children=(0.5,),  # kappa_a
        )
        # local volatility state x_b as volatility parent of x_b (node 5)

        .add_nodes(
            kind="continuous-state",
            volatility_children=3,
            mean=4.0, 
            precision=1.0,
            tonic_volatility=-3.0,  # omega_b
            volatility_coupling_children=(0.5,),  # kappa_b
        )
        #  global volatility state x_c as a gating parent
     
        .add_nodes(
            kind="continuous-state",
            mean=0.0,
            precision=1.0,
            tonic_volatility=0.0,  # omega_c
        )
        # gating connections from global to local volatilities
        .add_edges(
            kind="coupling",
            parent_idxs=6,  # Global volatility
            children_idxs=[4, 5],  # Both local volatility states
            coupling_strengths=[0.05, 0.05],  # Gating strength
        )
    )

    return network


def run_inference(network: Network, observations: np.ndarray) -> Network:
   
    network = network.input_data(input_data=observations)
    return network


def plot_results(sim_data: dict, network: Network, save_path: str = None):
   
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    
    n_trials = sim_data['n_trials']
    trial_numbers = np.arange(n_trials)
    
    # get belief trajectories 
    trajectories = network.node_trajectories
    
   
    
    axes[0, 0].plot(trial_numbers, sim_data['x_c'], 'k-', linewidth=2)
    axes[0, 0].set_ylabel('$x_c$ (global volatility)')
    axes[0, 0].set_title('Generative Model\n(Simulated States)')
    axes[0, 0].set_xlim([0, n_trials])
    
 
    axes[1, 0].plot(trial_numbers, sim_data['x_a_check'], 'r-', linewidth=2, label='$x_{\\check{a}}$ (local vol a)')
    axes[1, 0].plot(trial_numbers, sim_data['x_b_check'], 'b-', linewidth=2, label='$x_{\\check{b}}$ (local vol b)')
    axes[1, 0].set_ylabel('Local volatility')
    axes[1, 0].legend(loc='upper right')
    axes[1, 0].set_xlim([0, n_trials])
    
   
    axes[2, 0].scatter(trial_numbers, sim_data['u_a'], alpha=0.3, s=10, c='red', label='$u_a$ (obs)')
    axes[2, 0].scatter(trial_numbers, sim_data['u_b'], alpha=0.3, s=10, c='blue', label='$u_b$ (obs)')
    axes[2, 0].plot(trial_numbers, sim_data['x_a'], 'r-', linewidth=2, label='$x_a$ (state)')
    axes[2, 0].plot(trial_numbers, sim_data['x_b'], 'b-', linewidth=2, label='$x_b$ (state)')
    axes[2, 0].set_xlabel('Trial number')
    axes[2, 0].set_ylabel('Hidden states / Observations')
    axes[2, 0].legend(loc='upper right')
    axes[2, 0].set_xlim([0, n_trials])

    mu_c = trajectories[6]['mean']
    axes[0, 1].plot(trial_numbers, mu_c, 'k-', linewidth=2, label='$\\mu_c$')
    axes[0, 1].set_ylabel('$\\mu_c$ (belief about global vol)')
    axes[0, 1].set_title('Inference Model\n(Beliefs)')
    axes[0, 1].set_xlim([0, n_trials])
    
  
    mu_a_check = trajectories[4]['mean']
    mu_b_check = trajectories[5]['mean']
    axes[1, 1].plot(trial_numbers, mu_a_check, 'r-', linewidth=2, label='$\\mu_{\\check{a}}$')
    axes[1, 1].plot(trial_numbers, mu_b_check, 'b-', linewidth=2, label='$\\mu_{\\check{b}}$')
    axes[1, 1].set_ylabel('Beliefs about local volatility')
    axes[1, 1].legend(loc='upper right')
    axes[1, 1].set_xlim([0, n_trials])

    mu_a = trajectories[2]['mean']
    mu_b = trajectories[3]['mean']
    axes[2, 1].scatter(trial_numbers, sim_data['u_a'], alpha=0.3, s=10, c='red', label='$u_a$ (obs)')
    axes[2, 1].scatter(trial_numbers, sim_data['u_b'], alpha=0.3, s=10, c='blue', label='$u_b$ (obs)')
    axes[2, 1].plot(trial_numbers, mu_a, 'r-', linewidth=2, label='$\\mu_a$')
    axes[2, 1].plot(trial_numbers, mu_b, 'b-', linewidth=2, label='$\\mu_b$')
    axes[2, 1].set_xlabel('Trial number')
    axes[2, 1].set_ylabel('Beliefs about hidden states')
    axes[2, 1].legend(loc='upper right')
    axes[2, 1].set_xlim([0, n_trials])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()
    return fig


def main():
  
  
    print("=" * 60)
    print("gHGF Figure 6: Local vs Global Volatility")
    print("Standard Implementation using pyhgf")
    print("=" * 60)
    
    # simulate the generative model
    print("\n[1] Simulating generative model...")
    sim_data = simulate_generative_model(n_trials=175, seed=42)
    print(f"    Generated {sim_data['n_trials']} trials of observations")
    
    # create the HGF network
    print("\n[2] Creating HGF network structure...")
    network = create_local_global_volatility_hgf()
    print(f"    Network has {network.n_nodes} nodes")
    
    # visualize structure
    print("\n[3] Network structure:")
    network.plot_network()
    
    # prep observations
    observations = np.column_stack([sim_data['u_a'], sim_data['u_b']])
    print(f"\n[4] Running inference on {observations.shape[0]} observations...")
    
    # Run 
    network = run_inference(network, observations)
    print("    Inference complete!")
    
    # plotting  
    print("\n[5] Plotting results...")
    fig = plot_results(sim_data, network, save_path='outputs/ghgf_figure6_standard.png')
    
    # summary stats
    print("\n[6] Summary:")
    trajectories = network.node_trajectories
    print(f"    Global volatility belief (node 6) final mean: {trajectories[6]['mean'][-1]:.3f}")
    print(f"    Local vol a belief (node 4) final mean: {trajectories[4]['mean'][-1]:.3f}")
    print(f"    Local vol b belief (node 5) final mean: {trajectories[5]['mean'][-1]:.3f}")
    
    return network, sim_data


if __name__ == "__main__":
    network, sim_data = main()
