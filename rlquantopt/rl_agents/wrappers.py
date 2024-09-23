import gymnasium as gym
import numpy as np
from collections import defaultdict
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from tqdm import trange

from rlquantopt.rl_envs.zc_qpee import ZCQPEE
from rlquantopt.rl_envs.zcqubits import setup_ZCQubits4MKrauss_params

# Suppress warnings
warnings.filterwarnings("ignore")

# Set global font size to 8 using mpl.rcParams
mpl.rcParams.update({'font.size': 8})

class BinnedStateTransitionTracker(gym.Wrapper):
    def __init__(self, env: gym.Env, bins=10, edge=1.):
        super(BinnedStateTransitionTracker, self).__init__(env)
        self.bins = bins
        # Create bins between -1 and 1 for each dimension in the observation space
        self.bin_edges = np.linspace(-edge, edge, self.bins + 1)
        # Dictionary to store binned transition counts
        self.transition_counts = defaultdict(int)
        self.all_real_obs = []  # Collect all real state values for plotting

    def step(self, action):
        current_state = self._get_obs()  # Use real state values
        next_state, reward, terminated, truncated, info = self.env.step(action)

        # Record the real values and collect for plotting
        self.all_real_obs.append(current_state)

        return next_state, reward, terminated, truncated, info

    def _get_obs(self):
        # Extracts the current state observation (real values)
        return self.env.current_obs

    def _get_binned_obs(self, state):
        # Bins each dimension of the state into the predefined bin edges
        binned_obs = np.digitize(state, self.bin_edges) - 1  # Subtract 1 to make the bin index start from 0
        return binned_obs

    def reset(self, **kwargs):
        # Reset the environment and return the initial observation
        return self.env.reset(**kwargs)

    def get_transition_distribution(self):
        # Returns the transition counts between binned states
        return dict(self.transition_counts)

    def get_all_real_obs(self):
        # Returns all the real observations for plotting
        return np.array(self.all_real_obs)

    def reset_all_real_obs(self):
        self.all_real_obs = []

    def plot_obs_distributions(self, save_path, show=False):
        # Plotting real state values (not binned) on a single axes with each dimension shifted along the x-axis
        all_real_obs = self.get_all_real_obs()
        n_samples = all_real_obs.shape[0]
        num_state_dims = all_real_obs.shape[1]  # Get the number of state dimensions
        fig, ax = plt.subplots(figsize=(15, 6))

        # Plot each state dimension, shifted along the x-axis
        for i in range(num_state_dims):
            sns.violinplot(x=np.full(all_real_obs.shape[0], i), y=all_real_obs[:, i], ax=ax, inner="quart", linewidth=0.6)

        # Set x-axis to label the dimensions
        ax.set_xticks(np.arange(num_state_dims))
        ax.set_xticklabels([f'o[{i}]' for i in range(num_state_dims)])

        # Set axis labels and title
        ax.set_title(f'Distribution of Each State Dimension ({n_samples} samples)')
        ax.set_ylabel('Obs value')
        ax.set_xlabel('State Dimensions')

        # Reduce font size to 8 globally
        mpl.rcParams.update({'font.size': 8})

        plt.tight_layout()
        fig.savefig(save_path)
        if show: plt.show()
        self.reset_all_real_obs()

    def __str__(self):
        return super().__str__()


if __name__ == '__main__':
    # Setup the environment
    model_params = setup_ZCQubits4MKrauss_params(None)
    env_kw = dict(pulse_length=2000,
                  rew_scale=0.1,
                  delta_mode=True,
                  T=300.,
                  fid_thresh=0.99,
                  action_scaling={'z': 1.},
                  a_norm_max=5.,
                  n_time_steps=3,
                  tv_penalty_scale=1e-3)
    env = ZCQPEE(model_params=model_params, **env_kw)
    env = BinnedStateTransitionTracker(env, bins=25, edge=2)


    nb_eps = 1000  # Number of episodes
    n_act = env.n_act  # Assuming the action space is provided by the environment
    rews = []
    start_target_prob = 0.5
    for ep in trange(nb_eps):
        if np.random.random() <= start_target_prob:
            env.reset(init_state=env.target_states)
        else:
            # Reset the environment1
            env.reset()
        terminal = False
        while not terminal:
            action = env.action_space.sample() * 0.1  # Sampling random actions
            otp1, rew, terminated, truncated, info = env.step(action)
            rews.append(rew)
            terminal = info['terminal_state']

    # Collect all real observations
    all_real_obs = env.get_all_real_obs()
    print(f'Min obs. values = {np.min(all_real_obs, axis=0)}')
    print(f'Max obs. values = {np.max(all_real_obs, axis=0)}')
    print(f'Min rew = {min(rews)}')
    print(f'Max rew = {max(rews)}')
    print(f'Nb. samples = {all_real_obs.shape[0]}')

    save_name = f"{str(env)}_{nb_eps}eps"
    np.save(f'results/{save_name}_all_real_obs.npy', all_real_obs)
    env.plot_obs_distributions(save_path=f'results/{save_name}state_dists.pdf', show=True)

