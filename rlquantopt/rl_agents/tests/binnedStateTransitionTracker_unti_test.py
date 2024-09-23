import unittest
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from rlquantopt.rl_agents.wrappers import BinnedStateTransitionTracker
from rlquantopt.rl_envs.zc_qpee import ZCQPEE
from rlquantopt.rl_envs.zcqubits import setup_ZCQubits4MKrauss_params


# class MockEnv(gym.Env):
#     def __init__(self):
#         super(MockEnv, self).__init__()
#         # Define the observation space as normalized between -1 and 1, 3 dimensions
#         self.observation_space = Box(low=-1, high=1, shape=(3,), dtype=np.float32)
#         self.action_space = Box(low=-1, high=1, shape=(1,), dtype=np.float32)
#         self.state = np.array([0.0, 0.5, -0.5])  # Example initial state
#
#     def reset(self):
#         # Reset the environment to an initial state
#         self.state = np.array([0.0, 0.5, -0.5])
#         return self.state
#
#     def step(self, action):
#         # Simple mock step function for testing
#         next_state = self.state + action  # Apply the action
#         next_state = np.clip(next_state, -1, 1)  # Keep it within the bounds of [-1, 1]
#         reward = 1.0  # Mock reward
#         done = False  # Mock done flag
#         return next_state, reward, done, {}


class TestBinnedStateTransitionTracker(unittest.TestCase):

    def setUp(self):
        # Setup the environment and the wrapper
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
        self.env = ZCQPEE(model_params=model_params, **env_kw)
        self.env.reset()
        self.wrapped_env = BinnedStateTransitionTracker(self.env, bins=10)

    def test_initialization(self):
        # Test if the environment is initialized properly with the wrapper
        self.assertEqual(self.wrapped_env.bins, 10, "Number of bins should be 10.")
        self.assertEqual(len(self.wrapped_env.bin_edges), 11, "There should be 11 bin edges for 10 bins.")

    def test_step_binning(self):
        # Test if the binning of states works as expected
        action = self.wrapped_env.action_space.sample()
        next_state, reward, terminated, truncated, info = self.wrapped_env.step(action)

        # Check if the next_state is binned correctly and recorded
        binned_state = self.wrapped_env._get_binned_obs(self.env.current_obs)
        print(binned_state)
        # self.assertTrue((binned_state >= 0).all() and (binned_state <= 9).all(),
        #                 "Binned states should be between 0 and 9.")

        # Ensure that the transition counts have been updated
        initial_binned_state = self.wrapped_env._get_binned_obs(next_state)
        print('self.wrapped_env.transition_counts')
        print(self.wrapped_env.transition_counts)
        exit(23)
        self.assertIn(tuple(initial_binned_state), self.wrapped_env.transition_counts,
                      "Initial binned state should be recorded.")
        # self.assertIn(tuple(binned_state), self.wrapped_env.transition_counts[tuple(initial_binned_state)],
        #               "Next binned state should be recorded in transition.")

    def test_reset(self):
        # Test the reset functionality of the environment
        initial_state = self.wrapped_env.reset()[0]
        print(initial_state)
        print(self.wrapped_env.current_obs)
        self.assertTrue(np.array_equal(initial_state, self.wrapped_env.current_obs),
                        "The reset state should match the initial current state.")

    def test_transition_distribution(self):
        # Test that the transition distribution is returned correctly
        action = self.wrapped_env.action_space.sample()
        self.wrapped_env.step(action)  # Take one step
        self.wrapped_env.step(action)  # Take another step

        # Retrieve the transition distribution
        distribution = self.wrapped_env.get_transition_distribution()
        self.assertTrue(isinstance(distribution, dict), "Transition distribution should be a dictionary.")
        self.assertGreater(len(distribution), 0, "Transition distribution should contain entries after steps.")
        print(distribution)


# if __name__ == '__main__':
    # unittest.main()
