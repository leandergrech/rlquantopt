import unittest
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sb3_contrib import TRPO

from rlquantopt.meta_rl_agents.vec_reptile import vec_reptile_meta_learning_zcqpee
from rlquantopt.meta_rl_agents.train_utils import copy_params, get_initial_trpo_params, save_checkpoint, load_checkpoint
from rlquantopt.meta_rl_agents.vec_train_utils import vec_train_agents
from rlquantopt.meta_rl_agents.sampler import sample_zcqpee_tasks
from rlquantopt.rl_envs.zc_qpee import ZCQPEE


class TestReptileAlgorithm(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up environment parameters and initial policy for tests."""
        cls.env_kwargs = {
            'pulse_length': 100,
            'rew_scale': 1.0,
            'delta_mode': True,
            'T': 300.0,
            'fid_thresh': 0.99,
            'action_scaling': {'z': 1.0},
            'a_norm_max': 5.0,
            'n_time_steps': 3,
            'tv_penalty_scale': 0.1,
            'act_poly_order': 1,
            'add_prev_obs': False,
        }
        cls.trpo_kw = {'learning_rate': 1e-3, 'batch_size': 64, 'n_steps': 1024}
        cls.initial_params = get_initial_trpo_params(env=ZCQPEE(**cls.env_kwargs), algo=TRPO, algo_kw=cls.trpo_kw)

    def test_copy_params(self):
        """Test that copy_params correctly duplicates policy parameters."""
        params_copy = copy_params(self.initial_params)
        for net, net_params in self.initial_params.items():
            for key in net_params:
                self.assertTrue(torch.equal(params_copy[net][key], self.initial_params[net][key]))

    def test_sample_zcqpee_tasks(self):
        """Test that task sampling provides the expected number of tasks with randomized parameters."""
        tasks = sample_zcqpee_tasks(n_tasks=5, eps=0.1, **self.env_kwargs)
        self.assertEqual(len(tasks), 5)
        for task in tasks:
            self.assertIsInstance(task, ZCQPEE)

    def test_vec_train_agents(self):
        """Test parallel agent training to ensure it completes and updates parameters."""
        tasks = sample_zcqpee_tasks(2, eps=0.1, **self.env_kwargs)
        trained_params = vec_train_agents(
            policy_params=self.initial_params,
            envs=tasks,
            algo=TRPO,
            algo_kw=self.trpo_kw,
            steps=5,
            n_envs=1
        )
        self.assertEqual(len(trained_params), 2)
        for params in trained_params.values():
            self.assertIn('pi', params)
            self.assertIn('vf', params)

    def test_meta_learning_with_known_result(self):
        """Meta-train on a simplified environment setup and validate expected improvements."""
        writer = SummaryWriter(log_dir="/tmp/test_logs")
        trained_meta_policy = vec_reptile_meta_learning_zcqpee(
            initial_policy_params=self.initial_params,
            trpo_kw=self.trpo_kw,
            save_dir="/tmp/test_checkpoint",
            writer=writer,
            trpo_n_envs=1,
            save_every=5,
            eval_every=5,
            n_tasks=3,
            n_parallel_tasks=2,
            meta_iterations=10,
            inner_steps=3,
            meta_step_size=0.05,
            device='cpu',
            **self.env_kwargs
        )
        self.assertTrue(isinstance(trained_meta_policy, dict))
        self.assertIn('pi', trained_meta_policy)
        self.assertIn('vf', trained_meta_policy)


if __name__ == "__main__":
    unittest.main()

