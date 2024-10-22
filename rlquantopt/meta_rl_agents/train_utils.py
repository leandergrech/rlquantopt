import os
import copy
import torch
from stable_baselines3.common.env_util import make_vec_env

from rlquantopt.rl_envs.zc_qpee import ZCQPEE


def copy_params(params):
    return {'pi': copy.deepcopy(params['pi']), 'vf': copy.deepcopy(params['vf'])}


def get_initial_trpo_params(env, algo, algo_kw=None):
    """
    Initialize the RL algorithm with the given hyperparameters and return the initial model parameters,
    including both policy and value function (if applicable).

    :param algo: The RL algorithm class (e.g., TRPO, PPO, DDPG).
    :param env: The environment where the agent will interact.
    :param algo_kw: A dictionary of hyperparameters for initializing the algorithm.
    :return: A dictionary containing the initial parameters for all networks (policy, value, etc.).
    """
    if algo_kw is None:
        algo_kw = {}

    # Initialize the RL algorithm
    model = algo("MlpPolicy", env, **algo_kw)

    # Dictionary to store all relevant parameters
    all_params = dict()

    # Store policy (actor) parameters
    all_params['pi'] = model.policy.state_dict()

    # Store value (critic) network parameters
    all_params['vf'] = model.policy.value_net.state_dict()

    return all_params


def train_agent(policy_params, env, algo, algo_kw, steps, n_envs=8):
    """
    Train the agent on a task-specific environment and update the policy and value function parameters.

    :param policy_params: Initial parameters for the policy and value networks.
    :param env: The task-specific environment.
    :param hyparams: Hyperparameters for training.
    :param steps: Number of steps to train the agent.
    :return: Updated parameters after task-specific training.
    """
    env_kw = env.env_kwargs
    model_params = env.model_params
    env = make_vec_env(lambda: ZCQPEE(model_params=model_params, **env_kw), n_envs=n_envs)

    model = algo("MlpPolicy", env, **algo_kw)

    policy_params = copy_params(policy_params)
    model.policy.load_state_dict(policy_params['pi'])
    model.policy.value_net.load_state_dict(policy_params['vf'])

    real_steps = steps * n_envs * algo_kw.get('n_steps', 1)
    model.learn(total_timesteps=real_steps, log_interval=1)

    updated_policy_params = {
        'pi': model.policy.state_dict(),
        'vf': model.policy.value_net.state_dict()
    }
    return updated_policy_params


def save_checkpoint(params, save_name):
    torch.save(params['pi'], os.path.join(save_name, 'pi.pth'))
    torch.save(params['vf'], os.path.join(save_name, 'vf.pth'))


def load_checkpoint(load_dir):
    params = dict()
    params['pi'] = torch.load(os.path.join(load_dir, 'pi.pth'))
    params['vf'] = torch.load(os.path.join(load_dir, 'vf.pth'))

    return params
