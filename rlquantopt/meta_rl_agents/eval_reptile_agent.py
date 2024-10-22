import numpy as np
import copy
import time
from torch.utils.tensorboard import SummaryWriter
from gymnasium import Env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

from rlquantopt.rl_envs.zc_qpee import ZCQPEE
from rlquantopt.meta_rl_agents.train_utils import get_initial_trpo_params, train_agent


def evaluate_reptile_zcqpee(
    meta_policy_params,
    algo,
    algo_kw,
    eval_envs,
    deterministic: bool = True):
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param meta_policy_params: Policy parameters to be evaluated.
    :param algo: The RL algorithm class (e.g., TRPO).
    :param algo_kw: Additional keyword arguments for the algorithm.
    :param eval_envs: List of validation environments
    :param deterministic: Whether to use deterministic or stochastic actions
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """

    class ZCQPEEFactory():
        def __init__(self, val_envs):
            self.env_kws = [env.env_kwargs for env in val_envs]
            self.models_params = [env.model_params for env in val_envs]
            self.n_envs = len(val_envs)
            self.idx = 0

        def __call__(self, *args, **kwargs):
            idx = self.idx
            model_params = self.models_params[idx]
            env_kw = self.env_kws[idx]
            env = ZCQPEE(model_params=model_params, **env_kw)
            self.idx = (idx + 1) % n_envs
            return env

    if isinstance(eval_envs, list):
        n_envs = len(eval_envs)
    elif isinstance(eval_envs, Env):
        eval_envs = [eval_envs]
        n_envs = 1

    zcqpee_factory = ZCQPEEFactory(eval_envs)
    env = make_vec_env(zcqpee_factory, n_envs=n_envs)

    # Initialize the RL algorithm with the vectorized environment
    model = algo("MlpPolicy", env, **algo_kw)

    # Load the policy parameters
    model.policy.load_state_dict(meta_policy_params['pi'])
    model.policy.value_net.load_state_dict(meta_policy_params['vf'])

    episode_rewards = [[] for _ in range(n_envs)]

    observations = env.reset()
    states = None
    dones = np.repeat(False, n_envs)
    while not dones.any():
        actions, states = model.predict(observations,
                                        state=states,
                                        deterministic=deterministic)

        new_observations, rewards, dones, infos = env.step(actions)
        for i in range(n_envs):
            episode_rewards[i].append(rewards[i])

        observations = new_observations


    episode_rewards = np.array(episode_rewards)
    best_rewards = [max(ep_rews) for ep_rews in episode_rewards],
    worst_rewards = [min(ep_rews) for ep_rews in episode_rewards],
    mean_best_rewards = np.mean(best_rewards)
    mean_init_rewards = np.mean([ep_rews[0] for ep_rews in episode_rewards])

    data = dict(mean_reward = np.mean(episode_rewards),
                std_reward = np.std(episode_rewards),
                mean_init_rewards = mean_init_rewards,
                mean_best_rewards = mean_best_rewards,
                mean_worst_rewards = np.mean(worst_rewards),
                mean_ep_reward_improvement = mean_best_rewards - mean_init_rewards,
                mean_ep_len_best = np.mean([np.argmax(ep_rews) for ep_rews in episode_rewards]))

    return data


def log_metrics(data, step, writer: SummaryWriter, prefix=None):
    if prefix is None:
        prefix = ''
    else:
        prefix += '/'

    for k, v in data.items():
        writer.add_scalar(f'{prefix}{k}', v, global_step=step)


def validate_meta_policy(meta_policy_params,
                         algo,
                         algo_kw,
                         eval_env,
                         fine_tuning_steps=20,
                         fine_tuning_n_envs=1,
                         verbose=False):
    if verbose: print(f'\t\t\t`-> Evaluating initial parameters')
    eval_data_before = evaluate_reptile_zcqpee(meta_policy_params, algo, algo_kw, eval_env, True)
    if verbose: print(f'\t\t\t`-> Fine-tuning meta policy parameters')
    meta_policy_params = train_agent(policy_params=meta_policy_params, algo=algo, algo_kw=algo_kw,
                env=eval_env, steps=fine_tuning_steps, n_envs=fine_tuning_n_envs)
    if verbose:
        print(f'\t\t\t`-> Evaluating initial parameters')
    eval_data_after = evaluate_reptile_zcqpee(meta_policy_params, algo, algo_kw, eval_env, True)
    return eval_data_before, eval_data_after

