import os
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import pandas as pd

from stable_baselines3.common import type_aliases
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped


def linear_schedule(init_value):
    def func(x):
        return init_value * x
    return func


def harmonic_schedule(init_value=3e-4, k=10**0.4, tau=1):
    def func(x):
        return init_value / (1. + (k * (1 - x*tau)))
    return func


def gen_env_yaml(env_type, pulse_path, save_dir, ret_pulse=True, override_env_kw=None):
    tlist, pulse = get_pulse_data(pulse_path)

    env_kw = dict(delta_mode=True,
                  T=max(tlist),
                  pulse_length=len(pulse),
                  optimised_pulse_path=os.path.abspath(pulse_path))
    if override_env_kw is not None:
        env_kw.update(override_env_kw)

    env = env_type(**env_kw)
    save_path = os.path.join(save_dir, f'{str(env)}.yml')
    env.to_yaml(save_path)

    if ret_pulse:
        return save_path, dict(tlist=tlist, pulse=pulse)
    else:
        return save_path





def get_pulse_data(pulse_path, verbose=False):
    data = pd.read_csv(pulse_path)
    col_keys = list(data.keys())
    if len(col_keys) < 2:
        data = pd.read_csv(pulse_path)
        col_keys = list(data.keys())
    tkey, vkey = None, None
    print(col_keys)
    for key in col_keys:
        if 'amplist' in key or 'pulse' in key or 'amp' in key or 'value' in key:
            vkey = key
            continue
        elif 'tlist' in key or 'time' in key or 't' in key:
            tkey = key
            continue
    if tkey is None or vkey is None:
        raise Exception("There's something f***ed with the CSV pulse file.")

    tlist = data[tkey].to_numpy()
    amplist = data[vkey].to_numpy()
    if verbose:
        print(f'Pulse time: {tlist[0]:.4f}ns -> {tlist[-1]:.4f}ns; ΔT: {tlist[1]:.4f}ns')
        print(f'Pulse amps: min={min(amplist)}, max={max(amplist)}; PL: {len(amplist)}')

    return tlist, amplist


'''
DEPRECATED
'''
def evaluate_policy(
    model: "type_aliases.PolicyPredictor",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
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

    :param model: The RL agent you want to evaluate. This can be any object
        that implements a `predict` method, such as an RL algorithm (``BaseAlgorithm``)
        or policy (``BasePolicy``).
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    all_rewards_list = [[] for _ in range(n_eval_episodes)]
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    while (episode_counts < episode_count_targets).any():
        actions, states = model.predict(
            observations,  # type: ignore[arg-type]
            state=states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        new_observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals(), globals())

                all_rewards_list[i].append(rewards[i])

                if dones[i]:
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0

        observations = new_observations

        if render:
            env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths, all_rewards_list

    return mean_reward, std_reward, all_rewards_list


if __name__ == '__main__':
 from stable_baselines3.ppo import PPO
 model_path = '/home/leander/code/rlquantopt/rlquantopt/rl_agents/VQPEE-PPO/16-05-24_164915_32-envs/rl_model_501952_steps.zip'
 model = PPO.load(model_path)
 from rlquantopt.rl_envs.old.vec_qu_pulse_episodic_env import VecQuPulseEpisodicEnv as VQPEE
 env = VQPEE(num_envs=5, pulse_length=50, sparse_reward=False, use_full_liouville=False, inc_off_diag=False)
 print(evaluate_policy(model, env, n_eval_episodes=5, deterministic=True))