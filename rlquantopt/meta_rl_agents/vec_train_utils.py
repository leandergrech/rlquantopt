import multiprocessing
from sb3_contrib import TRPO
from stable_baselines3.common.env_util import make_vec_env
from rlquantopt.rl_envs.zc_qpee import ZCQPEE
from rlquantopt.meta_rl_agents.train_utils import copy_params


def _train_agent_with_id(agent_id, policy_params, env, algo, algo_kw, steps, inner_loop_tb_dir=None, n_envs=1):
    """
    Train a single agent on a specific environment.

    :param agent_id: ID for the agent (for process identification).
    :param policy_params: Initial parameters for the policy and value networks.
    :param env: gymnasium.Env environment instance.
    :param algo: The RL algorithm to use (e.g., TRPO).
    :param algo_kw: Additional keyword arguments for the algorithm.
    :param steps: Number of training steps for the agent.
    :param n_envs: Number of parallel environments to use for single agent training.
    :return: Updated parameters after task-specific training.
    """

    env_kw = env.env_kwargs
    model_params = env.model_params
    def env_factory():
        return ZCQPEE(model_params=model_params, **env_kw)
    env = make_vec_env(env_factory, n_envs=n_envs)

    model = algo("MlpPolicy", env, tensorboard_log=inner_loop_tb_dir, **algo_kw)

    policy_params = copy_params(policy_params)
    model.policy.load_state_dict(policy_params['pi'])
    model.policy.value_net.load_state_dict(policy_params['vf'])

    real_steps = steps * n_envs * algo_kw.get('n_steps', 1)
    model.learn(total_timesteps=real_steps, log_interval=1, progress_bar=True, tb_log_name=f'agent_{agent_id}')

    updated_policy_params = {
        'pi': model.policy.state_dict(),
        'vf': model.policy.value_net.state_dict()
    }

    # Return updated parameters
    return agent_id, updated_policy_params


def vec_train_agents(policy_params, envs, algo, algo_kw, inner_loop_tb_dir=None, steps=10, n_envs=1):
    """
    Train multiple RL agents in parallel using multiprocessing.

    :param policy_params: Initial parameters for the policy and value networks.
    :param envs: List of environments on which to train agents respectively and in parallel.
    :param algo: The RL algorithm to use (e.g., TRPO).
    :param algo_kw: Additional keyword arguments for the algorithm.
    :param steps: Number of steps for each agent to train.
    :param n_envs: Number of parallel environments to use for single agent training.
    :return: Dictionary containing updated policy parameters from all agents.
    """
    if algo_kw is None:
        algo_kw = {}

    n_agents = len(envs)

    # Prepare arguments for all agents (ensure it's a tuple of all parameters)
    tasks = [(agent_id, policy_params, envs[agent_id], algo, algo_kw, steps, inner_loop_tb_dir, n_envs) for agent_id in range(n_agents)]

    # Use pool.starmap for parallelism
    with multiprocessing.Pool(processes=n_agents) as pool:
        results = pool.starmap(_train_agent_with_id, tasks)

    # Collect results
    updated_params = {agent_id: params for agent_id, params in results}

    return updated_params

