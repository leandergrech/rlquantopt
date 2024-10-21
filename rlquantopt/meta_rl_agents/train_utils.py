import time
import torch
from torch.utils.tensorboard import SummaryWriter
from sb3_contrib import TRPO
from stable_baselines3.common.envs import DummyVecEnv


def get_initial_trpo_params(env, trpo_kw=None):
    """
    Initialize the RL algorithm with the given hyperparameters and return the initial model parameters,
    including both policy and value function (if applicable).

    :param algo: The RL algorithm class (e.g., TRPO, PPO, DDPG).
    :param env: The environment where the agent will interact.
    :param trpo_kw: A dictionary of hyperparameters for initializing the algorithm.
    :return: A dictionary containing the initial parameters for all networks (policy, value, etc.).
    """
    if trpo_kw is None:
        trpo_kw = {}

    # Initialize the RL algorithm
    model = TRPO("MlpPolicy", env, **trpo_kw)

    # Dictionary to store all relevant parameters
    all_params = {}

    # Store policy (actor) parameters
    all_params['policy_params'] = model.policy.state_dict()

    # Store value (critic) network parameters if they exist
    if hasattr(model.policy, 'value_net'):
        all_params['value_params'] = model.policy.value_net.state_dict()

    return all_params


def train_agent(policy_params, env, trpo_kw, steps):
    """
    Train the agent on a task-specific environment and update the policy and value function parameters.

    :param policy_params: Initial parameters for the policy and value networks.
    :param env: The task-specific environment.
    :param hyparams: Hyperparameters for training.
    :param steps: Number of steps to train the agent.
    :return: Updated parameters after task-specific training.
    """
    env = DummyVecEnv([lambda: env])

    model = TRPO("MlpPolicy", env, **trpo_kw)

    model.policy.load_state_dict(policy_params['pi'])
    model.policy.value_net.load_state_dict(policy_params['vf'])

    model.learn(total_timesteps=steps)

    updated_policy_params = {
        'pi': model.policy.state_dict(),
        'vf': model.policy.value_net.state_dict()
    }
    return updated_policy_params


def validate_meta_policy(meta_policy_params,
                         validation_envs,
                         trpo_kw,
                         writer: SummaryWriter,
                         fine_tune_steps=100,
                         log_interval=10):
    validation_results = []

    for i, env in enumerate(validation_envs):
        task_policy_params = copy.deepcopy(meta_policy_params)
        start_time = time.time()

        # Fine-tune the policy on the validation task
        task_policy_params = train_agent(task_policy_params, env, trpo_kw, fine_tune_steps)

        # Measure fine-tuning time
        fine_tuning_time = time.time() - start_time

        # Evaluate on the validation task (you could use task-specific evaluation metrics here)
        evaluation_metric = evaluate_policy(task_policy_params, env)

        validation_results.append((evaluation_metric, fine_tuning_time))

        # Log performance and fine-tuning time to TensorBoard
        writer.add_scalar(f"eval/Task_{i}_Performance", evaluation_metric, log_interval)
        writer.add_scalar(f"eval/Task_{i}_FineTuningTime", fine_tuning_time, log_interval)

    return validation_results


def save_checkpoint(params, save_name):
    torch.save(params['pi'], os.path.join(save_name, 'pi.pth'))
    torch.save(params['vf'], os.path.join(save_name, 'vf.pth'))


def load_checkpoint(load_dir):
    params = dict()
    params['pi'] = torch.load(os.path.join(load_dir, 'pi.pth'))
    params['vf'] = torch.load(os.path.join(load_dir, 'vf.pth'))

    return params
