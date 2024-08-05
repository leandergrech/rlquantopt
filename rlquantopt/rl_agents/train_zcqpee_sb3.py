import os
import argparse
import warnings
from datetime import datetime as dt
import json
from typing import Any, Dict, List, Optional, Union
import gymnasium as gym
import numpy as np
import torch as tc
from datetime import datetime as dt
from sb3_contrib import RecurrentPPO, TRPO
from stable_baselines3.common.vec_env import VecEnv, sync_envs_normalization
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EventCallback, BaseCallback, StopTrainingOnNoModelImprovement
from stable_baselines3 import PPO
from rlquantopt.rl_envs.zc_qpee import ZCQPEE
from rlquantopt.utils.rl_utils import evaluate_policy

import warnings
warnings.filterwarnings("ignore")


class EvalCallback(EventCallback):
    """
    Callback for evaluating an agent.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``

    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param callback_after_eval: Callback to trigger after every evaluation
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose: Verbosity level: 0 for no output, 1 for indicating information about evaluation results
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    """

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
        algo = 'PPO'
    ):
        self.start_time = dt.now()
        super().__init__(callback_after_eval, verbose=verbose)
        self.algo = algo
        self.callback_on_new_best = callback_on_new_best
        if self.callback_on_new_best is not None:
            # Give access to the parent
            self.callback_on_new_best.parent = self

        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.warn = warn

        # Convert to VecEnv for consistency
        # if not isinstance(eval_env, VecEnv):
        #     eval_env = DummyVecEnv([lambda: eval_env])  # type: ignore[list-item, return-value]

        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        # Logs will be written in ``evaluations.npz``
        if log_path is not None:
            log_path = os.path.join(log_path, "evaluations")
        self.log_path = log_path
        self.evaluations_rewards: List[List[float]] = []
        self.evaluations_timesteps: List[int] = []
        self.evaluations_length: List[List[int]] = []
        # For computing success rate
        self._is_success_buffer: List[bool] = []
        self.evaluations_successes: List[List[bool]] = []

    def _init_callback(self) -> None:
        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn("Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        # Init callback called on new best model
        if self.callback_on_new_best is not None:
            self.callback_on_new_best.init_callback(self.model)

    def _log_success_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.

        :param locals_:
        :param globals_:
        """
        info = locals_["info"]

        if locals_["done"]:
            maybe_is_success = info.get("is_success")
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)

    def eval_policy(self):
        rews = []
        for _ in range(self.n_eval_episodes):
            rews.append([])
            obs, _ = self.eval_env.reset()
            done = False
            lstm_states = None
            episode_start = np.ones((1,), dtype=bool)
            while not done:
                if self.algo == 'PPO' or self.algo == 'TRPO':
                    actions = self.model.predict(obs, deterministic=True)[0]
                elif self.algo == 'RecurrentPPO':
                    actions, lstm_states = self.model.predict(obs, state=lstm_states, episode_start=episode_start, deterministic=True)

                obs, r, te, tr, info = self.eval_env.step(actions)
                rews[-1].append(r)
                done = te or tr
        return rews

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.num_timesteps % self.eval_freq == 0:
            time_elapsed = dt.now() - self.start_time
            fps = self.num_timesteps / time_elapsed.total_seconds()

            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards = self.eval_policy()
            episode_lengths = [len(item) for item in episode_rewards]
            self._is_success_buffer = [ep_len < self.eval_env.pulse_length for ep_len in episode_lengths]

            if self.log_path is not None:
                assert isinstance(episode_rewards, list)
                assert isinstance(episode_lengths, list)
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_rewards.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if sum(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            mean_init_rewards = np.mean([ep_rews[0] for ep_rews in episode_rewards])
            best_rewards = [max(ep_rews) for ep_rews in episode_rewards]
            best_fidelities = [self.eval_env.rew2fid(r + self.eval_env.REW_THRESH) for r in best_rewards]

            mean_best_rewards = np.mean(best_rewards)
            mean_best_fidelities = np.mean(best_fidelities)
            mean_ep_reward_improvement = mean_best_rewards - mean_init_rewards
            mean_ep_len_best = np.mean([np.argmax(ep_rews) for ep_rews in episode_rewards])
            mean_best_T_ns = self.eval_env.dt * mean_ep_len_best
            mean_success = np.mean(np.array(self._is_success_buffer).astype(int))

            if self.verbose >= 1:
                print(
                    f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", float(mean_ep_length))
            self.logger.record("eval/std_ep_length", float(std_ep_length))
            self.logger.record('eval/mean_ep_reward_improvement', float(mean_ep_reward_improvement))
            self.logger.record('eval/mean_best_rewards', float(mean_best_rewards))
            self.logger.record('eval/mean_best_fidelities', float(mean_best_fidelities))
            self.logger.record('eval/mean_best_infidelities', float(1-mean_best_fidelities))
            self.logger.record('eval/mean_best_T_ns', float(mean_best_T_ns))
            self.logger.record('eval/mean_ep_len_best', float(mean_ep_len_best))
            self.logger.record('eval/success', float(mean_success))

            if sum(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps)
            self.logger.record("time/fps", fps)
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = float(mean_reward)
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        if self.callback:
            self.callback.update_locals(locals_)


def parse_args():
    parser = argparse.ArgumentParser('RLQuantOpt - training RL agent on ZCQPEE environment')
    # ZCQPEE parameters
    parser.add_argument('-p', '--pulse-length', default=120, type=int, help='Maximum number of samples in a pulse')
    parser.add_argument('-d', '--delta-mode', action='store_true', help='Use ZCQPEE environment in delta action mode')
    parser.add_argument('-T', '--max-time-ns', default=200.0, type=float, help='Pulse duration in ns')
    parser.add_argument('--a-scale', default=0.1, type=float, help='Set action scaling during normalisation')
    parser.add_argument('--a-norm-max', default=10.0, type=float, help='Set the maximum normalised amplitude when using delta-mode')
    parser.add_argument('--fid-thresh', default=0.995, type=float, help='Set goal fidelity threshold')

    # RL agent paramters
    parser.add_argument('--algo', type=str, default='PPO', help='Type of RL agent')
    parser.add_argument('--n-steps', type=int, default=10--00,
                        help='The number of steps to run for each environment per update'
                             '(i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)'
                             'NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)'
                             'See https://github.com/pytorch/pytorch/issues/29372')
    parser.add_argument('--n-epochs', type=int, default=10, help='Number of epoch when optimizing the surrogate loss')
    parser.add_argument('--ent-coef', type=float, default=0.0, help='Entropy coefficient for the loss calculation')
    parser.add_argument('--batch-size', type=int, default=64, help='Mini-batch size')
    parser.add_argument('--gamma', type=float, default=0.99, help='Mini-batch size')
    parser.add_argument('--seed', default=123, type=int, help='Set random seed')

    # Fine tuning an existing model
    parser.add_argument('-r', '--retrain-model-zip', type=str, default=None, help='By passing the path the model zipfile, you are instructing to continue training with these new parameters')


    # Training parameters
    parser.add_argument('--n-envs', default=8, type=int, help='Number of vectorised training environments')
    parser.add_argument('--n-train', default=250000, type=int, help='Number of training steps')
    parser.add_argument('--save-freq', default=10000, type=int, help='Save model every save_freq calls to env.step')
    parser.add_argument('--eval-freq', default=500, type=int, help='Evaluate model every eval_freq calls to env.step')
    parser.add_argument('--n-eval-eps', default=3--, type=int,
                        help='Number of evaluation episodes done every eval_freq calls to env.step')
    parser.add_argument('--log-interval', default=500, type=int, help='Log every N calls to env.step')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--msg', default='', type=str, help='User message to add to info.txt')
    parser.add_argument('-L', '--hidden-layer-size', default=128, type=int, help='Network hidden layer size. All layers are equal size')
    parser.add_argument('-H', '--n-hidden-layers', default=2, type=int, help='Nb. of networks will hidden layers')

    return parser.parse_args()


def ppo_learning_rate(x):
    lr1 = 3e-4
    return lr1 * x


def ppo_clip_range(x):
    clip_range1 = 0.4
    return clip_range1
    # return clip_range1 * x


def main():
    args = parse_args()

    env_kw = dict(pulse_length=int(args.pulse_length),
                  delta_mode=args.delta_mode,
                  T=float(args.max_time_ns),
                  fid_thresh=float(args.fid_thresh),
                  action_scaling={'z':float(args.a_scale)},
                  a_norm_max=float(args.a_norm_max))
                  # model_params={'omega_s': [5.8899, 5.0311],
                  #               'alpha_s': [-324e-3, -235e-3],
                  #               'g': [100e-3, 71.4e-3],
                  #               'n_levels': 3,
                  #               'coupler_dims': 3,
                  #               'num_qubits': 2})

    n_envs = args.n_envs
    env = make_vec_env(lambda: ZCQPEE(**env_kw), n_envs=n_envs)
    eval_env = ZCQPEE(**env_kw)
    env_yaml_fn = str(eval_env) + '.yml'

    info_fn = 'info.txt'
    TRAINING_MESSAGE = f"{repr(eval_env)}\n" + f"Nb. envs: {n_envs}\n\n{args.msg}\n. "
    n_eval_eps = int(args.n_eval_eps)
    print(f'n_obs={eval_env.n_obs}\tn_act={eval_env.n_act}')

    # Parameter setup
    n_train = int(args.n_train)
    n_steps = int(args.n_steps)
    batch_size = int(args.batch_size)
    n_epochs = int(args.n_epochs)
    # save_freq = max(n_envs * n_steps, int(args.save_freq))
    save_freq = int(args.save_freq)
    eval_freq = max(n_envs * n_steps, int(args.eval_freq))
    log_interval = eval_freq

    # Setting device on the CPU only for now since I am working on my laptop
    if args.no_cuda:
        device = 'cpu'
    else:
        device = 'cuda'

    gamma = args.gamma
    SEED = args.seed

    L = args.hidden_layer_size
    H = args.n_hidden_layers
    policy_kwargs = dict(activation_fn=tc.nn.ReLU,
                         net_arch=dict(pi=[L]*H, vf=[L]*H))

    algo_str = args.algo
    policy_type = 'MlpPolicy'
    if algo_str == 'PPO':
        algo = PPO
        algo_kw = dict(batch_size=batch_size, n_steps=n_steps, learning_rate=ppo_learning_rate, device=device, n_epochs=n_epochs, gamma=gamma, clip_range=ppo_clip_range, max_grad_norm=0.5, gae_lambda=0.95, ent_coef=args.ent_coef, vf_coef=0.2, use_sde=False, stats_window_size=10, seed=SEED, verbose=1)  # PPO
    elif algo_str == 'TRPO':
        algo = TRPO
        algo_kw = dict(batch_size=batch_size, n_steps=n_steps, learning_rate=ppo_learning_rate, device=device, gamma=gamma, gae_lambda=0.95, seed=SEED, verbose=1)  # TRPO
    elif algo_str == 'RecurrentPPO':
        algo = RecurrentPPO
        policy_type = 'MlpLstmPolicy'
        algo_kw = dict(batch_size=batch_size, n_steps=n_steps, learning_rate=ppo_learning_rate, device=device, n_epochs=n_epochs, gamma=gamma, clip_range=ppo_clip_range, max_grad_norm=0.5, gae_lambda=0.95, ent_coef=args.ent_coef, vf_coef=0.2, use_sde=False, stats_window_size=10, seed=SEED, verbose=1)  # RecurrentPPO
    else:
        raise NotImplementedError

    dt_fmt_str = '%d-%m-%y_%H%M%S'

    '''
    NAME NEW MODEL OR RETRAIN FROM MODEL CHECKPOINT
    '''
    model_checkpoint = None
    model_zip = args.retrain_model_zip
    if model_zip is None:
        work_dir = os.path.join(f'{str(eval_env)}-{algo_str}')
        model_name = f"{dt.now().strftime(dt_fmt_str)}"
        model_path = os.path.join(work_dir, model_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        eval_env.to_yaml(os.path.join(model_path, env_yaml_fn))
        _best_model_path = os.path.join(model_path, 'best_model')
        if not os.path.exists(_best_model_path):
            os.makedirs(_best_model_path)
        eval_env.to_yaml(os.path.join(_best_model_path, env_yaml_fn))
    else:
        eval_env.from_yaml(ZCQPEE.find_yaml_in_dir(os.path.dirname(model_zip)))
        for k, v in env_kw.items():
            assert eval_env.model_params[k] == v, f'New vs. original training env mismatch! New training env parameters don\'t match the original environment parameters: New {k}: {v} != Original {k}: {eval_env.model_params[k]}'

    '''
    SAVE INFORMATION ABOUT THIS TRAINING SESSION
    Including model args, script args, environment details, simulation details
    '''
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    with open(os.path.join(model_path, info_fn), 'w') as f:
        f.write(TRAINING_MESSAGE)
        f.write(f'\nRL algo:  {algo_str}\n')

        f.write('\npolicy_kwargs:\n')
        pk = policy_kwargs.copy()
        pk.pop('activation_fn')
        json.dump(pk, f, indent=10)

        f.write('\nalgo_kwargs:\n')
        ak = algo_kw.copy()

        if not isinstance(ak['learning_rate'], float):
            ak['learning_rate'] = f'Linear decay from {ppo_learning_rate(1)} -> {ppo_learning_rate(0)}'
        if 'PPO' in algo_str:
            if not isinstance(ak['clip_range'], float):
                ak['clip_range'] = f'Linear decay from {ppo_clip_range(1)} -> {ppo_clip_range(0)}'
        json.dump(ak, f, indent=10)

    '''
    INITIALISE NEW MODEL OR LOAD LATEST CHECKPOINT
    '''
    if model_checkpoint is not None:
        model = algo.load(os.path.join(model_path, model_checkpoint))
        model.set_env(env)
        reset_num_timesteps = False
    else:
        # model = algo('MlpPolicy', env, tensorboard_log=os.path.join(model_path, 'tb_logs'), policy_kwargs=policy_kwargs, **algo_kw)
        tb_log = os.path.join(model_path, 'tb_logs')
        os.makedirs(tb_log)
        model = algo(policy_type, env, tensorboard_log=tb_log, policy_kwargs=policy_kwargs, **algo_kw)
        reset_num_timesteps = True

    '''
    SETUP TRAINING CALLBACKS AND NEW LOGGER
    '''
    # Stop training if there is no improvement after more than 3 evaluations
    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=100, min_evals=1000, verbose=1)
    checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=model_path)
    eval_callback = EvalCallback(eval_env=eval_env,
                                 n_eval_episodes=n_eval_eps,
                                 eval_freq=eval_freq,
                                 best_model_save_path=os.path.join(model_path, 'best_model'),
                                 log_path=os.path.join(model_path, 'evals'),
                                 callback_after_eval=stop_train_callback,
                                 verbose=1,
                                 algo=algo_str)

    new_logger = configure(os.path.join(model_path, 'logs'), ['stdout', 'tensorboard'])
    model.set_logger(new_logger)

    '''
    START TRAINING MODEL
    '''
    print(f'Training {model_name} in {model_path}...')
    model.learn(total_timesteps=n_train, progress_bar=False, log_interval=log_interval, tb_log_name=model_name,
                callback=[checkpoint_callback, eval_callback], reset_num_timesteps=reset_num_timesteps)


if __name__ == '__main__':
    main()
