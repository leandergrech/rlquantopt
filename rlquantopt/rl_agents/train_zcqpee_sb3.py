import os
import argparse
import warnings
import json
from typing import Any, Dict, List, Optional, Union
import gymnasium as gym
import numpy as np
import torch as tc
from datetime import datetime as dt
from stable_baselines3.common.vec_env import VecMonitor, VecEnv, sync_envs_normalization, DummyVecEnv
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback, EventCallback, BaseCallback
from stable_baselines3 import PPO, SAC, DDPG, TD3
from rlquantopt.rl_envs.zc_qpee import ZCQPEE
from rlquantopt.utils.rl_utils import evaluate_policy


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
    ):
        super().__init__(callback_after_eval, verbose=verbose)

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
            while not done:
                actions = self.model.predict(
                    obs,
                    deterministic=True
                )[0]
                obs, r, te, tr, info = self.eval_env.step(actions)
                rews[-1].append(r)
                done = te or tr
        return rews

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
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

            episode_rewards, episode_lengths, all_rewards = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            episode_rewards = self.eval_policy()
            episode_lengths = [len(item) for item in episode_rewards]

            if self.log_path is not None:
                assert isinstance(episode_rewards, list)
                assert isinstance(episode_lengths, list)
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_rewards.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                # np.savez(
                #     self.log_path,
                #     timesteps=self.evaluations_timesteps,
                #     results=self.evaluations_rewards,
                #     ep_lengths=self.evaluations_length,
                #     **kwargs,
                # )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            mean_init_rewards = np.mean([ep_rews[0] for ep_rews in episode_rewards])
            mean_best_rewards = np.mean([max(ep_rews) for ep_rews in episode_rewards])

            # mean_ep_reward_improvement = np.mean(np.subtract(best_rewards, init_rewards))
            mean_ep_reward_improvement = mean_best_rewards - mean_init_rewards
            mean_ep_len_best = np.mean([np.argmax(ep_rews) for ep_rews in episode_rewards])

            if self.verbose >= 1:
                print(
                    f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/std_reward", float(std_reward))
            self.logger.record("eval/mean_ep_length", float(mean_ep_length))
            self.logger.record("eval/std_ep_length", float(std_ep_length))
            self.logger.record("eval/mean_init_reward", float(mean_init_rewards))
            self.logger.record('eval/mean_best_rewards', float(mean_best_rewards))
            self.logger.record('eval/mean_ep_reward_improvement', float(mean_ep_reward_improvement))
            self.logger.record('eval/mean_ep_len_best', float(mean_ep_len_best))

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
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
    # parser.add_argument('--n-envs', default=16, type=int, help='Number of parallel environments')
    parser.add_argument('--delta-mode', action='store_true', help='Use ZCQPEE environment in delta action mode')
    parser.add_argument('--a-norm-max', default=10, type=int, help='Set the maximum normalised amplitude when using delta-mode')
    parser.add_argument('--a-scale', default=1e-1, type=float, help='Set action scaling during normalisation')
    parser.add_argument('--seed', default=123, type=int, help='Set random seed')
    parser.add_argument('--pulse-length', default=120, type=int, help='Maximum number of samples in a pulse')
    parser.add_argument('--max-time-ns', default=200, type=int, help='Pulse duration in ns')
    parser.add_argument('--n-train', default=250000, type=int, help='Number of training steps')
    parser.add_argument('--save-freq', default=10000, type=int, help='Save model every save_freq calls to env.step')
    parser.add_argument('--eval-freq', default=500, type=int, help='Evaluate model every eval_freq calls to env.step')
    parser.add_argument('--n-eval-eps', default=5, type=int, help='Number of evaluation episodes done every eval_freq calls to env.step')
    parser.add_argument('--log-interval', default=500, type=int, help='Log every N calls to env.step')
    # parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--retrain-latest', action='store_true')

    return parser.parse_args()


def ppo_learning_rate(x):
    lr1 = 3e-4
    return lr1*x


def main():
    args = parse_args()

    pulse_length = args.pulse_length
    delta_mode = args.delta_mode
    T = args.max_time_ns

    env = ZCQPEE(pulse_length=pulse_length, delta_mode=delta_mode, T=T)
    eval_env = ZCQPEE(pulse_length=pulse_length, delta_mode=delta_mode, T=T)

    info_fn = 'info.txt'
    TRAINING_MESSAGE = (f"ZCQPEE:   pulse_length={pulse_length}\n"
                        f"          action scaling = {env.action_channel_scaling} \t A_norm_max = {env.A_norm_max}\n"
                        f"          T = {env.T} ns \t ΔT = {env.dt:.3f} ns\n"
                        f"          ISWAP gate optimisation.\n"
                        f"          Action delta_mode={delta_mode}\n"
                        "")

    n_eval_eps = int(args.n_eval_eps)
    print(f'ZCQPEE pulse_length={pulse_length}\tn_obs={env.n_obs}\tn_act={env.n_act}')

    # PPO parameter setup
    n_steps = pulse_length*2
    n_epochs = 8
    batch_size = 32

    n_train = int(args.n_train)
    log_interval = int(args.log_interval)
    save_freq = int(args.save_freq)
    eval_freq = int(args.eval_freq)

    # Setting device on the CPU only for now since I am working on my laptop
    device = 'cpu'

    algo_str = 'PPO'
    algo = PPO
    # SEED = 234
    SEED = None
    policy_kwargs = dict(activation_fn=tc.nn.ReLU,
                         net_arch=dict(pi=[256, 128], vf=[256, 128]))
    algo_kw = dict(batch_size=batch_size, n_steps=n_steps, learning_rate=ppo_learning_rate, device=device,
                   n_epochs=n_epochs, gamma=0.95, max_grad_norm=0.2, gae_lambda=0.99, ent_coef=0.05, vf_coef=0.2,
                   use_sde=False, stats_window_size=10,
                   # sde_sample_freq=eval_freq,
                   sde_sample_freq=-1, seed=SEED, verbose=1)   # PPO

    work_dir = os.path.join(f'ZCQPEE{pulse_length}pl-{algo_str}')
    dt_fmt_str = '%d-%m-%y_%H%M%S'

    retrain_latest = args.retrain_latest
    model_checkpoint = None
    if retrain_latest:
        # Assumes that the model names start with the date time information in the format defined by `dt_fmt_str`
        models_available = [item for item in os.listdir(work_dir)]
        model_latest_date = sorted([dt.strptime('_'.join(item.split('_')[:2]), dt_fmt_str) for item in models_available])[-1].strftime(dt_fmt_str)
        model_name = [item for item in models_available if model_latest_date in item][0]
        model_path = os.path.join(work_dir, model_name)
        model_checkpoint = sorted([item for item in os.listdir(model_path) if 'steps' in item], key=lambda x: int(x.split('_')[2]))[-1]
    else:
        model_name = f"{dt.now().strftime(dt_fmt_str)}_ZCQPEE{pulse_length}pl"
        model_path = os.path.join(work_dir, model_name)

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
        ak.pop('learning_rate')
        json.dump(ak, f, indent=10)

    if model_checkpoint is not None:
        model = algo.load(os.path.join(model_path, model_checkpoint))
        model.set_env(env)
        reset_num_timesteps = False
    else:
        # model = algo('MlpPolicy', env, tensorboard_log=os.path.join(model_path, 'tb_logs'), policy_kwargs=policy_kwargs, **algo_kw)
        tb_log = os.path.join(model_path, 'tb_logs')
        os.makedirs(tb_log)
        model = algo('MlpPolicy', env, tensorboard_log=tb_log, policy_kwargs=policy_kwargs, **algo_kw)
        reset_num_timesteps = True

    checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=model_path)
    eval_callback = EvalCallback(eval_env=eval_env, n_eval_episodes=n_eval_eps, eval_freq=eval_freq, verbose=1, best_model_save_path=os.path.join(model_path, 'best_model'), log_path=os.path.join(model_path, 'evals'))
    new_logger = configure(os.path.join(model_path, 'logs'), ['stdout', 'tensorboard'])

    model.set_logger(new_logger)

    print(f'Training {model_name} in {model_path}...')
    model.learn(total_timesteps=n_train, progress_bar=False, log_interval=log_interval, tb_log_name=model_name,
                callback=[checkpoint_callback, eval_callback], reset_num_timesteps=reset_num_timesteps)


if __name__ == '__main__':
    main()