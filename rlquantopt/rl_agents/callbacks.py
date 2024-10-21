import os
import warnings
from typing import Any, Dict, List, Optional, Union
from datetime import datetime as dt
import numpy as np
import gymnasium as gym
from stable_baselines3.common.vec_env import VecEnv, sync_envs_normalization
from stable_baselines3.common.callbacks import EventCallback, BaseCallback

from rlquantopt.rl_envs.zc_qpee import ZCQPEE


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

        self.eval_env: gym.Env = eval_env   # DOES NOT SUPPORT VecEnv!!!
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

    def eval_policy(self, norm_rewards=True):
        rews = []
        fids = []
        tv_penalties = []
        ep_lens = []
        MAX_STEPS = self.eval_env.pulse_length
        N = self.eval_env.N_TIME_STEPS
        for _ in range(self.n_eval_episodes):
            rews.append([])
            fids.append([])
            tv_penalties.append([])
            ep_lens.append(0)
            obs, _ = self.eval_env.reset()
            done = False
            lstm_states = None
            episode_start = np.ones((1,), dtype=bool)
            ep_len = 0
            while not done:
                if 'Recurrent' in self.algo:
                    actions, lstm_states = self.model.predict(obs, state=lstm_states, episode_start=episode_start, deterministic=True)
                else:
                    actions = self.model.predict(obs, deterministic=True)[0]


                obs, rew, te, tr, info = self.eval_env.step(actions)
                fid = info.get('fidelity', 0.0)
                tv_penalty = info.get('tv_penalty', 0.0)
                if norm_rewards:
                    rew /= self.eval_env.REW_SCALE
                    try:
                        tv_penalty /= self.eval_env.TV_PENALTY_SCALE
                    except ZeroDivisionError:
                        tv_penalty = 0.0

                rews[-1].append(rew)
                fids[-1].append(fid)
                tv_penalties[-1].append(tv_penalty)
                ep_lens[-1] += N
                done = te or tr
                ep_len += N

            remaining_steps = (MAX_STEPS - ep_len) // N
            rews[-1] = np.pad(rews[-1], (0, remaining_steps), mode='edge').tolist()
            fids[-1] = np.pad(fids[-1], (0, remaining_steps), mode='edge').tolist()
            tv_penalties[-1] = np.pad(tv_penalties[-1], (0, remaining_steps), mode='edge').tolist()

        return dict(episode_rewards=rews, episode_fidelities=fids, episode_tv_penalties=tv_penalties, episode_lengths=ep_lens)

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

            eval_episode_data = self.eval_policy()
            episode_rewards = eval_episode_data['episode_rewards']
            episode_fidelities = eval_episode_data['episode_fidelities']
            episode_tv_penalties = eval_episode_data['episode_tv_penalties']
            # episode_lengths = [len(item) for item in episode_rewards]
            episode_lengths = eval_episode_data['episode_lengths']

            self._is_success_buffer = [max(ep_rew) > 0 for ep_rew in episode_rewards]

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
            mean_tv_penalty, std_tv_penalty = np.mean(episode_tv_penalties), np.std(episode_tv_penalties)
            mean_init_rewards = np.mean([ep_rews[0] for ep_rews in episode_rewards])
            best_rewards = [max(ep_rews) for ep_rews in episode_rewards]
            worst_rewards = [min(ep_rews) for ep_rews in episode_rewards]
            best_fidelities = [max(ep_fids) for ep_fids in episode_fidelities]

            mean_best_rewards = np.mean(best_rewards)
            mean_worst_rewards = np.mean(worst_rewards)
            mean_best_fidelities = np.mean(best_fidelities)
            mean_ep_reward_improvement = mean_best_rewards - mean_init_rewards
            mean_ep_len_best = np.mean([np.argmax(ep_rews) for ep_rews in episode_rewards])
            mean_best_T_ns = self.eval_env.dt * mean_ep_len_best
            mean_success = np.mean(np.array(self._is_success_buffer).astype(int))

            if self.verbose >= 1:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/std_reward", float(std_reward))
            self.logger.record("eval/mean_ep_length", float(mean_ep_length))
            self.logger.record("eval/std_ep_length", float(std_ep_length))
            self.logger.record("eval/mean_tv_penalty", float(mean_tv_penalty))
            self.logger.record("eval/std_tv_penalty", float(std_tv_penalty))
            self.logger.record('eval/mean_ep_reward_improvement', float(mean_ep_reward_improvement))
            self.logger.record('eval/mean_best_rewards', float(mean_best_rewards))
            self.logger.record('eval/mean_worst_rewards', float(mean_worst_rewards))
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