import os
import argparse
from datetime import datetime as dt
import gymnasium as gym
import numpy as np
from tqdm import tqdm, trange
from datetime import datetime as dt
from multiprocessing import Process, Queue, Pool

from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import PPO, SAC

from rlquantopt.rl_envs.vec_qu_pulse_episodic_env import VecQuPulseEpisodicEnv as VQPEE
from rlquantopt.rl_envs.qu_pulse_episodic_env import QuPulseEpisodicEnv as QPEE


def parse_args():
    parser = argparse.ArgumentParser('RLQuantOpt - training RL agent on QuPulseEpisodic vectorised environment')
    parser.add_argument('--n-envs', default=10, help='Number of parallel environments')
    parser.add_argument('--n-train', default=int(5e5), help='Number of training steps')
    parser.add_argument('--save-freq', default=1000, help='Save model every N calls to env.step')
    parser.add_argument('--log-interval', default=1, help='Save model every N calls to env.step')
    # parser.add_argument('--lr', default=1e-5, type=float, help="Learning rate")
    # parser.add_argument('--max_steps', default=100, type=int)
    # parser.add_argument('--per_device_train_batch_size', default=2, type=int)
    # parser.add_argument('--per_device_eval_batch_size', default=2, type=int)
    # parser.add_argument('--eval_steps', default=25, type=int)
    # parser.add_argument('--save_steps', default=50, type=int)
    # parser.add_argument('--no_cuda', action='store_true')
    # parser.add_argument("--seed", type=int, default=42, help="For reproducibility")

    return parser.parse_args()


def main():
    args = parse_args()

    # n_envs = 10
    n_envs = args.n_envs
    pulse_length = 300
    env = VQPEE(num_envs=n_envs, pulse_length=pulse_length)

    n_steps = 2048
    batch_size = 256 #(n_envs * n_steps) // 100
    # n_train = int(5e5)
    n_train = args.n_train
    log_interval = args.log_interval
    save_freq = args.save_freq // n_envs

    # n_steps = 10
    # batch_size = 5#(n_envs * n_steps) // 100
    # n_train = 100
    # save_freq = max(1000 // n_envs, 1)

    algo = 'PPO'
    work_dir = os.path.join(f'VQPEE-{algo}')
    model_name = f"{dt.now().strftime('%d-%m-%y_%H%M%S')}_{n_envs}-envs"
    model_path = os.path.join(work_dir, model_name)

    callback = CheckpointCallback(save_freq=save_freq, save_path=model_path)
    new_logger = configure(os.path.join(model_path, 'logs'), ['stdout', 'csv', 'tensorboard'])
    env = VecMonitor(env, filename=os.path.join(model_path, 'vec_monitor'))

    model = PPO('MlpPolicy', env, tensorboard_log=os.path.join(model_path, 'tb_logs'), n_steps=n_steps, batch_size=batch_size, device='cuda')
    # model = SAC('MlpPolicy', env, tensorboard_log=os.path.join('VQPEE-SAC', model_name))
    model.set_logger(new_logger)

    print(f'Training {model_name}...')
    model.learn(total_timesteps=n_train, progress_bar=True, log_interval=log_interval, tb_log_name=model_name, callback=callback)
    # model.save(model_name)


if __name__ == '__main__':
    main()