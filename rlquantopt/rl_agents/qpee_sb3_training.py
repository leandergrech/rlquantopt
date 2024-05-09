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
from qutip import qobj
from rlquantopt.rl_envs.vec_qu_pulse_episodic_env import VecQuPulseEpisodicEnv as VQPEE
from rlquantopt.rl_envs.qu_pulse_episodic_env import QuPulseEpisodicEnv as QPEE


def parse_args():
    parser = argparse.ArgumentParser('RLQuantOpt - training RL agent on QuPulseEpisodic vectorised environment')
    parser.add_argument('--n-envs', default=16, type=int, help='Number of parallel environments')
    parser.add_argument('--n-train', default=int(5e5), type=int, help='Number of training steps')
    parser.add_argument('--save-freq', default=1000, type=int, help='Save model every N calls to env.step')
    parser.add_argument('--log-interval', default=1, type=int, help='Save model every N calls to env.step')
    parser.add_argument('--no_cuda', action='store_true')

    return parser.parse_args()


def main():
    args = parse_args()

    n_envs = int(args.n_envs)
    pulse_length = 50
    sparse_reward = False
    env = VQPEE(num_envs=n_envs, pulse_length=pulse_length, sparse_reward=sparse_reward)

    n_steps = 100
    batch_size = 256 #(n_envs * n_steps) // 100
    # n_train = int(5e5)
    n_train = int(args.n_train)
    log_interval = int(args.log_interval)
    save_freq = int(args.save_freq // n_envs)

    device = 'cuda'
    if args.no_cuda:
        device = 'cpu'

    algo = 'PPO'
    work_dir = os.path.join(f'VQPEE-{algo}')
    model_name = f"{dt.now().strftime('%d-%m-%y_%H%M%S')}_{n_envs}-envs"
    model_path = os.path.join(work_dir, model_name)

    callback = CheckpointCallback(save_freq=save_freq, save_path=model_path)
    new_logger = configure(os.path.join(model_path, 'logs'), ['stdout', 'csv', 'tensorboard'])
    env = VecMonitor(env, filename=os.path.join(model_path, 'vec_monitor'))

    if algo == 'PPO':
        model = PPO('MlpPolicy', env, tensorboard_log=os.path.join(model_path, 'tb_logs'), n_steps=n_steps, batch_size=batch_size, device=device)
    else:
        raise NotImplementedError

    model.set_logger(new_logger)

    print(f'Training {model_name}...')
    model.learn(total_timesteps=n_train, progress_bar=True, log_interval=log_interval, tb_log_name=model_name,
                callback=callback)
    # model.save(model_name)


if __name__ == '__main__':
    main()