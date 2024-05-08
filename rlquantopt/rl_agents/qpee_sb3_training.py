import os
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


def testing_vqpee():
    n_envs = 1
    # vec_env = VQPEE(num_envs=n_envs, pulse_length=300)
    # envs = [QPEE(pulse_length=300, sparse_reward=True) for _ in range(n_envs)]
    vec_env = VQPEE(n_envs, pulse_length=300)
    init_state = vec_env.reset()
    print(f'Initial state shape: {init_state.shape}')
    for i in trange(10):
        actions = [vec_env.action_space.sample() for _ in range(n_envs)]
        observations, rewards, dones, infos = vec_env.step(actions)

    vec_env.render()


def main():
    n_envs = 10
    pulse_length = 300
    env = VQPEE(num_envs=n_envs, pulse_length=pulse_length)

    n_steps = 2048
    batch_size = 256 #(n_envs * n_steps) // 100
    n_train = int(5e5)
    log_interval = 1
    save_freq = 100

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