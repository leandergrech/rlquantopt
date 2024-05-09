from typing import Type, List, Any, Optional
from multiprocessing import Process
from datetime import datetime as dt
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvIndices, VecEnvStepReturn

from rlquantopt.rl_envs.qu_pulse_episodic_env import QuPulseEpisodicEnv as QPEE


class VecQuPulseEpisodicEnv(SubprocVecEnv):
    def __init__(self, num_envs, pulse_length, sparse_reward=True):
        self.envs = [QPEE(pulse_length=pulse_length, sparse_reward=sparse_reward) for _ in range(num_envs)]
        super().__init__(self.envs, start_method='fork')

    def render(self, mode: Optional[str] = None, save_prefix=None):
        processes = []
        if save_prefix is None:
            save_prefix = f'{dt.now().strftime("%d-%m-%y_%H%M%S")}'
        for i in range(self.num_envs):
            save_path = f'{save_prefix}_{i}_qpee_render.gif'
            p = Process(target=self.envs[i].render, args=('human', save_path,))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()


if __name__ == '__main__':
    n_envs = 10
    envs = [QPEE(pulse_length=300, sparse_reward=True) for _ in range(n_envs)]
    venv = SubprocVecEnv(envs)
    venv = VecQuPulseEpisodicEnv(num_envs=4, pulse_length=300)
    print(venv.reset())

