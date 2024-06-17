from typing import Type, List, Any, Optional
from multiprocessing import Process
from datetime import datetime as dt
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvIndices, VecEnvStepReturn


from rlquantopt.rl_envs.zc_qpee import ZCQPEE

class VecZCQPEE(SubprocVecEnv):
    def __init__(self, num_envs, pulse_length):
        env_kwargs = dict(pulse_length=pulse_length)
        self.envs = [ZCQPEE(**env_kwargs) for _ in range(num_envs)]
        super().__init__(self.envs, start_method='fork')

    def render(self, mode: Optional[str] = None, save_prefix=None) -> Optional[np.ndarray]:
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
    from tqdm import tqdm
    n_envs = 2
    pulse_length = 120
    venv = VecZCQPEE(num_envs=n_envs, pulse_length=pulse_length)
    env_ = venv.envs[0]
    recons_amps, global_tlist = env_.get_reconstructed_pulses_with_uniform_time()

    idx = 0
    done = False
    pbar = tqdm(total=pulse_length)
    rews = []
    a = np.zeros(env_.n_act)
    os = venv.reset()
    while not done:
        if idx >= pulse_length - 1:
            break
        if idx > 0:
            for j, r_amp in enumerate(recons_amps):
                a[j] = r_amp[idx] - r_amp[idx - 1]
        else:
            for j, r_amp in enumerate(recons_amps):
                a[j] = r_amp[idx]

        a = env_.norm_action(a)
        a = np.repeat([a], n_envs, axis=0)
        # a += np.random.normal(0, 0.001, size=a.shape)
        a = np.squeeze(a).reshape(-1, 1)
        venv.step_async(a)
        o, r, dones, info = venv.step_wait()
        done = any(dones)
        rews.append(r)
        pbar.update(1)

        idx += 1

    import matplotlib.pyplot as plt
    plt.plot(np.array(rews).T)
    plt.show()
    # venv.render(save_prefix='testing_veczcqpee')