import numpy as np
from datetime import datetime as dt
from tqdm import trange
from rlquantopt.rl_envs.qu_pulse_episodic_env import QuPulseEpisodicEnv as QPEE

env  = QPEE(pulse_length=50, sparse_reward=False, use_full_liouville=False)
env.reset()
durations = []
for _ in trange(50):
    start = dt.now()
    action = np.random.uniform(-1, 1, env.n_act)
    env.step(action)
    dur = dt.now() - start
    durations.append(dur.total_seconds())

print(f'Using 4 basis states: {np.mean(durations):.4f}s')

env  = QPEE(pulse_length=50, sparse_reward=False, use_full_liouville=True)
env.reset()
durations = []
for _ in trange(50):
    start = dt.now()
    action = np.random.uniform(-1, 1, env.n_act)
    env.step(action)
    dur = dt.now() - start
    durations.append(dur.total_seconds())

print(f'Using full liouville basis states: {np.mean(durations):.2f}s')
