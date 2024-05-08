import matplotlib.pyplot as plt
from datetime import datetime as dt

import numpy as np
from tqdm import tqdm
from rlquantopt.rl_envs.qu_pulse_episodic_env import QuPulseEpisodicEnv as QPEE

pulse_length = 300
env = QPEE(pulse_length=pulse_length, sparse_reward=False)
recons_amps, global_tlist = env.get_reconstructed_pulses_with_uniform_time()

env.reset()

idx = 0
trunc = False
pbar = tqdm(total=pulse_length)
rews = []
a = np.zeros(env.n_act)
while not trunc:
    if idx > 0:
        for j, r_amp in enumerate(recons_amps):
            a[j] = r_amp[idx] - r_amp[idx-1]
    a = env.norm_action(a)
    o, r, term, trunc, info = env.step(a)
    # o, r, term, trunc, info = env.step(env.action_space.sample())
    rews.append(r)
    pbar.update(1)

    idx += 1
    # if idx > 5:
    #     break

save_path = f"{dt.now().strftime('%m-%d-%y_%H%M%S')}_qpee_render.mp4"
env.render(save_path=save_path)
print(save_path + ' saved')
# plt.show()
