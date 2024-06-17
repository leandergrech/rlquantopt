import matplotlib.pyplot as plt
from datetime import datetime as dt
# from qutip.qobjevo import QobjEvo
import numpy as np
from tqdm import tqdm
# from rlquantopt.rl_envs.qu_pulse_episodic_env import QuPulseEpisodicEnv as QPEE
from rlquantopt.rl_envs.zc_qpee import ZCQPEE

pulse_length = 60
env = ZCQPEE(pulse_length=pulse_length, sparse_reward=False, inc_off_diag=False)
recons_amps, global_tlist = env.get_optimal_pulse()
plt.plot(global_tlist, recons_amps, marker='.')
# recons_amps = [recons_amps]
recons_amps, global_tlist = env.get_reconstructed_pulses_with_uniform_time()
plt.plot(global_tlist[:-1], recons_amps[0], marker='x')

plt.show()
# recons_amps = [recons_amps]
# recons_amps, global_tlist = env.get_reconstructed_pulses_with_uniform_time()

o, _ = env.reset()

idx = 0
trunc = False
pbar = tqdm(total=pulse_length)
rews = []
a = np.zeros(env.n_act)
while not trunc:
    if idx > 0:
        for j, r_amp in enumerate(recons_amps):
            a[j] = r_amp[idx] - r_amp[idx-1]
    else:
        for j, r_amp in enumerate(recons_amps):
            a[j] = r_amp[idx]

    a = env.norm_action(a)
    # a = np.random.uniform(-1, 1, env.n_act)
    # if int(idx/8) % 2 == 0:
    #     a = np.array([1])
    # else:
    #     a = np.array([-1])
    # a = model.predict(o)[0]
    # a = np.ones(env.n_act)
    o, r, term, trunc, info = env.step(a)
    # o, r, term, trunc, info = env.step(env.action_space.sample())
    rews.append(r)
    pbar.update(1)

    idx += 1
    # if idx > 5:
    #     break

import matplotlib as mpl

save_path = f"{dt.now().strftime('%m-%d-%y_%H%M%S')}_zcqpee_render_{pulse_length}sample_optimised_pulse.mp4"
# ani = env.render(save_path=save_path)
ani = env.render()
# plt.show()
print(f'Saving to: {save_path}')
ffwriter = mpl.animation.FFMpegWriter(fps=10)
ani.save(save_path, dpi=100, writer=ffwriter)
print(save_path + ' saved')
# plt.show()
