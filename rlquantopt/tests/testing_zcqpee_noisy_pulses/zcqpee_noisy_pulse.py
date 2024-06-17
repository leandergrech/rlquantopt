import matplotlib.pyplot as plt
from datetime import datetime as dt
# from qutip.qobjevo import QobjEvo
import numpy as np
from tqdm import tqdm
# from rlquantopt.rl_envs.qu_pulse_episodic_env import QuPulseEpisodicEnv as QPEE
from rlquantopt.rl_envs.zc_qpee import ZCQPEE


def run_noisy_ep(env, pulse, noise_std):
    recons_amps_noisy = pulse + np.random.normal(0, noise_std, size=pulse.shape)
    o, _ = env.reset()
    idx = 0
    trunc = False
    pulse_length = env.pulse_length
    rew_shot = np.zeros(pulse_length)
    a = np.zeros(env.n_act)
    while not trunc:
        if idx > 0:
            for j, r_amp in enumerate(recons_amps_noisy):
                a[j] = r_amp[idx] - r_amp[idx - 1]
        else:
            for j, r_amp in enumerate(recons_amps_noisy):
                a[j] = r_amp[idx]

        a = env.norm_action(a)
        try:
            o, r, term, trunc, info = env.step(a)
        except Exception as e:
            # print(e)
            print(f'Crashed on step {idx}')
            break
        r = env.rew2fid(r)
        rew_shot[idx] = r
        idx += 1

    return rew_shot


fig, axs = plt.subplots(3, 2, figsize=(10, 10))
axs = axs.flatten()
pulse_length = 60
n_shots = 20
fig.suptitle(f'{n_shots} shots per noise std - pulse_length = {pulse_length}')

env = ZCQPEE(pulse_length=pulse_length, sparse_reward=False, inc_off_diag=False)
recons_amps, global_tlist = env.get_reconstructed_pulses_with_uniform_time()
zero_noise_rews = run_noisy_ep(env, recons_amps, 0)

noise_stds = (1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2e-1)
pbar = tqdm(total=len(noise_stds))
for ax, noise_std in zip(axs, noise_stds):
    rews = []
    for _ in range(n_shots):
        rew_shot = run_noisy_ep(env, recons_amps, noise_std)
        rews.append(rew_shot)

    pbar.update(1)
    ax.set_title(f'Noise std={noise_std:.0e}')
    ax.set_ylabel('Fidelity')
    ax.set_ylim(0, 1)
    ax.grid(which='major', linestyle='--', color='grey', linewidth=1)
    ax.grid(which='minor', linestyle=':', color='lightgrey', linewidth=0.5)
    ax.minorticks_on()
    ax.plot(global_tlist[:-1], zero_noise_rews, lw=0.5, c='k', label='No noise')
    ax.fill_between(global_tlist[:-1], np.min(rews, axis=0), np.max(rews, axis=0), alpha=0.5, label='Noisy (Min-Max)')
    ax.plot(global_tlist[:-1], np.median(rews, axis=0), label='Noisy (Median)')
    ax.plot(global_tlist[:-1], np.mean(rews, axis=0), label='Noisy (Mean)')
    ax.legend(loc='upper left')

fig.tight_layout()
fig.savefig(f'zcqpee_noisy_pulses_{pulse_length}pulse_length.pdf')
# plt.show()