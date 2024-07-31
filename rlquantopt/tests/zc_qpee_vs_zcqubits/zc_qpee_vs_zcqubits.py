import os
import numpy as np
import matplotlib.pyplot as plt
from qutip import ket, Qobj, expect

from rlquantopt.rl_envs.zc_qpee import ZCQPEE
from rlquantopt.rl_envs.zcqubits import test_step
from rlquantopt.utils.utils import decimal_to_base3

pulse_file = '/rlquantopt/rl_envs/configs/data_lilmc/data_lilmc.csv'
save_dir = '/home/leander/code/rlquantopt/rlquantopt/tests/zc_qpee_vs_zcqubits'
T = 200
env_kwargs = {'pulse_length': 200,
              'T': T,
              'delta_mode': False,
              'default_model_params': True}
# pulse_file = f'/home/leander/code/rlquantopt/rlquantopt/rl_agents/ZCQPEE300pl-PPO/20-06-24_004308_ZCQPEE300pl/best_model/pulses/best_model_T{T}ns_to_term.csv'
# save_dir = os.path.splitext(pulse_file)[0]
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

pulse_name = os.path.splitext(os.path.basename(pulse_file))[0]
test_name = f'zcqpee_render_{pulse_name}'
env_mp4_path = os.path.join(save_dir, f'zcqpee_render_{test_name}.mp4')

full_dims = [3, 3, 3]
basis_states = [ket((0, 0, 0), dim=full_dims),
                ket((0, 1, 0), dim=full_dims),
                ket((1, 0, 0), dim=full_dims),
                ket((1, 1, 0), dim=full_dims)]
unitary = Qobj(np.array([
    [1, 0, 0, 0],
    [0, 0, 1j, 0],
    [0, 1j, 0, 0],
    [0, 0, 0, 1]
]), dims=[[2, 2], [2, 2]])
target_states = [sum(unitary[i, j] * basis_states[i]
                     for i in range(unitary.shape[0])) for j in range(unitary.shape[1])]

states_env, acts, _, fids_env = ZCQPEE.evaluate_pulse(pulse_file=pulse_file, save_path=env_mp4_path, render=False, **env_kwargs)
states_sim, pulse, fids_sim = test_step(pulse_file=pulse_file, save_dir=save_dir, plot=False)

print(f'Len acts: {len(acts)}')
print(f'Len pulse: {len(pulse)}')

e_ops = [state.proj() for state in basis_states]


def get_populations(states):
    populations = []
    for k, (b, t) in enumerate(zip(basis_states, target_states)):
        pop = [[expect(e, b)] for e in e_ops]
        for i, s_t in enumerate(states[k]):
            for j, e in enumerate(e_ops):
                pop[j].append(expect(e, s_t))
        populations.append(pop)
    return np.array(populations)


def get_qobj_idx(qobj, idx):
    return qobj.full()[idx]


pops_env = get_populations(states_env)
pops_sim = get_populations(states_sim)

basis_labels = ['|000⟩', '|010⟩', '|100⟩', '|110⟩']
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10,8))
axs = np.ndarray.flatten(axs)

for i, (ax, title) in enumerate(zip(axs, basis_labels)):
    for k, label in enumerate(basis_labels):
        ax.plot(pops_sim[i, k], label=f'Sim {label}')
        ax.plot(pops_env[i, k], label=f'Env {label}', ls='dashed')
    ax.legend(loc='best')
    ax.set_title(title)

fig.suptitle('ZCQPEE vs. ZCQubits population evolutions')
fig.tight_layout()
fig.savefig(os.path.join(save_dir, 'zcqpee_vs_zcqubits_populations.pdf'))

fig, ax = plt.subplots()
fig.suptitle('Optimised pulse vs. Action used by Env')
ax.plot(acts, label='Actions')
ax.plot(pulse, label='Pulse')

fig, axs = plt.subplots(6, 5, figsize=(25,22))
axs = np.ndarray.flatten(axs)
axs = np.ndarray.flatten(axs)[:27]
for i, ax in enumerate(axs):
    ax.set_title(f'|{decimal_to_base3(i)}⟩')
    for j, b_label in enumerate(basis_labels):
        pops_env = [item.full()[i][0] for item in states_env[j]]
        pops_sim = [item.full()[i][0] for item in states_sim[j]]

        ax.plot(np.abs(pops_env), label=f'Env {b_label}')
        ax.plot(np.abs(pops_sim), label=f'Sim {b_label}', ls='dashed')
    ax.legend(loc='upper left')
fig.suptitle('State populations')
fig.tight_layout()
fig.savefig(os.path.join(save_dir, 'zcqpee_vs_zcqubits_state_evolutions.pdf'))

# plt.show()
