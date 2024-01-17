import matplotlib as mpl
import matplotlib.pyplot as plt
from qutip_qip.circuit import QubitCircuit
from tqdm import trange, tqdm
import numpy as np

from rlquantopt.rl_envs import QuPulseEnv
from rlquantopt.utils import grid_on

mpl.rcParams['font.size'] = 25


def get_ch_idx(ch, which):
    if ch == 'sx0':
        ch_idx = np.array([0])
    elif ch == 'sy0':
        ch_idx = np.array([2])
    elif ch == 'sz0':
        ch_idx = np.array([1])
    elif ch == 'all':
        ch_idx = np.arange(3)
    else:
        raise Exception('WTF')

    if which == 'dur':
        ch_idx += 3
    elif which == 'both':
        ch_idx = np.concatenate([ch_idx, ch_idx + 3])
    elif which != 'amp':
        raise Exception('WTF')

    return ch_idx


def constant_action_episode(ch, which):
    qc = QubitCircuit(N=1)
    qc.add_gate('X', targets=0)
    # qc.add_gate('X', targets=0)
    # qc.add_gate('X', targets=0)
    # processor_params = dict(zz_crosstalk=True, t1=50.e3, t2=20.e3)
    processor_params = dict(zz_crosstalk=True,t1=141.72e3, t2=31.22e3, wq=[5.26], alpha=[-0.33983]) # nairobi settings

    env = QuPulseEnv(qc=qc, processor_params=processor_params)
    print(repr(env))
    env.action_eps = 1e-1
    env.rew_scale = 1.
    env.max_steps = 100
    env.max_obs_scale = 10.
    # ch = 'sx0'  # sx0/sy0/sz0/all
    # which = 'both'   # amp/dur/both

    ch_idx = get_ch_idx(ch, which)

    init_state = np.ones(env.observation_space.shape[0])
    if isinstance(ch_idx, int):
        if ch_idx < 0:
            init_state *= env.action_eps
        else:
            init_state[ch_idx] *= env.action_eps
    else:
        init_state[ch_idx] *= env.action_eps

    o = env.reset(options=dict(init_state=init_state))
    d = False

    obses = []
    acts = []
    rews = []

    for _ in trange(env.max_steps):
        a = np.ones_like(init_state).astype(int)

        a[ch_idx] = 2

        # a = env.action_space.sample()
        otp1, r, d, info = env.step(a)

        acts.append(a)
        obses.append(otp1.copy())
        rews.append(r)
        o = otp1

    rews = np.array(rews)

    fig, axs = plt.subplots(3)
    for ax, oc, nc in zip(axs, env.orig_coeff.items(), env.processor.pulses):
        ax.plot(oc[1], c='k', label=oc[0])
        ax.plot(nc.coeff, c='r', ls='--')
        ax.legend(loc='best')

    fig, axs = plt.subplots(3, figsize=(18, 14), gridspec_kw={'height_ratios':[3, 3, 5]})
    fig.suptitle(processor_params)

    ax = axs[0]
    print([item for item in env.orig_coeff.keys()])
    lss = ('solid', 'dashed', 'dotted', 'dotted', 'dashed', 'solid')
    ms = ('o', 'x', '^')
    keys = [item for item in env.orig_coeff.keys()]
    for i, obs in enumerate(np.array(obses).T):
        label = keys[i % 3]
        label += '_amp' if i < 3 else '_duration'
        ax.plot(obs, marker=ms[i % 3], ls=lss[i], lw=1, label=label, markersize=2 * (int(i/3) + 1))
        # ax.plot(obs, ls=lss[i % 3], lw=3, label=label)
    ax.legend(loc='best', ncol=2, prop={'size':10})
    ax.set_ylabel('Pulse scaling')

    ax = axs[1]
    for i, act in enumerate(np.array(acts).T):
        label = keys[i % 3]
        label += '_amp' if i / 3 < 1 else '_duration'
        ax.plot(env.action_eps * (act - 1), marker=ms[i % 3], ls=lss[i], lw=1, label=label, markersize=2 * (int(i/3) + 1))
    ax.legend(loc='lower right', ncol=2, prop={'size':16})
    ax.set_ylim((-env.action_eps*1.2, env.action_eps*1.2))
    ax.set_ylabel('Discrete actions')

    ax = axs[2]
    ax.plot(rews, marker='x', lw=3)
    # for item in rews*100.: print(f'{item:.4f}%', end='\t')
    # print()
    # ax.set_yscale('log')
    ax.set_ylabel('Rewards')
    max_rews = max(rews)
    ax.set_title(f'Max fidelity = {max_rews/env.rew_scale}')
    ax.axhline(np.max(rews), ls='dashed', color='k')
    grid_on(ax=ax, axis='y', major_loc=0.1, minor_loc=1e-2)

    for ax in axs:
        ax.axvline(np.argmax(rews), ls='dashed', color='k')
        grid_on(ax=ax, axis='x', major_loc=5, minor_loc=1)
        ax.set_xlabel('Steps')

    fig.tight_layout()
    plt.savefig(f'../rl_analysis/results/{ch}_{which}.png')
    # plt.show()


def sweep_n_xgates():
    N = 50
    action_eps = 1e-1
    rew_scale = 1.0
    max_steps = 10
    processor_params = dict(zz_crosstalk=True,t1=141.72e3, t2=31.22e3, wq=[5.26], alpha=[-0.33983]) # Nairobi settings

    nb_xgates = np.arange(1, N, 2)
    fidelities = []
    for n in tqdm(nb_xgates):
        qc = QubitCircuit(N=1)
        for _ in range(n):
            qc.add_gate('X', targets=0)

        env = QuPulseEnv(qc=qc, processor_params=processor_params)
        env.action_eps = action_eps
        env.rew_scale = rew_scale
        env.max_steps = max_steps
        env.reset()
        _, fidelity, *_ = env.step(np.ones(env.observation_space.shape[0]))

        fidelities.append(fidelity)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(nb_xgates, fidelities, 'o-', markersize=10)
    ax.set_ylabel('Fidelity')
    ax.set_xlabel('Nb. consecutive X-gates')

    xticks = nb_xgates[0: len(nb_xgates): len(nb_xgates)//5]
    ax.set_xticks(xticks, xticks)

    grid_on(ax=ax, axis='y')
    grid_on(ax=ax, axis='x', minor_grid=False, major_loc=1)
    fig.tight_layout()
    plt.savefig(f'../rl_analysis/results/nb_xgate_sweep.png')
    plt.show()


if __name__ == '__main__':
    # sweep_n_xgates()
    for ch in ('sx0', 'sy0', 'sz0', 'all'):
        for which in ('amp', 'dur', 'both'):
            constant_action_episode(ch=ch, which=which)