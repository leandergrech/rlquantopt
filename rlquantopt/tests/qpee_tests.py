import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
from cycler import cycler
from datetime import datetime as dt
from rlquantopt.rl_envs.qu_pulse_episodic_env import QuPulseEpisodicEnv

mpl.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')


def check_reconstructed_ideal_pulse():
    env = QuPulseEpisodicEnv()
    print(env.channel_labels)
    env.reset()

    # Obtain ideal amplitudes reconstructions
    ideal_pulses = env.ideal_pulses

    recons_amps, global_tlist = env.get_reconstructed_pulses_with_uniform_time()

    # Compute original fidelity with ideal pulses from qutip
    for j, (i_tli, i_amp) in enumerate(zip(ideal_pulses['tlist'], ideal_pulses['coeff'])):
        env.simulator.pulses[j].coeff = i_amp
        env.simulator.pulses[j].tlist = i_tli
    fidelity = env.reward_function()
    print(f'Fidelity from ideal pulse: {fidelity * 100:.2f}%')  # Gets 98.29% fidelity (CNOT)

    # Set simulator pulses with reconstructions
    for j, r_amp in enumerate(recons_amps):
        env.simulator.pulses[j].coeff = r_amp   # Gets 98.29% fidelity like the original ideal pulses (CNOT)
        env.simulator.pulses[j].tlist = global_tlist

    fig, ax = plt.subplots()
    for i_amps, i_tlist, lbl in zip(ideal_pulses['coeff'], ideal_pulses['tlist'], ideal_pulses['labels']):
        ax.plot(i_tlist, i_amps, marker='o', label=f'Ideal: {lbl}')
    for r_amps, lbl in zip(recons_amps, ideal_pulses['labels']):
        ax.plot(global_tlist[:-1], r_amps, marker='x', label=f'Recons: {lbl}')

    ax.legend(loc='best')
    ax.set_title('Ideal SWAP gate')
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Amplitude')

    fidelity = env.reward_function()
    print(f'Fidelity from reconstructed pulse: {fidelity*100:.2f}%')
    plt.show()


def reconstructed_ideal_pulse_with_environment():
    env = QuPulseEpisodicEnv(pulse_length=300, sparse_reward=False)

    recons_amps, global_tlist = env.get_reconstructed_pulses_with_uniform_time()

    env.reset()
    rews = []
    durations = []
    all_actions_denorm = np.empty(shape=(0, env.n_act))
    all_actions_norm = np.empty(shape=(0, env.n_act))
    for i, t in enumerate(tqdm(global_tlist[:-1])):
        start = dt.now()
        cur_action = np.zeros(env.n_act)
        if i > 0:
            for j, r_amp in enumerate(recons_amps):
                cur_action[j] = r_amp[i] - r_amp[i-1]

        # if i > 10:
        #     break
        all_actions_denorm = np.vstack((all_actions_denorm, cur_action))
        cur_action = env.norm_action(cur_action)
        # Add action noise
        # cur_action += np.random.normal(0, 0.01, env.n_act)
        all_actions_norm = np.vstack((all_actions_norm, cur_action))

        _, rew, *_ = env.step(cur_action)
        rews.append(rew)
        duration = dt.now() - start
        durations.append(duration.total_seconds()*1000.)

    last_rew = rews[-1]*100
    print(f'Final fidelity: {last_rew:.2f}%')
    # fig, ax = plt.subplots()

    fig, axs = plt.subplots(4, figsize=(23, 10))
    fig.suptitle('Noisy actions. Gaussian noise at 0.01 sigma')

    N = all_actions_norm.shape[0]
    assert  N == len(rews)

    ax = axs[0]
    ax.plot(global_tlist[:N], durations, marker='x')
    ax.set_ylabel('Duration (ms)')
    ax.set_title('Duration of every step')

    ax = axs[1]
    ax.plot(global_tlist[:N], rews, marker='x')
    ax.set_ylabel('Sparse reward')
    ax.set_title(f'Final fidelity: {last_rew:.2f}%')

    ax = axs[2]
    cmap = plt.get_cmap('tab10')
    for j, (acts, ch_lbl) in enumerate(zip(all_actions_denorm.T, env.channel_labels)):
        ax.step(global_tlist[:N], acts, c=cmap(j / (env.n_act - 1)), marker='^', label=ch_lbl)
    ax.set_ylabel('Denormed action amps')

    ax = axs[3]
    cmap = plt.get_cmap('tab10')
    for j, (acts, ch_lbl) in enumerate(zip(all_actions_norm.T, env.channel_labels)):
        ax.step(global_tlist[:N], acts, c=cmap(j / (env.n_act - 1)), marker='^', label=ch_lbl)
    ax.set_ylabel('Normed action amps')

    for ax in axs:
        ax.set_xlabel('Time (ns)')
        ax.legend(loc='best')

    fig.tight_layout()
    plt.show()


def check_env():
    from stable_baselines3.common.env_checker import check_env
    env = QuPulseEpisodicEnv(pulse_length=300)
    check_env(env)


if __name__ == '__main__':
    # check_reconstructed_ideal_pulse()
    reconstructed_ideal_pulse_with_environment()
    # check_env()
