import os
import argparse

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# from pygments.lexer import default
from tqdm import tqdm
from stable_baselines3 import PPO, SAC

from rlquantopt.rl_analysis.check_pulses.check_pulses import pulse_file
from rlquantopt.rl_envs.zc_qpee import ZCQPEE
from rlquantopt.utils.rl_utils import evaluate_policy, get_pulse_data
# from rlquantopt.rl_envs.qu_pulse_episodic_env import QuPulseEpisodicEnv as QPEE
from rlquantopt.rl_envs.zcqubits import test_step, test_full, setup_ZCQubits4MKrauss_params


def parse_args():
    parser = argparse.ArgumentParser('RLQuantOpt - evaluating RL agent on ZCQPEE environment')
    # ZCQPEE parameters
    parser.add_argument('pulse_path', type=str, help='Path to pulse CSV file')
    parser.add_argument('-o', '--act-poly-order', type=int, default=1, help='Order of pulse transformation polynomial')
    parser.add_argument('--save-dir', type=str, default='', help='Path to save evaluation results')

    return parser.parse_args()


def main():
    args = parse_args()
    pulse_path = args.pulse_path
    pulse_dir = os.path.abspath(os.path.dirname(pulse_path))

    if (save_dir := args.save_dir) == '':
        save_dir = pulse_dir

    res_save_dir = os.path.join(save_dir, 'pulse_check')
    if not os.path.exists(res_save_dir):
        os.makedirs(res_save_dir)

    model_params = {}
    model_params = setup_ZCQubits4MKrauss_params(**model_params)

    env_temp = ZCQPEE(act_poly_order=args.act_poly_order)

    def transform_pulse(pulse):
        pulse_diffs = np.diff(pulse, prepend=0)
        pulse_diffs = env_temp.transform_action(pulse_diffs)
        return np.cumsum(pulse_diffs)

    test_step(model_params=model_params, pulse_file=pulse_path, save_dir=res_save_dir, transform_pulse=transform_pulse)
    test_full(model_params=model_params, pulse_file=pulse_path, save_dir=res_save_dir, transform_pulse=transform_pulse)

    tlist, pulse = get_pulse_data(pulse_path)

    # print(f'{os.sep}'.join(model_zip.split(os.sep)[-3:]) + ': ', end='')
    # print(f'Max. Fidelity = {max_fid:.2f}%')

    # Plot pulse deteails
    pulse_name = os.path.splitext(os.path.basename(pulse_file))[0]
    save_path = os.path.join(res_save_dir, f'{pulse_name}_spectrum.pdf')


    fig, axs = plt.subplots(2, figsize=(15, 10))
    fig.suptitle(pulse_path)
    ax = axs[0]
    ax.plot(tlist, pulse, c='k', lw=0.2)
    ax.set_xlabel('Time [ns]')
    ax.set_ylabel('Pulse amplitude')
    ax.grid(which='major', linestyle='--', color='grey', linewidth=1)
    ax.grid(which='minor', linestyle=':', color='lightgrey', linewidth=0.5)

    ax = axs[1]

    famps = np.fft.fft(pulse)
    fabsamps = np.abs(famps)
    # fabsamps = np.log10(famps)

    flist = np.fft.fftfreq(len(famps), d=tlist[1])

    flist = np.fft.fftshift(flist)
    famps = np.fft.fftshift(famps)
    fabsamps = np.fft.fftshift(fabsamps)
    faseamps = np.angle(famps)

    _log10 = np.log10(fabsamps)
    diff_log10 = np.diff(_log10, prepend=0)
    diff_log10 = np.where(diff_log10 > 0.3, diff_log10, 0)
    # cum_change = np.cumsum(diff_log10)
    # spike_idces = np.where(cum_change > 2)[0]
    asign = np.sign(diff_log10)
    signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)

    # spike_idces = np.where(np.diff(_log10) > 0.5)[0]
    # spike_idces = [idx for idx in spike_idces if _log10[idx] > 0]
    spike_idces = [item for item in np.arange(len(fabsamps)) * signchange if item > 0]
    # spike_idces = np.concatenate([np.arange(3) - 1 + idx for idx in spike_idces])
    # spike_idces = np.where(np.log10(np.abs(np.diff(fabsamps, prepend=famps[0]))) > 1)[0].astype(int)

    # thresh = 0.25 * np.max(fabsamps[3:])
    # spike_idces = np.where(fabsamps > thresh)[0].astype(int)

    spike_freqs = [flist[item] for item in spike_idces]
    spike_amps = [fabsamps[item] for item in spike_idces]
    ax.plot(flist[1:], fabsamps[1:], c='b', lw=0.5)
    ax.scatter(spike_freqs, spike_amps, marker='^', color='r')
    for freq, amp in zip(spike_freqs, spike_amps):
        ax.text(freq, amp, f'{freq:.2f}GHz', fontsize=8, ha='center', color='r')
    ax.set_xlabel('Frequency [GHz]')
    ax.set_ylabel('Amplitude')
    axx = ax.twinx()
    ax.set_yscale('log')
    axx.plot(flist, faseamps, c='grey', ls=':', lw=0.5)
    axx.set_ylabel('Phase')
    print(save_path)
    fig.savefig(save_path)
    # plt.show()


    # max_fid = max(env.fidelities) * 100
    # print(f'Max fidelity: {max_fid:.2f}%')
    #
    # infidelities = 1 - np.array(env.fidelities)
    # infidelities *= 100.
    #
    # if verbose > 1:
    #     fig, ax = plt.subplots()
    #     ax.plot(env.tlist, infidelities, c='k')
    #     ax.set_xlabel('Time [ns]')
    #     ax.set_ylabel('Infidelity (%)')
    #     ax.set_yscale('log')
    #     ax.grid(which='major', linestyle='--', color='grey', linewidth=1)
    #     ax.grid(which='minor', linestyle=':', color='lightgrey', linewidth=0.5)
    #     axx = ax.twinx()
    #     axx.plot(env.tlist, np.array(env.fidelities) * 100., c='g')
    #     axx.set_ylabel('Fidelity (%)', color='g')
    #     axx.spines['right'].set_color('g')
    #     axx.tick_params(axis='y', colors='g', color='g')
    #
    #     fig.suptitle(f'Max fidelity = {max_fid:.2f}%')
    #     axx.axhline(max_fid, c='g', linestyle='--', linewidth=0.5)
    #     axx.axvline(np.argmax(env.fidelities), c='g', linestyle='--', linewidth=0.5)
    #
    #     save_path = os.path.join(res_save_dir, f'{save_name}_Fmax-{max_fid:.1f}_fid_vs_infid_plot.pdf')
    #     if verbose > 2:
    #         print(f'Saving result to: {save_path}')
    #     fig.savefig(save_path)
    #
    # if verbose > 0:
    #     fig, ax = plt.subplots()
    #     ax.plot(env.tlist, rews, c='tab:orange')
    #     ax.set_xlabel('Time [ns]')
    #     ax.set_ylabel('Reward')
    #     ax.grid(which='major', linestyle='--', color='grey', linewidth=1)
    #     ax.grid(which='minor', linestyle=':', color='lightgrey', linewidth=0.5)
    #     save_path = os.path.join(res_save_dir, f'{save_name}_rew_plot.pdf')
    #     if verbose > 2:
    #         print(f'Saving result to: {save_path}')
    #     fig.savefig(save_path)
    #
    #     fig, ax = plt.subplots()
    #     ax.plot(pulse[tkey], pulse[vkey], c='k', lw=0.5)
    #     ax.set_xlabel('Time [ns]')
    #     ax.set_ylabel('Pulse amplitude')
    #     ax.grid(which='major', linestyle='--', color='grey', linewidth=1)
    #     ax.grid(which='minor', linestyle=':', color='lightgrey', linewidth=0.5)
    #
    #     save_path = os.path.join(res_save_dir, f'{save_name}_pulse.pdf')
    #     if verbose > 2:
    #         print(f'Saving result to: {save_path}')
    #     fig.savefig(save_path)
    #
    # if render:
    #     mp4_save_path = os.path.join(mp4_save_dir, f'{save_name}.mp4')
    #     print(f'Saving animations in: {mp4_save_path}')
    #     env.render(save_path=mp4_save_path)


if __name__ == '__main__':
    main()