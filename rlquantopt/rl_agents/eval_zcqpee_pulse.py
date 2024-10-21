import os
import argparse

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# from pygments.lexer import default
from tqdm import tqdm
from stable_baselines3 import PPO, SAC
from rlquantopt.utils.rl_utils import evaluate_policy
# from rlquantopt.rl_envs.qu_pulse_episodic_env import QuPulseEpisodicEnv as QPEE
from rlquantopt.rl_envs.zcqubits import test_step, test_full, setup_ZCQubits4MKrauss_params


def parse_args():
    parser = argparse.ArgumentParser('RLQuantOpt - evaluating RL agent on ZCQPEE environment')
    # ZCQPEE parameters
    parser.add_argument('pulse_path', type=str, help='Path to pulse CSV file')
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
    test_step(model_params=model_params, pulse_file=pulse_path, save_dir=res_save_dir)
    test_full(model_params=model_params, pulse_file=pulse_path, save_dir=res_save_dir)


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