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
from rlquantopt.rl_envs.zc_qpee import ZCQPEE


def parse_args():
    parser = argparse.ArgumentParser('RLQuantOpt - evaluating RL agent on ZCQPEE environment')
    # ZCQPEE parameters
    parser.add_argument('--pulse-path', type=str, help='Path to model zip file')
    parser.add_argument('--save-dir', type=str, default='', help='Path to save evaluation results')
    parser.add_argument('--n_eps', type=int, help='Nb. of evaluation episodes', default=1)
    parser.add_argument('-r', action='store_true', help='Render the episode/s')
    parser.add_argument('--verbosity', type=int, default=2,
                        help='Levels of analysis detail to perform on the pulse.\n'
                             'Level 0 - no analysis\n'
                             'Level 1 - plot rewards\n'
                             'Level 2 - plot rewards & fidelity/infidelities\n'
                             'Level 3 - plot rewards & a holistic projection of the synchronicity of felicity. Need to buy tarot cards.')

    return parser.parse_args()


def gen_env_yaml(pulse_path, save_dir, ret_pulse=True, a_scale=1):
    data = pd.read_csv(pulse_path)
    tkey, vkey = data.keys()[1:]
    tlist = data[tkey]
    pulse = data[vkey]

    env_kw = dict(delta_mode=False,
                  T=max(tlist),
                  pulse_length=len(pulse) - 1,
                  optimised_pulse_path=os.path.abspath(pulse_path),
                  action_scaling={'z': a_scale})
    env = ZCQPEE(**env_kw)
    save_path = os.path.join(save_dir, f'{str(env)}.yml')
    env.to_yaml(save_path)

    if ret_pulse:
        return save_path, dict(tlist=tlist, pulse=pulse)
    else:
        return save_path


def main():
    args = parse_args()
    n_eps = args.n_eps
    render = args.r
    verbosity = args.verbosity
    # pulse_path = args.pulse_path
    # pulse_path = '../rl_envs/configs/data_mkrauss/pulse_bad_guess.csv'
    pulse_path = '../rl_envs/configs/data_lilmc/data_lilmc.csv'
    pulse_dir = os.path.abspath(os.path.dirname(pulse_path))
    pulse_name = os.path.splitext(os.path.basename(pulse_path))[0]

    if (save_dir := args.save_dir) == '':
        save_dir = os.path.dirname(pulse_path)
    mp4_save_dir = os.path.join(save_dir, 'evals')
    res_save_dir = os.path.join(save_dir, 'results')
    for d in (mp4_save_dir, res_save_dir):
        if not os.path.exists(d):
            os.makedirs(d)

    env_yml_path, pulse = gen_env_yaml(pulse_path, pulse_dir, ret_pulse=True)
    tkey, vkey = pulse.keys()
    env = ZCQPEE.from_yaml(env_yml_path)
    for i in range(n_eps):
        save_name = f"{pulse_name}_{str(env)}_ep{i}"

        env.reset()
        term = False
        trunc = False
        idx = 0
        rews = []
        pbar= tqdm(total=env.pulse_length)
        while not (term or trunc):
            a = env.norm_action(np.array([pulse[vkey][idx]]))
            obs, r, term, trunc, info = env.step(a)

            rews.append(r)
            pbar.update(1)
            idx += 1

        max_fid = max(env.fidelities) * 100
        print(f'Max fidelity: {max_fid:.2f}%')

        infidelities = 1 - np.array(env.fidelities)
        infidelities *= 100.

        if verbosity > 1:
            fig, ax = plt.subplots()
            ax.plot(env.tlist, infidelities, c='k')
            ax.set_xlabel('Time [ns]')
            ax.set_ylabel('Infidelity (%)')
            ax.set_yscale('log')
            ax.grid(which='major', linestyle='--', color='grey', linewidth=1)
            ax.grid(which='minor', linestyle=':', color='lightgrey', linewidth=0.5)
            axx = ax.twinx()
            axx.plot(env.tlist, np.array(env.fidelities) * 100., c='g')
            axx.set_ylabel('Fidelity (%)', color='g')
            axx.spines['right'].set_color('g')
            axx.tick_params(axis='y', colors='g', color='g')

            fig.suptitle(f'Max fidelity = {max_fid:.2f}%')
            axx.axhline(max_fid, c='g', linestyle='--', linewidth=0.5)
            axx.axvline(np.argmax(env.fidelities), c='g', linestyle='--', linewidth=0.5)
            save_path = os.path.join(res_save_dir, 'fid_vs_infid_plot.pdf')
            fig.savefig(save_path)
            
        if verbosity > 0:
            fig, ax = plt.subplots()
            ax.plot(env.tlist, rews, c='tab:orange')
            ax.set_xlabel('Time [ns]')
            ax.set_ylabel('Reward')
            ax.grid(which='major', linestyle='--', color='grey', linewidth=1)
            ax.grid(which='minor', linestyle=':', color='lightgrey', linewidth=0.5)
            save_path = os.path.join(res_save_dir, 'rew_plot.pdf')
            fig.savefig(save_path)

            fig, ax = plt.subplots()
            ax.plot(pulse[tkey], pulse[vkey], c='k', lw=0.5)
            ax.set_xlabel('Time [ns]')
            ax.set_ylabel('Pulse amplitude')
            ax.grid(which='major', linestyle='--', color='grey', linewidth=1)
            ax.grid(which='minor', linestyle=':', color='lightgrey', linewidth=0.5)
            save_path = os.path.join(res_save_dir, 'pulse.pdf')
            fig.savefig(save_path)

        if render:
            mp4_save_path = os.path.join(mp4_save_dir, f'{save_name}.mp4')
            print(f'Saving animations in: {mp4_save_path}')
            env.render(save_path=mp4_save_path)


if __name__ == '__main__':
    main()