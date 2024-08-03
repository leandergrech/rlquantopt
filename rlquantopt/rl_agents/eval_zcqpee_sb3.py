import os
import glob
import argparse
import pandas as pd
import numpy as np
from sb3_contrib import RecurrentPPO, TRPO
# from pygments.lexer import default
from tqdm import tqdm
from stable_baselines3 import PPO, SAC
import matplotlib.pyplot as plt
from rlquantopt.utils.rl_utils import evaluate_policy
# from rlquantopt.rl_envs.qu_pulse_episodic_env import QuPulseEpisodicEnv as QPEE
from rlquantopt.rl_envs.zc_qpee import ZCQPEE


def parse_args():
    parser = argparse.ArgumentParser('RLQuantOpt - evaluating RL agent on ZCQPEE environment')
    # ZCQPEE parameters
    parser.add_argument('model_zip', type=str, nargs='+', help='Path to model zip file')
    parser.add_argument('-a', '--algo', type=str, default='RecurrentPPO', help='Type of RL algorithm')
    parser.add_argument('--save-dir', type=str, default='', help='Path to save evaluation results')
    parser.add_argument('-e', '--env-yml-dir', type=str, default='', help='Directory containing only one yaml file with env arguments. Default, model directory.')
    parser.add_argument('--no-term', action='store_true', help='Keep running episode until the end of the pulse not until termination.')
    parser.add_argument('--clip-best', action='store_true', help='Keep running episode until the end of the pulse, and clip pulse until best performance is reached.')
    parser.add_argument('--n-eps', type=int, help='Nb. of evaluation episodes', default=1)
    parser.add_argument('-r', action='store_true', help='Render the episode/s')
    parser.add_argument('-v', '--verbose', type=int, default=2,
                        help='Levels of analysis detail to perform on the pulse.\n'
                             'Level 0 - no analysis\n'
                             'Level 1 - plot rewards\n'
                             'Level 2 - plot rewards & fidelity/infidelities\n'
                             'Level 3 - plot rewards & a holistic projection of the synchronicity of felicity. Need to buy tarot cards.')

    return parser.parse_args()


def main():
    args = parse_args()
    no_term = args.no_term
    if clip_best := args.clip_best:
        no_term = True

    # assert not (no_term and clip_best), "You can either evaluate until end of pulse (--no-term) OR until best performance observed (--clip-best)."

    model_zips = args.model_zip
    if isinstance(model_zips, (list, tuple)):
        for item in model_zips:
            if not isinstance(item, str):
                raise argparse.ArgumentTypeError(f'{item} sub-argument is not a string')
    elif isinstance(model_zips, str):
        model_zips = [model_zips]

    for model_zip in sorted(model_zips):
        model_dir = os.path.dirname(model_zip)
        if (save_dir:= args.save_dir) == '':
            save_dir = os.path.dirname(model_zip)
        n_eps = args.n_eps
        render = args.r
        verbose = args.verbose

        algo_str = args.algo
        if algo_str == 'RecurrentPPO':
            algo = RecurrentPPO
        elif algo_str == 'PPO':
            algo = PPO
        elif algo_str == 'TRPO':
            algo = TRPO
        else:
            raise NotImplementedError

        if (env_yml_dir := args.env_yml_dir) == '':
            env_yml_dir = model_dir
        elif env_yml_dir.startswith('../'):
            env_yml_dir = os.path.abspath(os.path.join(model_dir, env_yml_dir))

        env_yml_path = None
        for item in os.listdir(env_yml_dir):
            if item.endswith('.yml') or item.endswith('.yaml'):
                env_yml_path = os.path.join(env_yml_dir, item)
                break
        if env_yml_path is None:
            raise Exception('No ZCQPEE environment configuration was found')
        env = ZCQPEE.from_yaml(env_yml_path)

        mp4_save_dir = os.path.join(save_dir, 'evals')
        if render and not os.path.exists(mp4_save_dir):
            os.makedirs(mp4_save_dir)

        res_save_dir = os.path.join(save_dir, 'results')
        if clip_best:
            res_save_dir = os.path.join(res_save_dir, 'clip_best')
        elif no_term:
            res_save_dir = os.path.join(res_save_dir, 'no_term')
        if not os.path.exists(res_save_dir):
            os.makedirs(res_save_dir)

        for i in range(n_eps):
            model_name = os.path.splitext(os.path.basename(model_zip))[0]
            save_name = f"{model_name}_{str(env)}_ep{i}"
            if clip_best:
                save_name += '_clip_best'
            elif no_term:
                save_name += '_no_term'

            model = algo.load(model_zip)

            obs = env.reset()[0]
            term = False
            trunc = False
            idx = 0
            rews = []
            if verbose > 3:
                pbar= tqdm(total=env.pulse_length)
            prev_amp = 0.
            pulse = [prev_amp]
            lstm_states = None
            episode_starts = np.ones((1,), dtype=bool)
            while not (term or trunc):
                idx += 1
                if 'Recurrent' in algo_str:
                    a, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
                else:
                    a = model.predict(obs, deterministic=True)[0]
                # a += np.random.normal(0, 1e-3, env.n_act)
                obs, r, term, trunc, info = env.step(a, can_early_term=(not no_term))

                a_denorm = env.denorm_action(a)[0]
                if env.delta_mode:
                    prev_amp += a_denorm
                else:
                    prev_amp = a_denorm
                pulse.append(prev_amp)

                rews.append(r)
                if verbose > 3:
                    pbar.update(1)

            max_fid = max(env.fidelities) * 100

            infidelities = 1 - np.array(env.fidelities)
            infidelities *= 100.

            idx_best = np.argmin(infidelities)
            t_best = env.tlist[idx_best]

            if clip_best:
                ep_len = idx_best + 1
            else:
                ep_len = len(infidelities)

            title = f'Best infidelity {min(infidelities)/100.:.3e}\nFidelity={max_fid:.3f}% @ T={t_best:.1f}ns PL={idx_best + 1} Δt={t_best/(idx_best + 1):.3f}ns'
            if verbose > 0:
                print(f'Model: {model_zip}; Ep: {i}')
                print(title + '\n')

            if verbose > 1:
                fig, ax = plt.subplots()
                fig.suptitle(title)
                ax.plot(env.tlist[:ep_len], infidelities[:ep_len], c='k')
                ax.set_xlabel('Time [ns]')
                ax.set_ylabel('Infidelity (%)')
                ax.set_yscale('log')
                ax.grid(which='major', linestyle='--', color='grey', linewidth=1)
                ax.grid(which='minor', linestyle=':', color='lightgrey', linewidth=0.5)

                axx = ax.twinx()
                axx.plot(env.tlist[:ep_len], np.array(env.fidelities[:ep_len]) * 100., c='g')
                axx.set_ylabel('Fidelity (%)', color='g')
                axx.spines['right'].set_color('g')
                axx.tick_params(axis='y', colors='g', color='g')

                axx.axhline(max_fid, c='g', linestyle='--', linewidth=0.5)
                axx.axvline(env.tlist[np.argmax(env.fidelities)], c='g', linestyle='--', linewidth=0.5)

                save_path = os.path.join(res_save_dir, f'{save_name}_Fmax-{max_fid:.2f}_fid_vs_infid_plot.pdf')
                if verbose > 2:
                    print(f'Saving result to: {save_path}')
                fig.savefig(save_path)

            if verbose > 0:
                # print(f'{os.sep}'.join(model_zip.split(os.sep)[-3:]) + ': ', end='')
                # print(f'Max. Fidelity = {max_fid:.2f}%')
                fig, ax = plt.subplots()
                fig.suptitle(title)
                ax.plot(env.tlist[:ep_len], rews[:ep_len], c='tab:orange')
                ax.set_xlabel('Time [ns]')
                ax.set_ylabel('Reward')
                ax.grid(which='major', linestyle='--', color='grey', linewidth=1)
                ax.grid(which='minor', linestyle=':', color='lightgrey', linewidth=0.5)
                ax.axvline(t_best, color='g', ls='--', linewidth=0.5)
                save_path = os.path.join(res_save_dir, f'{save_name}_rew_plot.pdf')
                if verbose > 2:
                    print(f'Saving result to: {save_path}')
                fig.savefig(save_path)

                fig, axs = plt.subplots(2, figsize=(15,10))
                fig.suptitle(title)
                ax = axs[0]
                ax.plot(env.tlist[:ep_len], pulse[:ep_len], c='k', lw=0.2)
                ax.set_xlabel('Time [ns]')
                ax.set_ylabel('Pulse amplitude')
                ax.grid(which='major', linestyle='--', color='grey', linewidth=1)
                ax.grid(which='minor', linestyle=':', color='lightgrey', linewidth=0.5)
                ax.axvline(t_best, color='g', ls='--', linewidth=0.5)

                ax = axs[1]
                famps = np.fft.fft(pulse)[1:]
                flist = np.linspace(-1/(2*env.dt), 1/(2*env.dt), len(famps))

                fabsamps = np.abs(famps)
                faseamps = np.angle(famps)
                thresh = 0.1 * np.max(fabsamps)
                spike_idces = np.where(fabsamps > thresh)[0].astype(int)
                print(len(fabsamps))
                spike_freqs = [flist[item] for item in spike_idces]
                spike_amps = [fabsamps[item] for item in spike_idces]
                ax.plot(flist[1:], fabsamps[1:], c='b', lw=0.5)
                ax.scatter(spike_freqs, spike_amps, marker='^', color='r')
                for freq, amp in zip(spike_freqs, spike_amps):
                    ax.text(freq, amp, f'{freq:.2f}GHz', fontsize=8, ha='center', color='r')
                ax.set_xlabel('Frequency [GHz]')
                ax.set_ylabel('Amplitude')
                axx = ax.twinx()
                axx.plot(flist, faseamps, c='grey', ls=':', lw=0.5)
                axx.set_ylabel('Phase')
                # plt.show()

                save_path = os.path.join(res_save_dir, f'{save_name}_pulse.pdf')
                if verbose > 2:
                    print(f'Saving result to: {save_path}')
                fig.savefig(save_path)

            infidelities = 1 - np.array(env.fidelities)
            infidelities *= 100.

            tlist = np.arange(len(pulse)) * env.dt
            # pulse = [env.denorm_action(item) for item in pulse]

            # Save pulse to CSV
            dat = pd.DataFrame()
            dat['amplist'] = pulse[:ep_len]
            dat['tlist'] = tlist[:ep_len]
            dat.to_csv(os.path.join(res_save_dir, f'{save_name}.csv'))

            if render:
                mp4_save_path = os.path.join(mp4_save_dir, f'{save_name}.mp4')
                print(f'Saving animations in: {mp4_save_path}')
                env.render(save_path=mp4_save_path)

            exit(23)


if __name__ == '__main__':
    main()