import os
import argparse
import pandas as pd
import numpy as np
from sb3_contrib import RecurrentPPO, TRPO
from tqdm import tqdm
from stable_baselines3 import PPO, SAC
import matplotlib.pyplot as plt

# from rlquantopt.rl_analysis.check_pulses.check_pulses import pulse_file
from rlquantopt.rl_envs.zc_qpee import ZCQPEE
from rlquantopt.utils.rl_utils import get_pulse_data


def parse_args():
    parser = argparse.ArgumentParser('RLQuantOpt - evaluating RL agent on ZCQPEE environment')
    # ZCQPEE parameters
    parser.add_argument('-m', '--model_zip', type=str, nargs='+', help='Path to model zip file')
    parser.add_argument('-p', '--pulse-csv', default='', type=str, help='Path to pulse CSV file')

    parser.add_argument('-a', '--algo', type=str, default='TRPO', help='Type of RL algorithm')
    parser.add_argument('-x', '--max-t', type=float, default=50., help='')

    parser.add_argument('--save-dir', type=str, default='', help='Path to save evaluation results')
    parser.add_argument('-e', '--env-yml-dir', type=str, default='', help='Directory containing only one yaml file with env arguments. Default, model directory.')
    parser.add_argument('--no-term', action='store_true', help='Keep running episode until the end of the pulse not until termination.')
    parser.add_argument('--clip-best', action='store_true', help='Keep running episode until the end of the pulse, and clip pulse until best performance is reached.')
    parser.add_argument('--clip-to-fid', action='store_true', help='Keep running episode until the fidelity specified by --fid-thresh kwarg is reached.')
    parser.add_argument('--fid-thresh', type=float, default=0.99, help='Used when --clip-to-fid flag is set')
    parser.add_argument('--n-eps', type=int, help='Nb. of evaluation episodes', default=1)
    parser.add_argument('-r', action='store_true', help='Render the episode/s')
    parser.add_argument('-v', '--verbose', type=int, default=2,
                        help='Levels of analysis detail to perform on the evaluation episode.\n'
                             '0 - Save pulse CSV only\n'
                             '1 - Add pulse plots\n'
                             '2 - Add reward metrics\n'
                             '3 - Add reward plot')
    parser.add_argument('--save-suffix', type=str, default='', help='Result files save name suffix. Default ""')

    return parser.parse_args()


def main():
    args = parse_args()
    no_term = args.no_term
    clip_best = args.clip_best
    clip_to_fid = args.clip_to_fid
    save_suffix = args.save_suffix
    fid_thresh = args.fid_thresh
    pulse_file = args.pulse_csv

    # no_term = True
    # clip_best = False
    # clip_to_fid = False
    # save_suffix = args.save_suffix
    # fid_thresh = args.fid_thresh
    # pulse_file = args.pulse_csv

    assert sum([no_term, clip_best, clip_to_fid]) <= 1, 'Only one episode termination option can be set at a time. Either --no-term, --clip-best, or --clip-to-fid.'

    if clip_best:
        no_term = True

    if pulse_file != '':
        tlist, amps = get_pulse_data(pulse_file)
        model_zips = np.arange(1)
    else:
        tlist, amps = None, None

        model_zips = args.model_zip
        # model_zips = 'ZCQPEE_pl-2000_T-300ns_delta_mode-PPO/24-09-24_013133/rl_model_8667136_steps.zip'
        if isinstance(model_zips, (list, tuple)):
            for item in model_zips:
                if not isinstance(item, str):
                    raise argparse.ArgumentTypeError(f'{item} sub-argument is not a string')
        elif isinstance(model_zips, str):
            model_zips = [model_zips]

    # Iterate over every model zip passed as argument
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
        env.DUMP_INFO = True
        print(repr(env))

        mp4_save_dir = os.path.join(save_dir, 'evals')
        if render and not os.path.exists(mp4_save_dir):
            os.makedirs(mp4_save_dir)

        res_save_dir = os.path.join(save_dir, 'results')
        if clip_best:
            res_save_dir = os.path.join(res_save_dir, 'clip_best')
        elif no_term:
            res_save_dir = os.path.join(res_save_dir, 'no_term')
        elif clip_to_fid:
            res_save_dir = os.path.join(res_save_dir, f'clip_to_fid_{fid_thresh:.4f}')

        os.makedirs(res_save_dir, exist_ok=True)

        model = None
        for i in range(n_eps):
            if amps is None:
                model_name = os.path.splitext(os.path.basename(model_zip))[0]
                model = algo.load(model_zip)

            save_name = f"{model_name}_{str(env)}_ep{i}"
            if clip_best:
                save_name += '_clip_best'
            elif no_term:
                save_name += '_no_term'
            elif clip_to_fid:
                save_name += f'_fid_thresh_{fid_thresh:.4f}'
            save_name += save_suffix


            obs = env.reset()[0]
            term = False
            trunc = False
            idx = 0
            rews = []
            if verbose >= 0:
                pbar= tqdm(total=env.pulse_length)
            prev_amp = 0.
            pulse = [prev_amp]
            lstm_states = None
            episode_starts = np.ones((1,), dtype=bool)
            info = None
            # while not (term or trunc):
            while not term:
                if amps is not None:
                    a = amps[idx]
                    if env.delta_mode:
                        if idx > 0:
                            a -= amps[idx - 1]
                elif 'Recurrent' in algo_str:
                    a, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
                else:
                    a = model.predict(obs, deterministic=True)[0]

                idx += 1
                obs, r, term, trunc, info = env.step(a, can_early_term=(not no_term))

                a_trans = env.transform_action(a)
                a_denorm = env.denorm_action(a_trans).reshape(env.N_TIME_STEPS)
                if env.delta_mode:
                    amp_abs_denorm = prev_amp + np.cumsum(a_denorm)
                else:
                    amp_abs_denorm = a_denorm
                prev_amp = amp_abs_denorm[-1]
                pulse.extend(amp_abs_denorm)

                rews.append(r)
                if verbose >= 0:
                    pbar.update(env.N_TIME_STEPS)

                if clip_to_fid and info['fidelity'] >= fid_thresh:
                    break

                if idx * env.dt >= args.max_t:
                    break

            final_realised_gate = info['realised_gate']

            # Prepare time-axis and pulse amplitude data for plotting
            infidelities = 1 - np.array(env.fidelities)

            n_steps_coarse = len(infidelities)
            tlist_coarse = np.arange(n_steps_coarse) * env.dt * env.N_TIME_STEPS

            n_steps_fine = len(pulse)
            tlist_fine = np.arange(n_steps_fine) * env.dt

            idx_best_coarse = np.argmax(rews)
            t_best = tlist_coarse[idx_best_coarse]
            idx_best_fine = np.argmin(np.abs(np.subtract(tlist_fine, t_best)))

            print(f"n_steps_coarse={n_steps_coarse}, idx_best_coarse={idx_best_coarse}, idx_best_fine={idx_best_fine}")

            if clip_best:
                ep_len_coarse = idx_best_coarse + 1
                ep_len_fine = idx_best_fine + 1
            else:
                ep_len_coarse = n_steps_coarse
                ep_len_fine = n_steps_fine

            tlist_coarse = tlist_coarse[:ep_len_coarse]
            tlist_fine = tlist_fine[:ep_len_fine]
            pulse = pulse[:ep_len_fine]
            infidelities = infidelities[:ep_len_coarse]
            concurrences = np.array(env.concurrences[:ep_len_coarse])
            unitarities = np.array(env.unitarities[:ep_len_coarse])
            #
            # data = dict(concurrences=concurrences,
            #             unitarities=unitarities,
            #             infidelities=infidelities,
            #             tlist_coarse=tlist_coarse,)
            # pd.DataFrame.from_dict(data).to_csv('pulse_metrics_for_mkrauss.csv', index=False)
            #
            # exit(23)

            max_fid = env.fidelities[idx_best_coarse] * 100.
            idx_min_infidelity = np.argmin(infidelities)
            idx_max_concurrence = np.argmax(concurrences)
            idx_max_unitarity = np.argmax(unitarities)
            metrics_str = lambda j: f'Rew={rews[j]:.2f} => C={concurrences[j]:.4f}; U={unitarities[j]:.4f}; (1-F)={infidelities[j]:.4f} @ T={tlist_coarse[j]:.2f}ns; PL={j+1:d}'
            title = (f'Best Reward: {metrics_str(idx_best_coarse)}\n'
                     f'Best Concurrence: {metrics_str(idx_max_concurrence)}\n'
                     f'Best Unitarity: {metrics_str(idx_max_unitarity)}\n'
                     f'Best Fidelity: {metrics_str(idx_min_infidelity)}'
                     )
            if verbose > 0:
                print(f'Model: {model_zip}; Ep: {i}')
                print(title + '\n')

            '''
            Pulse plot
            '''
            if verbose >= 1:
                fig, axs = plt.subplots(2, figsize=(15, 10))
                fig.suptitle(title)
                ax = axs[0]
                ax.plot(tlist_fine, pulse, c='k', lw=0.2)
                ax.set_xlabel('Time [ns]')
                ax.set_ylabel('Pulse amplitude')
                ax.grid(which='major', linestyle='--', color='grey', linewidth=1)
                ax.grid(which='minor', linestyle=':', color='lightgrey', linewidth=0.5)
                # ax.axvline(t_best, color='g', ls='--', linewidth=0.5)
                try:
                    ax.axvline(tlist_fine[idx_best_coarse * env.N_TIME_STEPS], c='m', linestyle='solid', linewidth=2,
                               alpha=0.6,
                               label='Best Reward')
                    ax.axvline(tlist_fine[idx_max_concurrence * env.N_TIME_STEPS], c='b', linestyle='--', linewidth=1.2,
                               label='Best Concurrence')
                    ax.axvline(tlist_fine[idx_max_unitarity * env.N_TIME_STEPS], c='g', linestyle='--', linewidth=1.2,
                               label='Best Unitarity')
                except Exception as e:
                    print(e)
                ax.legend(loc='best')

                ax_spectrum = axs[1]
                # Perform FFT on pulse
                famps = np.fft.fft(pulse)
                flist = np.fft.fftfreq(len(famps), d=env.dt)

                # Fix freq axis
                famps = np.fft.fftshift(famps)
                flist = np.fft.fftshift(flist)

                # Decompose complex amplitudes
                fabsamps = np.abs(famps)
                phaseamps = np.angle(famps)

                # Complicated & boring spike calculation and plotting in red with GHz value of spike
                _log10 = np.log10(fabsamps)
                diff_log10 = np.diff(_log10, prepend=0)
                diff_log10 = np.where(diff_log10 > 0.3, diff_log10, 0)
                asign = np.sign(diff_log10)
                signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
                spike_idces = [item for item in np.arange(len(fabsamps)) * signchange if item > 0]
                _spike_freqs = [flist[item] for item in spike_idces]
                _spike_amps = [fabsamps[item] for item in spike_idces]
                spike_freqs, spike_amps = [], []
                for j, (f, a) in enumerate(sorted(zip(_spike_freqs, _spike_amps), key=lambda x: x[1], reverse=True)):
                    if j > 15:
                        break
                    spike_freqs.append(f)
                    spike_amps.append(a)
                ax_spectrum.plot(flist[1:], fabsamps[1:], c='b', lw=0.5)
                ax_spectrum.scatter(spike_freqs, spike_amps, marker='.', color='r')
                text_pad_v = max(fabsamps) / 50.
                for freq, amp in zip(spike_freqs, spike_amps):
                    ax_spectrum.text(freq, amp + text_pad_v, f'{freq:.2f}GHz', fontsize=5, ha='center', color='r')

                # Set axes labels & plot phase
                ax_spectrum.set_xlabel('Frequency [GHz]')
                ax_spectrum.set_ylabel('Amplitude')

                ax_phase = ax_spectrum.twinx()
                ax_phase.plot(flist, phaseamps, c='grey', ls=':', lw=0.5)
                ax_phase.set_ylabel('Phase')
                # plt.show()

                save_path = os.path.join(res_save_dir, f'{save_name}_Fmax-{max_fid:.2f}_pulse.pdf')
                if verbose > 2:
                    print(f'Saving result to: {save_path}')
                fig.tight_layout()
                fig.savefig(save_path)

            '''
            Reward metrics
            '''
            if verbose >= 2:
                fig, ax_metrics = plt.subplots(figsize=(15, 10))
                fig.suptitle(title, fontsize=12)

                # axx.plot(tlist_coarse, rews, c='tab:orange', label='Reward')
                # axx.plot(tlist_coarse, fidelities, c='m', label='Fidelity')
                ax_metrics.plot(tlist_coarse, 1 - concurrences, c='b', label='1 - Concurrences')
                ax_metrics.plot(tlist_coarse, 1 - unitarities, c='g', label='1 - Unitarities')
                ax_metrics.set_yscale('symlog', linthresh=1e-6)
                ax_metrics.legend(loc='lower right')
                ax_metrics.set_ylabel('Reward metrics')
                ax_metrics.set_ylim(0, 1)

                # axx.axhline(max_fid, c='g', linestyle='--', linewidth=0.5)
                ax_metrics.axvline(tlist_coarse[idx_best_coarse], c='m', linestyle='solid', linewidth=2, alpha=0.6, label='Best Reward')
                ax_metrics.axvline(tlist_coarse[idx_max_concurrence], c='b', linestyle='--', linewidth=0.8, label='Best Concurrence')
                ax_metrics.axvline(tlist_coarse[idx_max_unitarity], c='g', linestyle='--', linewidth=0.8, label='Best Unitarity')

                ax_metrics.legend(loc='lower right')

                # ax_rew = ax_metrics.twinx()
                # ax_rew.spines['right'].set_position(("axes", 1.14))
                # rew_c = 'tab:orange'
                # ax_rew.plot(tlist_coarse, rews, c=rew_c)
                # ax_rew.set_ylabel('Reward', color=rew_c)
                # ax_rew.tick_params(axis='y', colors=rew_c, color=rew_c)
                # ax_rew.spines['right'].set_color(rew_c)

                ax_infid = ax_metrics.twinx()
                infid_c = 'm'
                ax_infid.plot(tlist_coarse, infidelities, c=infid_c, ls='dotted')
                ax_infid.set_xlabel('Time [ns]')
                ax_infid.set_ylabel('Infidelity (%)', color=infid_c)
                ax_infid.set_yscale('log')
                ax_infid.grid(which='major', linestyle='--', color='grey', linewidth=1)
                ax_infid.grid(which='minor', linestyle=':', color='lightgrey', linewidth=0.5)
                ax_infid.spines['right'].set_color(infid_c)
                ax_infid.tick_params(axis='y', colors=infid_c, color=infid_c)

                save_path = os.path.join(res_save_dir, f'{save_name}_Fmax-{max_fid:.2f}_fid_vs_infid_plot.pdf')
                if verbose > 2:
                    print(f'Saving result to: {save_path}')
                fig.tight_layout()
                fig.savefig(save_path)

            '''
            Reward plot
            '''
            if verbose >= 3:
                fig, ax = plt.subplots(figsize=(15,10))
                fig.suptitle(title)
                ax.plot(tlist_coarse, rews[:ep_len_coarse], c='tab:orange')
                ax.set_xlabel('Time [ns]')
                ax.set_ylabel('Reward')
                ax.grid(which='major', linestyle='--', color='grey', linewidth=1)
                ax.grid(which='minor', linestyle=':', color='lightgrey', linewidth=0.5)
                ax.axvline(t_best, color='g', ls='--', linewidth=0.5)

                save_path = os.path.join(res_save_dir, f'{save_name}_Fmax-{max_fid:.2f}_rew_plot.pdf')
                if verbose > 2:
                    print(f'Saving result to: {save_path}')
                fig.savefig(save_path)

            # Save pulse to CSV
            dat = pd.DataFrame()
            dat['amplist'] = pulse
            dat['tlist'] = tlist_fine
            dat.to_csv(os.path.join(res_save_dir, f'{save_name}.csv'))

            dat = pd.DataFrame()
            dat['concurrences'] = concurrences
            dat['unitarities'] = unitarities
            dat['rewards'] = rews
            dat['tlist'] = tlist_coarse
            dat.to_csv(os.path.join(res_save_dir, f'{save_name}_metrics.csv'))



            if render:
                mp4_save_path = os.path.join(mp4_save_dir, f'{save_name}_Fmax-{max_fid:.2f}.mp4')
                print(f'Saving animations in: {mp4_save_path}')
                env.render(save_path=mp4_save_path)

            # exit(23)


if __name__ == '__main__':
    main()
