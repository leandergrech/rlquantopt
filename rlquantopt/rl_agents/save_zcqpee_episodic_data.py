import os
import argparse
import pandas as pd
import numpy as np
from sb3_contrib import RecurrentPPO, TRPO
from tqdm import tqdm
from stable_baselines3 import PPO, SAC
import matplotlib.pyplot as plt
import torch as th

# from rlquantopt.rl_analysis.check_pulses.check_pulses import pulse_file
from rlquantopt.rl_envs.zc_qpee import ZCQPEE
from rlquantopt.utils.rl_utils import get_pulse_data



def main():
    # args = parse_args()
    no_term = True
    clip_best = False
    clip_to_fid = False
    save_suffix = ''
    fid_thresh = 0.99
    model_zips = ['/home/leander/code/rlquantopt/rlquantopt/rl_agents/ZCQPEE_pl-1000_T-50ns_delta_mode-TRPO/05-12-24_201634/rl_model_12566528_steps.zip']
    # model_zips = ['/home/leander/code/rlquantopt/rlquantopt/rl_agents/ZCQPEE_pl-1000_T-50ns_delta_mode-TRPO/05-12-24_201634/rl_model_6758400_steps.zip']
    # model_zips = ['/home/leander/code/rlquantopt/rlquantopt/rl_agents/ZCQPEE_pl-1000_T-50ns_delta_mode-TRPO/05-12-24_201634/rl_model_8192_steps.zip']
    max_t = 50
    n_eps = 25
    render = False
    verbose = 2
    tmax = 20

    # no_term = True
    # clip_best = False
    # clip_to_fid = False
    # save_suffix = args.save_suffix
    # fid_thresh = args.fid_thresh
    # pulse_file = args.pulse_csv

    assert sum([no_term, clip_best, clip_to_fid]) <= 1, 'Only one episode termination option can be set at a time. Either --no-term, --clip-best, or --clip-to-fid.'

    if clip_best:
        no_term = True

    tlist, amps = None, None

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

        save_dir = os.path.dirname(model_zip)
        algo_str = 'TRPO'
        if algo_str == 'RecurrentPPO':
            algo = RecurrentPPO
        elif algo_str == 'PPO':
            algo = PPO
        elif algo_str == 'TRPO':
            algo = TRPO
        else:
            raise NotImplementedError

        env_yml_dir = model_dir

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
        pulses = []
        rewards = []
        model = algo.load(model_zip)
        for i in range(n_eps):
            model_name = os.path.splitext(os.path.basename(model_zip))[0]

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

            all_actions = [0.]
            policy = model.policy
            qobjs = []
            while not term:
                obs_tensor, vectorized_env = policy.obs_to_tensor(obs)
                with th.no_grad():
                    a_dist = policy.get_distribution(obs_tensor).distribution
                a_mean = a_dist.mean.cpu().numpy()
                a_scale = a_dist.scale.cpu().numpy()
                # a = a_dist.get_actions(deterministic=False)
                # a = np.random.normal(a_mean, a_scale)
                a = a_mean

                # Convert to numpy, and reshape to the original action shape
                # a = a.cpu().numpy().reshape((-1, env.N_TIME_STEPS))
                # a = model.predict(obs, deterministic=False)[0]
                # a_dist = model.policy.get_distribution(th.tensor(obs))
                # a = a_dist.get_actions(deterministic=False)

                all_actions.extend(list(a.squeeze()).copy())
                idx += 1
                obs, r, term, trunc, info = env.step(a, can_early_term=(not no_term))

                qobjs.append(info['step_states'])

                rews.append(r)
                if verbose >= 0:
                    pbar.update(env.N_TIME_STEPS)

                if (idx * env.dt) >= max_t:
                    break

            all_actions = env.transform_action(all_actions)
            all_actions = env.denorm_action(all_actions)
            action_pulse = np.cumsum(all_actions)

            pulses.append(action_pulse.copy())
            rewards.append(rews.copy())

        '''
        *******************************
        *******************************
        *********  PLOTTING  **********
        *******************************
        *******************************
        '''
        # Prepare time-axis and pulse amplitude data for plotting
        infidelities = 1 - np.array(env.fidelities)

        n_steps_coarse = len(infidelities)
        tlist_coarse = np.arange(n_steps_coarse) * env.dt * env.N_TIME_STEPS

        n_steps_fine = len(pulses[0])
        tlist_fine = np.arange(n_steps_fine) * env.dt

        idx_best_coarse = np.argmax(rews)
        t_best = tlist_coarse[idx_best_coarse]
        idx_best_fine = np.argmin(np.abs(np.subtract(tlist_fine, t_best)))

        print(f"n_steps_coarse={n_steps_coarse}, idx_best_coarse={idx_best_coarse}, idx_best_fine={idx_best_fine}")

        ep_len_coarse = n_steps_coarse
        ep_len_fine = n_steps_fine

        tlist_coarse = tlist_coarse[:ep_len_coarse]
        tlist_fine = tlist_fine[:ep_len_fine]

        print(f"pulse average std: {np.mean(np.std(pulses, axis=0))}")
        print(f"rewards average std: {np.mean(np.std(rewards, axis=0))}")

        '''
        Pulse plot
        '''
        if verbose >= 1:
            import seaborn as sns
            sns.set_theme(style="ticks")
            sns.set_palette("deep")
            # fig, axs = plt.subplots(2, figsize=(10, 6), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
            fig, ax = plt.subplots(figsize=(10, 5))
            # fig.suptitle(title)
            # ax = axs[0]
            # for i, _pulse in enumerate(pulses):
            #     ax.plot(tlist_fine, pulses[i][:ep_len_fine], lw=0.5, alpha=0.5, label=f'Ep #{i}')
            pulse_mean = np.mean(pulses, axis=0)[:ep_len_fine]
            pulse_std = np.std(pulses, axis=0)[:ep_len_fine]
            ax.plot(tlist_fine, pulse_mean, lw=1, c='k', label='Mean')
            ax.fill_between(tlist_fine, pulse_mean - pulse_std, pulse_mean + pulse_std, alpha=0.2, color='k')
            ax.set_ylabel('Pulse amplitude')
            ax.set_xlabel('Time [ns]')
            ax.set_xlim(0, tmax)
            ax.grid(which='major', linestyle='--', color='grey', linewidth=1)
            ax.grid(which='minor', linestyle=':', color='lightgrey', linewidth=0.5)

            '''
            ax = axs[1]
            # for i, rews in enumerate(rewards):
            #     ax.plot(tlist_coarse, rewards[i][:ep_len_coarse], lw=1., label=f'Ep #{i}')
            rew_mean = np.mean(rewards, axis=0)[:ep_len_coarse]
            rew_std = np.std(rewards, axis=0)[:ep_len_coarse]
            ax.plot(tlist_coarse, rew_mean, lw=1, c='k', label='Mean')
            ax.fill_between(tlist_coarse, rew_mean-rew_std, rew_mean+rew_std, alpha=0.2, color='k')
            ax.set_xlabel('Time [ns]')
            ax.set_ylabel('Reward')
            ax.grid(which='major', linestyle='--', color='grey', linewidth=1)
            ax.grid(which='minor', linestyle=':', color='lightgrey', linewidth=0.5)
            ax.set_ylim(0, np.max(rewards))

            for ax in axs:
                # ax.legend(loc="best")
                ax.set_xlim(0, tmax)
            '''

            save_path = os.path.join(res_save_dir, f'{save_name}_pulse_{n_eps}eps_{tmax:.1f}tmax.pdf')
            if verbose > 2:
                print(f'Saving result to: {save_path}')
            fig.tight_layout()
            fig.savefig(save_path)
            plt.show()

        infidelities = infidelities[:ep_len_coarse]
        concurrences = np.array(env.concurrences[:ep_len_coarse])
        unitarities = np.array(env.unitarities[:ep_len_coarse])

        max_fid = env.fidelities[idx_best_coarse] * 100.
        idx_min_infidelity = np.argmin(infidelities)
        idx_max_concurrence = np.argmax(concurrences)
        idx_max_unitarity = np.argmax(unitarities)
        metrics_str = lambda \
                j: f'Rew={rews[j]:.2f} => C={concurrences[j]:.4f}; U={unitarities[j]:.4f}; (1-F)={infidelities[j]:.4f} @ T={tlist_coarse[j]:.2f}ns; PL={j + 1:d}'
        title = (f'Best Reward: {metrics_str(idx_best_coarse)}\n'
                 f'Best Concurrence: {metrics_str(idx_max_concurrence)}\n'
                 f'Best Unitarity: {metrics_str(idx_max_unitarity)}\n'
                 f'Best Fidelity: {metrics_str(idx_min_infidelity)}'
                 )
        if verbose > 0:
            print(f'Model: {model_zip}; Ep: {i}')
            print(title + '\n')

        '''
        Reward metrics
        '''
        if verbose >= 5:
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

        # # Save pulse to CSV
        # dat = pd.DataFrame()
        # dat['amplist'] = pulse
        # dat['tlist'] = tlist_fine
        # dat.to_csv(os.path.join(res_save_dir, f'{save_name}.csv'))
        #
        # dat = pd.DataFrame()
        # dat['concurrences'] = concurrences
        # dat['unitarities'] = unitarities
        # dat['rewards'] = rews
        # dat['tlist'] = tlist_coarse
        # dat.to_csv(os.path.join(res_save_dir, f'{save_name}_metrics.csv'))


if __name__ == '__main__':
    main()
