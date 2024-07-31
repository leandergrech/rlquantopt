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
    parser.add_argument('-a', '--algo', type=str, default='PPO', help='Type of RL algorithm')
    parser.add_argument('--save-dir', type=str, default='', help='Path to save evaluation results')
    parser.add_argument('-e', '--env-yml-dir', type=str, default='', help='Directory containing only one yaml file with env arguments. Default, model directory.')
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
    model_zips = args.model_zip
    if isinstance(model_zips, (list, tuple)):
        for item in model_zips:
            if not isinstance(item, str):
                raise argparse.ArgumentTypeError(f'{item} sub-argument is not a string')
    elif isinstance(model_zips, str):
        model_zips = [model_zips]
    # print(model_zip[0])
    # exit(23)
    for model_zip in sorted(model_zips):
        model_dir = os.path.dirname(model_zip)
        if (save_dir:= args.save_dir) == '':
            save_dir = os.path.dirname(model_zip)
        n_eps = args.n_eps
        render = args.r
        verbose = args.verbose

        algo_str = args.algo
        if algo_str == 'PPO':
            algo = PPO
        elif algo_str == 'TRPO':
            algo = TRPO
        elif algo_str == 'RecurrentPPO':
            algo = RecurrentPPO

        if (env_yml_dir := args.env_yml_dir) == '':
            env_yml_dir = model_dir
        env_yml_path = None
        for item in os.listdir(env_yml_dir):
            if item.endswith('.yml') or item.endswith('.yaml'):
                env_yml_path = os.path.join(env_yml_dir, item)
                break
        if env_yml_path is None:
            raise Exception('No ZCQPEE environment configuration was found')
        #
        # pulse_length = 120
        # T = 50
        # delta_mode = False
        # default_model_params = False
        # env = ZCQPEE(pulse_length=pulse_length, delta_mode=delta_mode, default_model_params=default_model_params, T=T, action_scaling={'z':1e-1}, a_norm_max=10)
        env = ZCQPEE.from_yaml(env_yml_path)

        mp4_save_dir = os.path.join(save_dir, 'evals')
        if render and not os.path.exists(mp4_save_dir):
            os.makedirs(mp4_save_dir)

        res_save_dir = os.path.join(save_dir, 'results')

        for d in (mp4_save_dir, res_save_dir):
            if not os.path.exists(d):
                os.makedirs(d)

        for i in range(n_eps):
            model_name = os.path.splitext(os.path.basename(model_zip))[0]
            save_name = f"{model_name}_{str(env)}_ep{i}"
            mp4_save_path = os.path.join(mp4_save_dir, f'{save_name}.mp4')
            print(f'Saving animations in: {mp4_save_path}')

            model = algo.load(model_zip)

            obs = env.reset()[0]
            term = False
            trunc = False
            idx = 0
            rews = []
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
                obs, r, term, trunc, info = env.step(a, can_term=True)

                a_denorm = env.denorm_action(a)[0]
                if env.delta_mode:
                    prev_amp += a_denorm
                else:
                    prev_amp = a_denorm
                pulse.append(prev_amp)

                rews.append(r)
                pbar.update(1)

            max_fid = max(env.fidelities) * 100

            infidelities = 1 - np.array(env.fidelities)
            infidelities *= 100.

            ep_len = len(infidelities)
            if verbose > 1:
                fig, ax = plt.subplots()
                ax.plot(env.tlist[:ep_len], infidelities, c='k')
                ax.set_xlabel('Time [ns]')
                ax.set_ylabel('Infidelity (%)')
                ax.set_yscale('log')
                ax.grid(which='major', linestyle='--', color='grey', linewidth=1)
                ax.grid(which='minor', linestyle=':', color='lightgrey', linewidth=0.5)

                axx = ax.twinx()
                axx.plot(env.tlist[:ep_len], np.array(env.fidelities) * 100., c='g')
                axx.set_ylabel('Fidelity (%)', color='g')
                axx.spines['right'].set_color('g')
                axx.tick_params(axis='y', colors='g', color='g')

                fig.suptitle(f'Max fidelity = {max_fid:.2f}%')
                axx.axhline(max_fid, c='g', linestyle='--', linewidth=0.5)
                axx.axvline(env.tlist[np.argmax(env.fidelities)], c='g', linestyle='--', linewidth=0.5)

                save_path = os.path.join(res_save_dir, f'{save_name}_Fmax-{max_fid:.1f}_fid_vs_infid_plot.pdf')
                if verbose > 2:
                    print(f'Saving result to: {save_path}')
                fig.savefig(save_path)

            if verbose > 0:
                print(f'{os.sep}'.join(model_zip.split(os.sep)[-3:]) + ': ', end='')
                print(f'Max. Fidelity = {max_fid:.2f}%')
                fig, ax = plt.subplots()
                fig.suptitle(f'Max reward = {max(rews):.2f}%')
                ax.plot(env.tlist[:ep_len], rews, c='tab:orange')
                ax.set_xlabel('Time [ns]')
                ax.set_ylabel('Reward')
                ax.grid(which='major', linestyle='--', color='grey', linewidth=1)
                ax.grid(which='minor', linestyle=':', color='lightgrey', linewidth=0.5)
                save_path = os.path.join(res_save_dir, f'{save_name}_rew_plot.pdf')
                if verbose > 2:
                    print(f'Saving result to: {save_path}')
                fig.savefig(save_path)

                fig, ax = plt.subplots()
                ax.plot(env.tlist[:ep_len], pulse[:-1], c='k', lw=0.5)
                ax.set_xlabel('Time [ns]')
                ax.set_ylabel('Pulse amplitude')
                ax.grid(which='major', linestyle='--', color='grey', linewidth=1)
                ax.grid(which='minor', linestyle=':', color='lightgrey', linewidth=0.5)
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
            dat['amplist'] = pulse
            dat['tlist'] = tlist
            dat.to_csv(os.path.join(res_save_dir, f'{save_name}.csv'))

            if render:
                env.render(save_path=mp4_save_path)


if __name__ == '__main__':
    main()