import os
import argparse
import pandas as pd
import numpy as np
from sb3_contrib import RecurrentPPO, TRPO
from tqdm import tqdm
from stable_baselines3 import PPO, SAC
import matplotlib.pyplot as plt
import weylchamber

# from rlquantopt.rl_analysis.check_pulses.check_pulses import pulse_file
from rlquantopt.rl_envs.zc_qpee import ZCQPEE
# from rlquantopt.tests.zcqpee_tests import pulse
from rlquantopt.utils.rl_utils import get_pulse_data


def parse_args():
    parser = argparse.ArgumentParser('RLQuantOpt - evaluating RL agent on ZCQPEE environment with Weyl Chamber plots')
    # ZCQPEE parameters
    parser.add_argument('pulse-csv', default='', type=str, help='Path to pulse CSV file')

    parser.add_argument('-a', '--algo', type=str, default='TRPO', help='Type of RL algorithm')
    parser.add_argument('-x', '--max-t', type=float, default=300., help='')

    parser.add_argument('--save-dir', type=str, default='', help='Path to save evaluation results')
    parser.add_argument('-e', '--env-yml-dir', type=str, default='', help='Directory containing only one yaml file with env arguments. Default, model directory.')
    parser.add_argument('--no-term', action='store_true', help='Keep running episode until the end of the pulse not until termination.')
    parser.add_argument('--clip-best', action='store_true', help='Keep running episode until the end of the pulse, and clip pulse until best performance is reached.')
    parser.add_argument('-v', '--verbose', type=int, default=2,
                        help='Levels of analysis detail to perform on the evaluation episode.\n'
                             '0 - Save pulse CSV only\n'
                             '1 - Add pulse plots\n'
                             '2 - Add reward metrics\n'
                             '3 - Add reward plot')
    parser.add_argument('--save-suffix', type=str, default='', help='Result files save name suffix. Default ""')

    return parser.parse_args()


def main():
    # args = parse_args()
    # pulse_file = args.pulse_csv   # arg
    # no_term = args.no_term
    # clip_best = args.clip_best
    # save_suffix = args.save_suffix
    # verbose = args.verbose
    # max_t = args.max_t

    # pulse_dir = os.path.dirname(pulse_file)
    # if (save_dir := args.save_dir) == '':
    #     save_dir = os.path.dirname(pulse_file)
    #
    # if (env_yml_dir := args.env_yml_dir) == '':
    #     env_yml_dir = pulse_dir
    # elif env_yml_dir.startswith('../'):
    #     env_yml_dir = os.path.abspath(os.path.join(pulse_dir, env_yml_dir))

    pulse_file = '/home/leander/code/rlquantopt/rlquantopt/rl_agents/ZCQPEE_pl-1000_T-50ns_delta_mode-TRPO/05-12-24_201634/results/no_term/rl_model_12566528_steps_ZCQPEE_pl-1000_T-50ns_delta_mode_ep0_no_term.csv'
    no_term = True
    clip_best = False
    save_suffix = ""
    save_dir = ''
    env_yml_dir = '.'
    verbose = 0
    max_t = 50.

    assert sum([no_term, clip_best]) <= 1, 'Only one episode termination option can be set at a time. Either --no-term or --clip-best.'

    if clip_best:
        no_term = True

    # Get pulse related data
    tlist, amps = get_pulse_data(pulse_file)
    amps_diff = np.diff(amps, prepend=0)

    # Load ZCQPEE environment from YAML configuration file
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

    # Setup result directories
    res_save_dir = os.path.join(save_dir, 'results')
    if clip_best:
        res_save_dir = os.path.join(res_save_dir, 'clip_best')
    elif no_term:
        res_save_dir = os.path.join(res_save_dir, 'no_term')
    os.makedirs(res_save_dir, exist_ok=True)

    # Create result name
    save_name = f"{os.path.splitext(os.path.basename(pulse_file))[0]}_{str(env)}"
    if clip_best:
        save_name += '_clip_best'
    elif no_term:
        save_name += '_no_term'
    save_name += save_suffix

    # Initialise episode
    env.reset()

    pbar = None
    if verbose >= 0:
        pbar = tqdm(total=env.pulse_length)

    term = False
    idx = 0
    rews = []
    prev_amp = 0.
    pulse = [prev_amp]
    realised_gates = []

    step_states = []

    while not term:
        _idx = idx * env.N_TIME_STEPS
        if env.delta_mode:
            a = amps_diff[_idx: _idx + env.N_TIME_STEPS]
        else:
            a = amps[_idx: _idx + env.N_TIME_STEPS]

        a = env.norm_action(a)

        idx += 1
        obs, r, term, trunc, info = env.step(a, can_early_term=(not no_term))

        realised_gates.append(info['realised_gate'])
        step_states.append(info['step_states'])

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

        if idx * env.dt >= max_t:
            break

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
        print(f'Pulse: {pulse_file}; Ep: {i}')
        print(title + '\n')



if __name__ == '__main__':
    main()
