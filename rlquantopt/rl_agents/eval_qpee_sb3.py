import os
import argparse
import pandas as pd
import numpy as np
from pygments.lexer import default
from tqdm import tqdm
from stable_baselines3 import PPO, SAC
# from stable_baselines3.common.evaluation import evaluate_policy
from rlquantopt.utils.rl_utils import evaluate_policy
# from rlquantopt.rl_envs.qu_pulse_episodic_env import QuPulseEpisodicEnv as QPEE
from rlquantopt.rl_envs.zc_qpee import ZCQPEE


def parse_args():
    parser = argparse.ArgumentParser('RLQuantOpt - training RL agent on ZCQPEE environment')
    # ZCQPEE parameters
    parser.add_argument('--pulse-length', default=120, type=int, help='Maximum number of samples in a pulse')
    parser.add_argument('--delta-mode', action='store_true', help='Use ZCQPEE environment in delta action mode')
    parser.add_argument('')


    return parser.parse_args()

algo_str = 'PPO'
algo = PPO
pulse_length = 120
T = 50
delta_mode = False
default_model_params = False
model_datetime = '13-06-24_115241'

render = True

model_dir = f'ZCQPEE{pulse_length}pl-{algo_str}/{model_datetime}_ZCQPEE{pulse_length}pl'
model_dir = os.path.join(model_dir, 'best_model')
env = ZCQPEE(pulse_length=pulse_length, delta_mode=delta_mode, default_model_params=default_model_params, T=T)
env.action_scaling['z'] = 1e-1

mp4_save_dir = os.path.join(model_dir, 'evals')
pulse_save_dir = os.path.join(model_dir, 'pulses')

for d in (mp4_save_dir, pulse_save_dir):
    if not os.path.exists(d):
        os.makedirs(d)

# for train_step in np.arange(140000, 240001, 20000):
for _ in range(1):
    model_fn = 'best_model'
    # model_fn = f'rl_model_{train_step}_steps'
    model_path = os.path.join(model_dir, model_fn)
    model_name = os.path.splitext(model_fn)[0]
    save_name = f"{model_name}_T{T}ns{'_delta_mode' if delta_mode else ''}{'_default_model' if default_model_params else ''}_to_term"
    save_path = os.path.join(mp4_save_dir, f'{save_name}.mp4')

    print(save_name)

    model = algo.load(model_path)

    obs = env.reset()[0]
    term = False
    trunc = False
    idx = 0
    rews = []
    fidelities = []
    pbar= tqdm(total=pulse_length)
    prev_amp = 0.
    pulse = [prev_amp]
    while not (term or trunc):
        idx += 1
        a = model.predict(obs, deterministic=True)[0]
        # a += np.random.normal(0, 1e-3, env.n_act)
        obs, r, term, trunc, info = env.step(a)

        a_denorm = env.denorm_action(a)[0]
        if delta_mode:
            prev_amp += a_denorm
        else:
            prev_amp = a_denorm
        pulse.append(prev_amp)

        rews.append(r)
        fidelities.append(env.rew2fid(r))
        pbar.update(1)

    print(f'Max fidelity: {max(fidelities) * 100:.2f}%')

    infidelities = 1 - np.array(fidelities)
    infidelities *= 100.

    tlist = np.arange(len(pulse)) * env.dt
    # pulse = [env.denorm_action(item) for item in pulse]

    # Save pulse to CSV
    dat = pd.DataFrame()
    dat['amplist'] = pulse
    dat['tlist'] = tlist
    dat.to_csv(os.path.join(pulse_save_dir, f'{save_name}.csv'))

    # # Save evaluation episode details
    # fig, ax = plt.subplots()
    # axx =ax.twinx()
    # ax.plot(tlist[:-1], infidelities)
    # ax.set_title(f'Evaluating policy in {model_path}')
    # ax.set_xlabel('Time (ns)')
    # ax.set_ylabel('Infidelity (%)')
    # ax.set_yscale('log')
    #
    # axx.plot(tlist[:-1], rews, c='tab:orange')
    # axx.set_ylabel('Rewards')
    # fig.savefig(os.path.join(mp4_save_dir, f'{model_name}.png'))

    if render:
        env.render(save_path=save_path)
