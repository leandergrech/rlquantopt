import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from stable_baselines3 import PPO, SAC
# from stable_baselines3.common.evaluation import evaluate_policy
from rlquantopt.utils.rl_utils import evaluate_policy
# from rlquantopt.rl_envs.qu_pulse_episodic_env import QuPulseEpisodicEnv as QPEE
from rlquantopt.rl_envs.zc_qpee import ZCQPEE

algo_str = 'PPO'
algo = PPO
pulse_length = 120
# model_dir = f'ZCQPEE{pulse_length}pl-PPO/09-06-24_172549_ZCQPEE{pulse_length}pl'
# model_dir = os.path.join(f'ZCQPEE{pulse_length}pl-{algo_str}/10-06-24_143544_ZCQPEE{pulse_length}pl', 'best_model')
# model_dir = os.path.join(f'ZCQPEE{pulse_length}pl-{algo_str}/10-06-24_143544_ZCQPEE{pulse_length}pl')
model_dir = os.path.join(f'ZCQPEE{pulse_length}pl-{algo_str}/13-06-24_115241_ZCQPEE{pulse_length}pl', 'best_model')
# model_dir = os.path.join(f'ZCQPEE{pulse_length}pl-{algo_str}/13-06-24_140527_ZCQPEE{pulse_length}pl')
# model_dir = os.path.join(f'ZCQPEE{pulse_length}pl-{algo_str}/13-06-24_150202_ZCQPEE{pulse_length}pl', 'best_model')
# model_dir = os.path.join(f'ZCQPEE{pulse_length}pl-{algo_str}/13-06-24_200612_ZCQPEE{pulse_length}pl')
env = ZCQPEE(pulse_length=pulse_length)
env.action_channel_scaling['z'] = 1e-1

mp4_save_dir = os.path.join(model_dir, 'evals')
pulse_save_dir = os.path.join(model_dir, 'pulses')

for d in (mp4_save_dir, pulse_save_dir):
    if not os.path.exists(d):
        os.makedirs(d)

# env = QPEE(pulse_length=pulse_length, sparse_reward=sparse_reward, use_full_liouville=use_full_liouville, inc_off_diag=True)

# for train_step in tqdm(np.arange(0, 5)*5000 + 55000):
#     model_fn = f'rl_model_{train_step}_steps.zip'

# for train_step in (40000, 50000, 30000):
# for train_step in np.arange(10000, 250001, 10000):
for _ in range(1):
    model_fn = 'best_model'
    # model_fn = f'rl_model_{train_step}_steps'
    model_path = os.path.join(model_dir, model_fn)
    model_name = os.path.splitext(model_fn)[0]
    save_path = os.path.join(mp4_save_dir, f'{model_name}2.mp4')

    model = algo.load(model_path)

    # print(f'Mean reward = {mean_rew:.2f}')
    # print(f'Std reward  = {std_rew:.2f}')

    obs = env.reset()[0]
    term = False
    trunc = False
    idx = 0
    rews = []
    fidelities = []
    pbar= tqdm(total=pulse_length)
    prev_amp = 0.
    pulse = [prev_amp]
    while not term and not trunc:
        idx += 1
        a = model.predict(obs, deterministic=True)[0]
        # a += np.random.normal(0, 1e-3, env.n_act)
        obs, r, term, trunc, info = env.step(a)

        prev_amp += env.denorm_action(a)[0]
        pulse.append(prev_amp)

        rews.append(r)
        fidelities.append(env.rew2fid(r))
        pbar.update(1)

    infidelities = 1 - np.array(fidelities)
    infidelities *= 100.

    tlist = np.arange(len(pulse)) * env.dt
    # pulse = [env.denorm_action(item) for item in pulse]

    # Save pulse to CSV
    dat = pd.DataFrame()
    dat['amplist'] = pulse
    dat['tlist'] = tlist
    dat.to_csv(os.path.join(pulse_save_dir, f'{model_name}.csv'))

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

    env.render(save_path=save_path)
