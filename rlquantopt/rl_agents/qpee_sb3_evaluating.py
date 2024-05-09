import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from rlquantopt.rl_envs.qu_pulse_episodic_env import QuPulseEpisodicEnv as QPEE

# model_path = '03-05-24_171958_VQPEE_14-envs_PPO_1000-n_steps_140-batch_size.zip'
# model_path = '08-05-24_010313_VQPEE_16-envs_PPO_2048-n_steps_64-batch_size.zip'
model_dir = 'VQPEE-PPO/09-05-24_095621_16-envs'
model_name = 'rl_model_163680_steps.zip'
model_path = os.path.join(model_dir, model_name)

model = PPO.load(model_path)
pulse_length = 50
sparse_reward = False
env = QPEE(pulse_length=pulse_length, sparse_reward=sparse_reward)

# mean_rew, std_rew = evaluate_policy(model, env=env, n_eval_episodes=1)
# print(f'Mean reward = {mean_rew:.2f}')
# print(f'Std reward  = {std_rew:.2f}')

obs = env.reset()[0]
term = False
trunc = False
idx = 0
rews = []
pbar= tqdm(total=pulse_length)
while not term and not trunc:
    idx += 1
    a = model.predict(obs, deterministic=True)[0]
    obs, r, term, trunc, info = env.step(a)
    rews.append(r)
    pbar.update(1)
    # print(f'Step {idx}: Reward {r}')

# fig, ax = plt.subplots()
# ax.plot(rews)
# ax.set_title(f'Evaluating policy in {model_path}')
# ax.set_xlabel('Steps')
# ax.set_ylabel('Reward (F)')
#
# plt.show()
save_dir = os.path.join(model_dir, 'evals')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
env.render(save_path=os.path.join(save_dir, os.path.splitext(model_name)[0]+'.mp4'))
