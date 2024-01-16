import os
import numpy as np
from datetime import datetime as dt
import yaml

from qutip_qip.circuit import QubitCircuit
from pulses_workspace.utils import LinearDecay, boltzmann, eps_greedy
from pulses_workspace.rl_envs.qu_pulse_env import QuPulseEnv
from pulses_workspace.rl_agents.sarsa import train_instance_early_termination

experiment_dir = f"qupulseenv_sarsa_{dt.now().strftime('%m%d%y_%H%M%S')}"

nb_training_steps = 1000
eval_every = 25
save_every = 50
eval_eps = 4
start_eval = 0

explore_until = decay_lr_until = nb_training_steps

exp_fun = LinearDecay(0.1, 1e-2, explore_until, label='EPS')
# lr_fun = LinearDecay(5e-2, 5e-3, decay_lr_until, label='LR')
lr_fun = LinearDecay(1e-2, 5e-3, decay_lr_until, label='LR')
gamma = 0.9

if 'EPS' in exp_fun.label:
    policy = eps_greedy
else:
    policy = boltzmann

if not os.path.exists(experiment_dir):
    os.makedirs(experiment_dir)
else:
    raise FileExistsError

np.random.seed(seed=123)
qc = QubitCircuit(1)
qc.add_gate('X', targets=0)
env = QuPulseEnv(qc)
eval_env = QuPulseEnv(qc)

train_params = dict(
    quantumCircuit=qc,
    lr_fun=lr_fun,
    exp_fun=exp_fun,
    gamma=gamma,
    nb_training_steps=nb_training_steps,
    eval_every=eval_every,
    eval_eps=eval_eps,
    save_path=experiment_dir,
    save_every=save_every,
    nb_successes_early_termination=5,
    start_eval=start_eval
)

with open(os.path.join(experiment_dir, "train_params.yml"), "w") as f:
    f.write(yaml.dump(train_params))

train_params['policy'] = policy
train_params['env'] = env
train_params['eval_env'] = eval_env

finish_timestep = train_instance_early_termination(**train_params)

with open(os.path.join(experiment_dir, 'train_info.md'), 'w') as f:
    f.write(f'{finish_timestep}')
