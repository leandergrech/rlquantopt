import os
import shutil
import argparse
import json
import torch as tc
from datetime import datetime as dt

from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, StopTrainingOnNoModelImprovement
from stable_baselines3 import PPO, SAC
from sb3_contrib import RecurrentPPO, TRPO

from rlquantopt.rl_envs.zc_qpee import ZCQPEE
from rlquantopt.rl_agents.callbacks import EvalCallback
from rlquantopt.utils.utils import get_latest_experiment, save_exp_info
from rlquantopt.utils.rl_utils import linear_schedule, harmonic_schedule


DT_FMT_STR = '%d-%m-%y_%H%M%S'


def parse_args():
    parser = argparse.ArgumentParser('RLQuantOpt - training RL agent on ZCQPEE environment')
    # ZCQPEE parameters
    parser.add_argument('-p', '--pulse-length', default=1000, type=int, help='Maximum number of samples in a pulse')
    parser.add_argument('-d', '--delta-mode', action='store_true', help='Use ZCQPEE environment in delta action mode')
    parser.add_argument('-T', '--max-time-ns', default=50.0, type=float, help='Pulse duration in ns')
    parser.add_argument('-N', '--n-time-steps', default=3, type=int, help='Number of time-steps per action')
    parser.add_argument('--tv-penalty-scale', default=1e-3, type=float, help='Number of time-steps per action')
    parser.add_argument('-o', '--act-poly-order', default=1, type=float, help='Order of action transformation polynomial ')
    parser.add_argument('-a', '--add-prev-obs', action='store_true', help='Add previous observable w/o the action and time, to the current observable.')
    parser.add_argument('-f', '--use-fidelity', action='store_true', help='Use fidelity metric to form reward instead of concurrence & unitarity')
    parser.add_argument('--a-scale', default=20., type=float, help='Set action scaling during normalisation')
    parser.add_argument('--a-norm-max', default=1.0, type=float, help='Set the maximum normalised amplitude when using delta-mode')
    parser.add_argument('--rew-scale', default=1.0, type=float, help='Multiply reward by rew_scale after each step')
    parser.add_argument('--fid-thresh', default=0.99, type=float, help='Set goal fidelity threshold')
    parser.add_argument('--concurrence_weight', default=1, type=float, help='When --use-fidelity is not active, sets the weight of concurrence in the reward function')
    parser.add_argument('--unitarity_weight', default=3, type=float, help='When --use-fidelity is not active, sets the weight of unitarity in the reward function')
    parser.add_argument('--fid-thresh', default=0.99, type=float, help='Set goal fidelity threshold')

    # RL agent paramters
    parser.add_argument('--algo', type=str, default='TRPO', help='Type of RL agent')
    n_steps = 2048
    n_envs = 4
    parser.add_argument('--n-steps', type=int, default=n_steps, help='The number of steps to run for each environment per update'
                             '(i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)'
                             'NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)'
                             'See https://github.com/pytorch/pytorch/issues/29372')
    parser.add_argument('--n-epochs', type=int, default=10, help='Number of epoch when optimizing the surrogate loss')
    parser.add_argument('--ent-coef', type=float, default=0, help='Entropy coefficient for the loss calculation')
    parser.add_argument('--batch-size', type=int, default=128, help='Mini-batch size')
    # parser.add_argument('--batch-size', type=int, default=256, help='Mini-batch size')  # SAC
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--seed', default=123, type=int, help='Set random seed')

    # Training parameters
    parser.add_argument('--n-envs', default=n_envs, type=int, help='Number of vectorised training environments')
    parser.add_argument('--n-train', default=20000000, type=int, help='Number of training steps')
    parser.add_argument('--save-freq', default=n_steps, type=int, help='Save model every save_freq calls to env.step')
    parser.add_argument('--eval-freq', default=int(n_steps*n_envs/2), type=int, help='Evaluate model every eval_freq calls to env.step')
    parser.add_argument('--log-interval', default=n_steps, type=int, help='Log every N calls to env.step')
    parser.add_argument('--n-eval-eps', default=1, type=int, help='Number of evaluation episodes done every eval_freq calls to env.step')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--msg', default='', type=str, help='User message to add to info.txt')
    parser.add_argument('-L', '--hidden-layer-size', default=128, type=int, help='Network hidden layer size. All layers are equal size')
    parser.add_argument('-H', '--n-hidden-layers', default=2, type=int, help='Nb. of hidden layers in last layer MLP')

    # Fine-tuning an existing model
    parser.add_argument('-r', '--retrain-model', type=str, default=None,
                        help='By passing the path the model directory where the zip files are located, you are instructing to continue training with these new parameters')

    return parser.parse_args()


def copy_scripts_to_model_path(model_path):

    save_dir = os.path.join(model_path, 'saved_scripts')
    os.makedirs(save_dir, exist_ok=True)

    def copy_file_to_dir(file_path):
        nonlocal save_dir
        file_name = os.path.basename(file_path)
        shutil.copy(file_path, os.path.join(save_dir, file_name))

    import rlquantopt.rl_envs.zcqubits as temp
    copy_file_to_dir(temp.__file__)
    import rlquantopt.rl_envs.zc_qpee as temp
    copy_file_to_dir(temp.__file__)
    import rlquantopt.rl_agents.train_zcqpee_sb3 as temp
    copy_file_to_dir(temp.__file__)


def main():
    args = parse_args()

    # Define environment parameters
    env_kw = dict(pulse_length=int(args.pulse_length),
                  rew_scale=float(args.rew_scale),
                  delta_mode=args.delta_mode,
                  T=float(args.max_time_ns),
                  fid_thresh=float(args.fid_thresh),
                  action_scaling={'z': float(args.a_scale)},
                  a_norm_max=float(args.a_norm_max),
                  n_time_steps=args.n_time_steps,
                  tv_penalty_scale=args.tv_penalty_scale,
                  act_poly_order=args.act_poly_order,
                  add_prev_obs=args.add_prev_obs,
                  use_fidelity=args.use_fidelity,
                  concurrence_weight=args.concurrence_weight,
                  unitarity_weight=args.unitarity_weight
                  )
    n_envs = args.n_envs
    gamma = args.gamma

    # env = make_vec_env(lambda: NormalizeObservation(NormalizeReward(ZCQPEE(**env_kw), gamma=gamma), epsilon=1e-8), n_envs=n_envs)
    env = make_vec_env(lambda: ZCQPEE(**env_kw), n_envs=n_envs)

    eval_env = ZCQPEE(**env_kw)
    env_yaml_fn = str(eval_env) + '.yml'

    # Prepare training info message
    info_fn = 'info.txt'
    TRAINING_MESSAGE = f"{repr(eval_env)}\n" + f"Nb. envs: {n_envs}\n\n{args.msg}\n. "

    print(f'n_obs={eval_env.n_obs}\tn_act={eval_env.n_act}')

    # Parameter setup
    learning_rate = harmonic_schedule(init_value=3e-4, k=10**0.4)
    lr_type = 'Harmonic decay'
    # learning_rate = 3e-4
    # lr_type = 'Constant'
    ppo_clip_range = lambda x: 0.2
    # ppo_learning_rate = 3e-4

    n_train = int(args.n_train)
    n_steps = int(args.n_steps)
    batch_size = int(args.batch_size)
    n_epochs = int(args.n_epochs)
    save_freq = int(args.save_freq)
    eval_freq = max(n_envs * n_steps, int(args.eval_freq))
    log_interval = eval_freq
    n_eval_eps = int(args.n_eval_eps)
    SEED = args.seed
    L = args.hidden_layer_size
    H = args.n_hidden_layers

    # Setting device on the CPU only for now since I am working on my laptop
    if args.no_cuda:
        device = 'cpu'
    else:
        device = 'cuda'

    # RL algorithm setup
    algo_str = args.algo

    # Network architecture setup
    net_arch = dict(pi=[L]*H, vf=[L]*H)
    if 'SAC' in algo_str: net_arch['qf'] = net_arch.pop('vf')
    policy_kwargs = dict(activation_fn=tc.nn.ReLU,
                         net_arch=net_arch
                         )
    if 'Recurrent' in algo_str:
        policy_kwargs.update(dict(lstm_hidden_size=args.lstm_hidden_size,
                                  n_lstm_layers=args.n_lstm_layers,
                                  enable_critic_lstm=True))

    policy_type = 'MlpPolicy'
    if algo_str == 'PPO':
        algo = PPO
        algo_kw = dict(batch_size=batch_size, n_steps=n_steps, learning_rate=learning_rate, device=device,
                       n_epochs=n_epochs, gamma=gamma, clip_range=ppo_clip_range, max_grad_norm=0.5, gae_lambda=0.95,
                       ent_coef=args.ent_coef, vf_coef=0.5, use_sde=False, stats_window_size=100, seed=SEED, verbose=1)
    elif algo_str == 'TRPO':
        algo = TRPO
        algo_kw = dict(batch_size=batch_size, n_steps=n_steps, learning_rate=learning_rate, device=device,
                       gamma=gamma, seed=SEED, verbose=1)
    elif algo_str == 'SAC':
        algo = SAC
        algo_kw = dict(learning_starts=1000, train_freq=(10, 'step'), batch_size=batch_size, learning_rate=learning_rate,
                       device=device, gamma=gamma, seed=SEED, verbose=1)
    elif algo_str == 'RecurrentPPO':
        algo = RecurrentPPO
        policy_type = 'MlpLstmPolicy'
        algo_kw = dict(batch_size=batch_size, n_steps=n_steps, learning_rate=learning_rate, device=device,
                       n_epochs=n_epochs, gamma=gamma, clip_range=ppo_clip_range, max_grad_norm=0.5, gae_lambda=0.95,
                       ent_coef=args.ent_coef, vf_coef=0.2, use_sde=False, stats_window_size=10, seed=SEED, verbose=1)
    else:
        raise NotImplementedError

    '''
    NAME NEW MODEL OR RETRAIN FROM MODEL CHECKPOINT
    '''
    model_checkpoint = None
    model_path = args.retrain_model
    if model_path is None:
        model_name = f"{dt.now().strftime(DT_FMT_STR)}"
        model_path = os.path.join(os.path.join(f'{str(eval_env)}-{algo_str}'), model_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        eval_env.to_yaml(os.path.join(model_path, env_yaml_fn))
        _best_model_path = os.path.join(model_path, 'best_model')
        if not os.path.exists(_best_model_path):
            os.makedirs(_best_model_path)
        eval_env.to_yaml(os.path.join(_best_model_path, env_yaml_fn))
    else:
        model_name = os.path.basename(model_path)
        model_checkpoint = get_latest_experiment(model_path, pattern='rl_model_')
        eval_env.from_yaml(ZCQPEE.find_yaml_in_dir(model_path))
        # for k, v in env_kw.items():
            # assert eval_env.model_params[k] == v, f'New vs. original training env mismatch! New training env parameters don\'t match the original environment parameters: New {k}: {v} != Original {k}: {eval_env.model_params[k]}'

    '''
    SAVE INFORMATION ABOUT THIS TRAINING SESSION
    Including model args, script args, environment details, simulation details
    '''
    print(f'Using {os.path.abspath(model_path)} directory')

    info_path = os.path.join(model_path, info_fn)

    _pk = policy_kwargs.copy()
    _pk.pop('activation_fn')
    _ak = algo_kw.copy()
    if not isinstance(_ak['learning_rate'], float):
        _ak['learning_rate'] = f'{lr_type} from {learning_rate(1)} -> {learning_rate(0)}'
    if 'PPO' in algo_str:
        if not isinstance(_ak['clip_range'], float):
            _ak['clip_range'] = f'Constant clip range {ppo_clip_range(1)} -> {ppo_clip_range(0)}'

    save_exp_info(info_path=info_path,
                  training_message=TRAINING_MESSAGE,
                  env_kwargs=env_kw,
                  rl_algo=algo_str,
                  algo_kwargs=_ak,
                  policy_kwargs=_pk)

    copy_scripts_to_model_path(model_path)

    '''
    INITIALISE NEW MODEL OR LOAD LATEST CHECKPOINT
    '''
    if model_checkpoint is not None:
        print(f'Loading model from {model_checkpoint} and continuing training...')
        model = algo.load(model_checkpoint)
        model.set_env(env)
        reset_num_timesteps = False
    else:
        # model = algo('MlpPolicy', env, tensorboard_log=os.path.join(model_path, 'tb_logs'), policy_kwargs=policy_kwargs, **algo_kw)
        tb_log = os.path.join(model_path, 'tb_logs')
        os.makedirs(tb_log)
        model = algo(policy_type, env, tensorboard_log=tb_log, policy_kwargs=policy_kwargs, **algo_kw)
        reset_num_timesteps = True

    '''
    SETUP TRAINING CALLBACKS AND NEW LOGGER
    '''
    # Stop training if there is no improvement after more than 3 evaluations
    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=100, min_evals=1000, verbose=1)
    checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=model_path)
    eval_callback = EvalCallback(eval_env=eval_env,
                                 n_eval_episodes=n_eval_eps,
                                 eval_freq=eval_freq,
                                 best_model_save_path=os.path.join(model_path, 'best_model'),
                                 log_path=os.path.join(model_path, 'evals'),
                                 callback_after_eval=stop_train_callback,
                                 verbose=1,
                                 norm_rewards=False,
                                 algo=algo_str)

    new_logger = configure(os.path.join(model_path, 'logs'), ['stdout', 'tensorboard'])
    model.set_logger(new_logger)

    '''
    START TRAINING MODEL
    '''
    print(f'\nTraining {model_name} in {model_path}...\n')
    model.learn(total_timesteps=n_train, progress_bar=False, log_interval=log_interval, tb_log_name=model_name,
                callback=[checkpoint_callback, eval_callback], reset_num_timesteps=reset_num_timesteps)


if __name__ == '__main__':
    main()
