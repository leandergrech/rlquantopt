import os
import argparse
import torch
import multiprocessing as mp
from datetime import datetime as dt
from torch.utils.tensorboard import SummaryWriter
from sb3_contrib import TRPO

from rlquantopt.rl_envs.zc_qpee import ZCQPEE
from rlquantopt.meta_rl_agents.reptile import reptile_meta_learning_zcqpee
from rlquantopt.meta_rl_agents.vec_reptile import vec_reptile_meta_learning_zcqpee
from rlquantopt.meta_rl_agents.train_utils import get_initial_trpo_params, save_checkpoint, load_checkpoint

DT_FMT_STR = '%d-%m-%y_%H%M%S'


def parse_args():
    parser = argparse.ArgumentParser('RLQuantOpt - meta-training RL agent on ZCQPEE environment using Reptile')
    # ZCQPEE parameters
    parser.add_argument('-p', '--pulse-length', default=500, type=int, help='Maximum number of samples in a pulse')
    # parser.add_argument('-d', '--delta-mode', action='store_true', help='Use ZCQPEE environment in delta action mode')
    parser.add_argument('-T', '--max-time-ns', default=300.0, type=float, help='Pulse duration in ns')
    parser.add_argument('-N', '--n-time-steps', default=3, type=int, help='Number of time-steps per action')
    parser.add_argument('--tv-penalty-scale', default=1e-1, type=float, help='Number of time-steps per action')
    parser.add_argument('-o', '--act-poly-order', default=1, type=float, help='Order of action transformation polynomial ')
    parser.add_argument('-a', '--add-prev-obs', action='store_true', help='Add previous observable w/o the action and time, to the current observable.')
    parser.add_argument('--a-scale', default=0.5, type=float, help='Set action scaling during normalisation')
    parser.add_argument('--a-norm-max', default=1.0, type=float, help='Set the maximum normalised amplitude when using delta-mode')
    parser.add_argument('--rew-scale', default=1., type=float, help='Multiply reward by rew_scale after each step')
    parser.add_argument('--fid-thresh', default=0.99, type=float, help='Set goal fidelity threshold')

    # RL agent paramters
    parser.add_argument('--algo', type=str, default='TRPO', help='Type of RL agent')
    n_steps = 2048
    n_envs = 8
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='TRPO learning rate')
    parser.add_argument('--n-steps', type=int, default=n_steps, help='The number of steps to run for each environment per update'
                             '(i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)'
                             'NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)'
                             'See https://github.com/pytorch/pytorch/issues/29372')
    parser.add_argument('--n-epochs', type=int, default=10, help='Number of epoch when optimizing the surrogate loss')
    parser.add_argument('--batch-size', type=int, default=128, help='Mini-batch size')
    parser.add_argument('--verbose', type=int, default=1, help='Verbosity')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--seed', default=123, type=int, help='Set random seed')

    # Training parameters
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--msg', default='', type=str, help='User message to add to info.txt')
    parser.add_argument('-L', '--hidden-layer-size', default=128, type=int, help='Network hidden layer size. All layers are equal size')
    parser.add_argument('-H', '--n-hidden-layers', default=2, type=int, help='Nb. of hidden layers in last layer MLP')

    return parser.parse_args()


def main():
    args = parse_args()

    # exp_path = f"{dt.now().strftime(DT_FMT_STR)}"
    exp_path = "22-10-24_134015"
    os.makedirs(exp_path, exist_ok=True)
    cur_chkpt = 0
    retrain = True

    # Define environment parameters
    env_kw = dict(pulse_length=int(args.pulse_length),
                  rew_scale=float(args.rew_scale),
                  delta_mode=True,
                  T=float(args.max_time_ns),
                  fid_thresh=float(args.fid_thresh),
                  action_scaling={'z': float(args.a_scale)},
                  a_norm_max=float(args.a_norm_max),
                  n_time_steps=args.n_time_steps,
                  tv_penalty_scale=args.tv_penalty_scale,
                  act_poly_order=args.act_poly_order,
                  add_prev_obs=args.add_prev_obs)

    # Setting device on the CPU only for now since I am working on my laptop
    if args.no_cuda:
        device = 'cpu'
    else:
        device = 'cuda'


    net_layers = [args.hidden_layer_size] * args.n_hidden_layers
    policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                         net_arch=dict(pi=net_layers, vf=net_layers))
    trpo_kw = dict(batch_size=args.batch_size, n_steps=args.n_steps, learning_rate=args.learning_rate, device=device,
                   gamma=args.gamma, seed=args.seed, verbose=args.verbose, policy_kwargs=policy_kwargs)




    # Initialize the policy parameters and TRPO hyperparameters
    if not retrain:
        initial_policy_params = get_initial_trpo_params(env=ZCQPEE(**env_kw), algo=TRPO, algo_kw=trpo_kw)  # Assuming no pre-trained parameters
    else:
        initial_policy_params = load_checkpoint(os.path.join(exp_path, f'chkpt_{cur_chkpt}'))

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(exp_path, 'logs'))

    # Meta-learn the policy
    # final_meta_policy = reptile_meta_learning_zcqpee(
    #     initial_policy_params=initial_policy_params,
    #     trpo_kw=trpo_kw,
    #     save_dir=exp_path,
    #     writer=writer,
    #     device=device,
    #     # n_tasks=1,
    #     # inner_steps=1,
    #     # trpo_n_envs=1,
    #     **env_kw
    # )

    mp.set_start_method('spawn')

    final_meta_policy = vec_reptile_meta_learning_zcqpee(
        initial_policy_params=initial_policy_params,
        trpo_kw=trpo_kw,
        save_dir=exp_path,
        writer=writer,
        trpo_n_envs=1,
        save_every=1,
        n_tasks=9,
        cur_chkpt=cur_chkpt,
        n_eval_envs=1,
        eval_eps=5e-2,
        n_parallel_tasks=3,
        meta_iterations=1000,
        inner_steps=10,
        meta_step_size=0.1,
        eps=1e-1,
        device=device,
        **env_kw
    )


if __name__ == '__main__':
    main()
