import os
from _ast import expr
from itertools import product
from collections import defaultdict
import numpy as np
from pandas import Series
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle as pkl
from tqdm import tqdm as pbar
from tqdm import tqdm
import yaml
from qutip_qip.circuit import QubitCircuit

from pulses_workspace.utils import grid_on, get_q_func_filenames, get_q_func_xrange, get_val, get_q_func_step, get_latest_experiment, eval_agent
from pulses_workspace.rl_agents.linear_q_function import QValueFunctionLinear
from pulses_workspace.rl_envs.qu_pulse_env import QuPulseEnv


def get_training_params(yaml_path):
    f = open(yaml_path, 'r')
    d = yaml.load(f, yaml.Loader)
    f.close()
    return d


SUB_EXP_COLORS = ['r', 'g', 'b', 'k'] # I never exceed 4 sub-experiments


def create_training_stats(exp_dir, env, eval_eps=1, qvf_type=QValueFunctionLinear):
    print(f'Creating training_stats.pkl for {exp_dir}')
    qfns = get_q_func_filenames(exp_dir)

    rets = defaultdict(list)
    # for qfn in tqdm(qfns[0:len(qfns):5]):
    for qfn in tqdm(qfns):
        q = qvf_type.load(qfn)
        ret = eval_agent(env, q, eval_eps)
        for k, v in ret.items():
            rets[k].append(v)
    for k in rets.keys():
        rets[k] = np.array(rets[k])
    with open(os.path.join(exp_dir, 'training_stats.pkl'), 'wb') as f:
        pkl.dump(rets, f)


def plot_experiment_training_stats(exp_dir, exp_label):
    """
    Access training_stats.pkl for a single experiment and plot training stats.
    Top plot:    Episode length mean+std obtained using latest q-table greedily
    Middle plot: Return mean obtained using latest q-table greedily
    Bottom plot: Regret mean obtained using latest q-table greedily
    """
    stats_file = os.path.join(exp_dir, 'training_stats.pkl')
    training_params = get_training_params(os.path.join(exp_dir, 'train_params.yml'))

    lr_fun = training_params['lr_fun']
    exp_fun = training_params['exp_fun']
    if 'TAU' in exp_fun.label:
        exploration_label='Boltzmann temperature'
    elif 'EPS' in exp_fun.label:
        exploration_label='Epsilon'

    with open(stats_file, 'rb') as f:
        data = pkl.load(f)

    ep_lens = data['ep_lens']
    returns = data['returns']
    # regrets = data['regrets']

    # xrange = [int(os.path.splitext(item)[0].split('_')[-1]) for item in get_q_func_filenames(exp_dir)]
    # xrange = get_q_func_xrange(get_q_func_filenames(exp_dir))
    xrange = np.arange(len(ep_lens)) * 100
    maxx = int(max(xrange))

    fig, axs = plt.subplots(2, gridspec_kw=dict(height_ratios=[3, 3]), figsize=(15, 10))

    ax = axs[0]
    ep_lens_mean = np.mean(ep_lens, axis=1)
    ep_lens_std = np.std(ep_lens, axis=1)
    el_line, = ax.plot(xrange, ep_lens_mean, marker='o', ls='dashed', label=f'{exp_label} Mean')
    ax.set_title('Using greedy policy')
    ax.set_ylabel('Episode length')
    grid_on(ax, 'y', major_loc=20, minor_loc=5, major_grid=True, minor_grid=False)

    ax = axs[1]
    returns_mean = np.mean(returns, axis=1)
    returns_std = np.std(returns, axis=1)
    ret_line, = ax.plot(xrange, returns_mean, marker='o', label=f'{exp_label} Mean')
    # ax.set_yscale('symlog')
    # ax.yaxis.set_major_locator(mpl.ticker.LogLocator(10, (np.arange(2, 10, 2))))

    # ret_ptp = np.ptp(returns_mean)//10
    # ret_ptp = ret_ptp if ret_ptp > 0 else 0.1
    # grid_on(ax, 'y', major_loc=ret_ptp//10, minor_loc=ret_ptp//25, major_grid=True, minor_grid=True)

    ax.set_ylabel('Returns')

    # ax = axs[2]
    # regrets_mean = np.mean(regrets, axis=1)
    # regrets_std = np.std(regrets, axis=1)
    # reg_line, = ax.plot(xrange, regrets_mean, marker='o', label=f'{exp_label} Mean')
    # ax.set_ylabel('Regrets')

    # data_lines = [el_line, ret_line, reg_line]
    data_lines = [el_line, ret_line]

    lrs = [lr_fun(x) for x in xrange]
    exps = [exp_fun(x) for x in xrange]

    for ax in axs:
        ax.set_xlabel('Training steps')
        max_x = max(xrange)
        xtick_labels = np.arange(0, max_x, max_x // 10)
        ax.set_xticks(xtick_labels, xtick_labels)
        ax.legend(loc='best', prop=dict(size=8))
        grid_on(ax, 'x', major_loc=maxx//10, minor_loc=maxx//50)

        axx1 = ax.twinx()
        lr_line, = axx1.plot(xrange, lrs, c='g', lw=2)
        axx1.set_ylabel('Learning rate')
        axx1.yaxis.label.set_color(lr_line.get_color())
        axx1.tick_params(axis='y', colors=lr_line.get_color())

        axx2 = ax.twinx()
        axx2.spines.right.set_position(("axes", 1.05))
        exp_line, = axx2.plot(xrange, exps, c='r', ls=':', lw=2)
        axx2.set_ylabel(exploration_label)
        axx2.yaxis.label.set_color(exp_line.get_color())
        axx2.tick_params(axis='y', colors=exp_line.get_color())

    fig.tight_layout()
    save_name = 'training_stats.png'
    plt.savefig(os.path.join(exp_dir, save_name))
    plt.show()


def plot_experiment_all_subs_training_stats(exp_dir, exp_label):
    """
    Access training_stats.pkl for a single experiment and plot training stats.
    Top plot:    Episode length mean+std obtained using latest q-table greedily
    Middle plot: Return mean obtained using latest q-table greedily
    Bottom plot: Regret mean obtained using latest q-table greedily
    """
    sub_exp_paths = []
    for fn in os.listdir(exp_dir):
        if '.png' not in fn:
            sub_exp_paths.append(os.path.join(exp_dir, fn))

    training_params = get_training_params(os.path.join(sub_exp_paths[0], 'train_params.yml'))

    eval_every = training_params['eval_every']
    lr_fun = training_params['lr_fun']
    exp_fun = training_params['exp_fun']
    if 'TAU' in exp_fun.label:
        exploration_label = 'Boltzmann temperature'
    elif 'EPS' in exp_fun.label:
        exploration_label = 'Epsilon'

    ep_lens = []
    returns = []
    regrets = []

    env_str = None
    for sub_exp in sub_exp_paths:
        if env_str is None:
            for item in os.listdir(sub_exp):
                if 'dynamics' in item:
                    env_str = item.split('_')[:-1]
        with open(os.path.join(sub_exp, 'training_stats.pkl'), 'rb') as f:
            data = pkl.load(f)
            ep_lens.append(data['ep_lens'])
            returns.append(data['returns'])
            regrets.append(data['regrets'])

    xrange = (np.arange(len(ep_lens[0])) + 1) * eval_every
    maxx = max(xrange)
    print(maxx)

    fig, axs = plt.subplots(3, gridspec_kw=dict(height_ratios=[3, 3, 3]), figsize=(15, 10))


    ax = axs[0]
    ep_lens_mean = np.mean(np.mean(ep_lens, axis=-1), axis=0)
    ep_lens_med = np.mean(np.median(ep_lens, axis=-1), axis=0)
    ep_lens_min = np.mean(np.min(ep_lens, axis=-1), axis=0)
    ep_lens_max = np.mean(np.max(ep_lens, axis=-1), axis=0)
    ax.fill_between(xrange, ep_lens_min, ep_lens_max, edgecolor='b', facecolor='None', hatch='//', alpha=0.5)
    el_line, = ax.plot(xrange, ep_lens_mean, ls='solid', label=f'{exp_label} Mean', c='b')
    el_line, = ax.plot(xrange, ep_lens_med, ls='dashed', label=f'{exp_label} Median', c='b')
    ax.set_ylabel('Episode length')
    grid_on(ax, 'y', major_loc=20, minor_loc=5, major_grid=True, minor_grid=False)

    ax = axs[1]
    returns_mean = np.mean(np.mean(returns, axis=-1), axis=0)
    returns_med = np.mean(np.median(returns, axis=-1), axis=0)
    returns_min = np.mean(np.min(returns, axis=-1), axis=0)
    returns_max = np.mean(np.max(returns, axis=-1), axis=0)
    ax.fill_between(xrange, returns_min, returns_max, edgecolor='b', facecolor='None', hatch='//', alpha=0.5)
    ret_line, = ax.plot(xrange, returns_mean, ls='solid', label=f'{exp_label} Mean', c='b')
    ret_line, = ax.plot(xrange, returns_med, ls='dashed', label=f'{exp_label} Median', c='b')
    grid_on(ax, 'y', major_loc=10.0, minor_loc=2., major_grid=True, minor_grid=True)
    ax.set_ylabel('Returns')

    ax = axs[2]
    regrets_mean = np.mean(np.mean(regrets, axis=-1), axis=0)
    regrets_med = np.mean(np.median(regrets, axis=-1), axis=0)
    regrets_min = np.mean(np.min(regrets, axis=-1), axis=0)
    regrets_max = np.mean(np.max(regrets, axis=-1), axis=0)
    ax.fill_between(xrange, regrets_min, regrets_max, edgecolor='b', facecolor='None', hatch='//', alpha=0.5)
    reg_line, = ax.plot(xrange, regrets_mean, ls='solid', label=f'{exp_label} Mean', c='b')
    reg_line, = ax.plot(xrange, regrets_med, ls='dashed', label=f'{exp_label} Median', c='b')
    ax.set_ylabel('Regrets')

    data_lines = [el_line, ret_line, reg_line]

    lrs = [lr_fun(x) for x in xrange]
    exps = [exp_fun(x) for x in xrange]

    for ax in axs:
        ax.set_xlabel('Training steps')
        max_x = max(xrange)
        xtick_labels = np.arange(0, max_x, max_x // 10)
        ax.set_xticks(xtick_labels, xtick_labels)
        ax.legend(loc='best', prop=dict(size=8))
        grid_on(ax, 'x', major_loc=maxx // 10, minor_loc=maxx // 50)

        axx1 = ax.twinx()
        lr_line, = axx1.plot(xrange, lrs, c='g', lw=2)
        axx1.set_ylabel('Learning rate')
        axx1.yaxis.label.set_color(lr_line.get_color())
        axx1.tick_params(axis='y', colors=lr_line.get_color())

        axx2 = ax.twinx()
        axx2.spines.right.set_position(("axes", 1.05))
        exp_line, = axx2.plot(xrange, exps, c='r', ls=':', lw=2)
        axx2.set_ylabel(exploration_label)
        axx2.yaxis.label.set_color(exp_line.get_color())
        axx2.tick_params(axis='y', colors=exp_line.get_color())

    fig.suptitle(f'{len(sub_exp_paths)}x {env_str}')
    fig.tight_layout()
    save_name = 'training_stats.png'
    plt.savefig(os.path.join(exp_dir, save_name))


def plot_all_experiments_training_stats(exp_pardir, exp_subdirs, exp_labels, exp_filter=''):
    """
        Access training_stats.pkl for all experiments found in exp_pardir and
        plot their combined training stats.
        Top plot:    IHT count evolution
        Middle plot: Episode length mean+std obtained using latest q-table greedily
        Bottom plot: Return mean obtained using latest q-table greedily
    """
    regrets = defaultdict(list)
    returns = defaultdict(list)
    ep_lens = defaultdict(list)
    xrange = None
    eval_every = None

    all_exp = []

    # Iterate over experiment with different environment
    for exp_name in sorted(os.listdir(exp_pardir)):
        if 'sarsa' not in exp_name or exp_filter not in exp_name or '.py' in exp_name:
            continue

        if eval_every is None:
            eval_every = 100#get_eval_every_hack(os.path.join(exp_pardir, exp_name, exp_subdirs[0], 'train_params.yml'))

        all_exp.append(exp_name)
        experiment_dir = os.path.join(exp_pardir, exp_name)

        # Iterate over different state initialisation schemes
        for sub_exp in exp_subdirs:
            pkl_file = os.path.join(experiment_dir, sub_exp, 'training_stats.pkl')
            with open(pkl_file, 'rb') as f:
                data = pkl.load(f)

                regrets[sub_exp].append(data['regrets'])
                returns[sub_exp].append(data['returns'])
                ep_lens[sub_exp].append(data['ep_lens'])

                if xrange is None:
                    xrange = np.arange(len(data['regrets'])) * eval_every

    print(f'Found {len(all_exp)} experiments')

    fig, axs = plt.subplots(3, gridspec_kw=dict(height_ratios=[3, 3, 3]), figsize=(15, 10))
    for label, sub_exp, c in zip(exp_labels, exp_subdirs, SUB_EXP_COLORS):
        for dataset, ax, ylabel in zip((ep_lens, returns, regrets), axs, ('Episode length', 'Returns', 'Regrets')):
            data = dataset[sub_exp]
            data_mean = np.mean(np.mean(data, axis=-1), axis=0)
            # data_std = np.sqrt(np.mean(np.square(np.std(data, axis=-1)), axis=0))
            data_min = np.min(np.min(data, axis=-1), axis=0)
            data_max = np.max(np.max(data, axis=-1), axis=0)

            ax.plot(xrange, data_mean, ls='solid', c=c, label=f'{label} ' + r'$\mu$', lw=2, zorder=10)
            # for i, d in enumerate(data):
            #     ax.plot(xrange, np.mean(d, axis=-1), lw=0.5, zorder=15, label=all_exp[i])
            # ax.fill_between(xrange, data_mean - data_std, data_mean + data_std, facecolor='None', edgecolor=c, hatch='//', alpha=0.6)
            ax.fill_between(xrange, data_min, data_max, facecolor='None', edgecolor=c, hatch='//', alpha=0.6)
            # ax.set_title('Using greedy policy')
            ax.set_ylabel(ylabel)

    for ax in axs:
        ax.legend(loc='best', prop=dict(size=10))
        y_grid_on(ax)
        ax.set_xlabel('Training steps')
    fig.suptitle(f'{len(all_exp)} different environments')
    fig.tight_layout()
    plt.savefig(os.path.join(exp_pardir, 'results.png'))
    plt.show()


def plot_episodes(exp_dir, train_step, env, nrows=2, ncols=4, save_dir=None,
                  qvf_type=QValueFunctionLinear):
    if save_dir is None:
        save_dir = exp_dir

    # env = env_type.load_from_dir(exp_dir)
    # if transform_to_env_type is not None:
    #     env = transform_to_env_type(env.obs_dimension, env.act_dimension, state_clip=env.state_clip, model_info=env.model_info)
    q_func_file = os.path.join(exp_dir, 'q_func', f'q_step_{train_step}.pkl')
    # q = QValueFunctionLinear.load(q_func_file)
    q = qvf_type.load(q_func_file)
    n_obs, n_act = env.obs_dimension, env.act_dimension
    init_func = env.reset

    fig, axs = plt.subplots(nrows * 2, ncols, figsize=(20, 15))
    # axs = np.ravel(axs)
    nb_eps = nrows * ncols

    label_fs = 18
    title_fs = 22

    max_ep_len = -np.inf
    for i in range(nb_eps):
        obses = []
        acts = []
        d = False
        o = env.reset(init_func())
        obses.append(o.copy())
        acts.append(np.zeros(n_act))
        step = 1
        while not d:
            a = q.actions[q.greedy_action(o)]
            otp1, _, d, _ = env.step(a)
            o = otp1.copy()

            obses.append(o)
            acts.append(a)

            step += 1
        max_ep_len = max(step, max_ep_len)
        obses = np.array(obses)
        acts = np.array(acts) - 1

        if nb_eps == 1:
            ax_obs = axs[0]
            ax_act = axs[1]
        else:
            ax_obs = axs[(i // ncols) * 2, i % ncols]
            ax_act = axs[(i // ncols) * 2 + 1, i % ncols]

        ax_obs.set_title(f'Ep {i + 1}', size=15)
        ax_obs.axhline(-env.GOAL, c='g', ls='--', lw=2)
        ax_obs.axhline(env.GOAL, c='g', ls='--', lw=2)
        ax_obs.plot(obses, c='b')
        grid_on(ax_obs, 'y', 0.1, 0.02, True, False)

        ax_act.axhline(0.0, c='k', ls='-.', lw=2)
        ax_act.plot(acts, c='r')

        for ax, ylab in zip((ax_obs, ax_act), ('States', 'Actions')):
            print(max_ep_len)
            grid_on(ax, 'x', max_ep_len//5, max_ep_len//25, True, True)
            # ax.set_xticks(np.arange(step))
            ax.set_ylabel(ylab, size=label_fs)

        ax_obs.get_shared_x_axes().join(ax_obs, ax_act)
        ax_act.set_xlabel('Step', size=label_fs)

    fig.suptitle(f'Environment: {repr(env)}\n'
                 f'Experiment:  {exp_dir}\n'
                 f'At step:     {train_step}', size=title_fs)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'{train_step}_step.png'))
    # plt.show()

# def plot_episodes_actfail(exp_dir, train_step, env, nb_failures, nrows=2, ncols=4, save_dir=None,
#                   qvf_type=QValueFunctionLinear):
#     if save_dir is None:
#         save_dir = exp_dir
#
#     # env = env_type.load_from_dir(exp_dir)
#     # if transform_to_env_type is not None:
#     #     env = transform_to_env_type(env.obs_dimension, env.act_dimension, state_clip=env.state_clip, model_info=env.model_info)
#     q_func_file = os.path.join(exp_dir, 'q_func', f'q_step_{train_step}.pkl')
#     # q = QValueFunctionLinear.load(q_func_file)
#     q = qvf_type.load(q_func_file)
#     n_obs, n_act = env.obs_dimension, env.act_dimension
#     init_func = env.reset
#
#     fig, axs = plt.subplots(nrows * 2, ncols, figsize=(15, 10))
#     # axs = np.ravel(axs)
#     nb_eps = nrows * ncols
#
#     label_fs = 18
#     title_fs = 22
#
#     max_ep_len = -np.inf
#     for i in range(nb_eps):
#         obses = []
#         acts = []
#         d = False
#         o = env.reset(init_func())
#         obses.append(o.copy())
#         acts.append(np.zeros(n_act))
#         step = 1
#
#         action_malfunctions = np.random.choice(len(q.actions), nb_failures)
#         action_mask = [1 if i in action_malfunctions else 0 for i in range(len(q.actions))]
#         while not d:
#             action_idx = q.greedy_action(o)
#             if action_idx in action_malfunctions:
#                 for _ in range(int(1/env.ACTION_EPS)):
#                     *_ = env.step()
#             a = q.actions[action_idx]
#             otp1, _, d, _ = env.step(a)
#             o = otp1.copy()
#
#             obses.append(o)
#             acts.append(a)
#
#             step += 1
#         max_ep_len = max(step, max_ep_len)
#         obses = np.array(obses)
#         acts = np.array(acts) - 1
#
#         if nb_eps == 1:
#             ax_obs = axs[0]
#             ax_act = axs[1]
#         else:
#             ax_obs = axs[(i // ncols) * 2, i % ncols]
#             ax_act = axs[(i // ncols) * 2 + 1, i % ncols]
#
#         ax_obs.set_title(f'Ep {i + 1}', size=15)
#         ax_obs.axhline(-env.GOAL, c='g', ls='--', lw=2)
#         ax_obs.axhline(env.GOAL, c='g', ls='--', lw=2)
#         ax_obs.plot(obses, c='b')
#         grid_on(ax_obs, 'y', 0.1, 0.02, True, False)
#
#         ax_act.axhline(0.0, c='k', ls='-.', lw=2)
#         ax_act.plot(acts, c='r')
#
#         for ax, ylab in zip((ax_obs, ax_act), ('States', 'Actions')):
#             print(max_ep_len)
#             grid_on(ax, 'x', max_ep_len//5, max_ep_len//25, True, True)
#             # ax.set_xticks(np.arange(step))
#             ax.set_ylabel(ylab, size=label_fs)
#
#         ax_obs.get_shared_x_axes().join(ax_obs, ax_act)
#         ax_act.set_xlabel('Step', size=label_fs)
#
#     fig.suptitle(f'Environment: {repr(env)}\n'
#                  f'Experiment:  {exp_dir}\n'
#                  f'At step:     {train_step}', size=title_fs)
#     fig.tight_layout()
#     fig.savefig(os.path.join(save_dir, f'{train_step}_step_{nb_failures}_action_malfunction.png'))


def plot_weight_evolution(exp_dir, save_dir=None):
    if save_dir is None:
        save_dir = exp_dir

    q_func_fns = get_q_func_filenames(exp_dir)
    x = get_q_func_xrange(q_func_fns)
    maxx = max(x)

    weights = None
    actions = None
    for qfn in q_func_fns:
        q = QValueFunctionLinear.load(qfn)
        if actions is None:
            actions = q.actions
        if weights is None:
            weights = np.expand_dims(q.w, axis=-1)
        else:
            weights2 = np.expand_dims(q.w, axis=-1)
            weights = np.concatenate([weights, weights2], axis=-1)

    if len(weights.shape) > 2:
        fig, axs = plt.subplots(3, 3, figsize=(20, 12))
        axs = np.ravel(axs)
    else:
        fig, ax = plt.subplots(figsize=(20, 12))
        axs = np.repeat([ax], len(weights))

    for i, (ax, per_action_weights) in enumerate(zip(axs, weights)):
        if len(per_action_weights.shape) == 2:
            for j, per_action_w in enumerate(per_action_weights):
                ax.plot(x, per_action_w, label=f'w_{j}')
        else:
            ax.plot(x, per_action_weights, label=f'w_{i}')
    for ax, a in zip(axs, actions):
        # grid_on(ax, 'x', maxx//10, maxx//50)
        # grid_on(ax, 'y', 1, 0.2)
        ax.axhline(0.0, ls='-.', lw=2, c='k')
        ax.set_title(np.subtract(a, 1), size=18)
        ax.set_xlabel('Training step', size=15)
        ax.set_ylabel('Weight', size=15)
        ax.legend(loc='best')

    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, 'tracked_weights.png'))


# def create_grid_tracking_states(env, n_dim):
#     """
#     Utility function to create grid of states for passed env. Each env state
#     dimension is split into n_dim parts. Returns 1D list with n_dim**n_obs
#     total states.
#     """
#     tracking_ranges = [[l, h] for l, h in zip(env.observation_space.low, env.observation_space.high)]
#     tracking_states = np.array(
#         [list(item) for item in product(*np.array([np.linspace(l, h, n_dim) for l, h in tracking_ranges]))])
#
#     return tracking_states


# def plot_q_vals_grid_tracking_states(experiment_dir, n_tracking_dim, env_type, save_dir=None):
#     """
#     Create a grid of states within the environment state limits, access the q-table for every eval
#     step during training, and plot the evolution of the tracked q-values during training.
#     """
#     if save_dir is None:
#         save_dir = experiment_dir
#
#     experiment_name = os.path.split(experiment_dir)
#
#     env = env_type.load_from_dir(experiment_dir)
#     actions = get_discrete_actions(env.act_dimension, 3)
#     nb_actions = len(actions)
#
#     q_func_filenames = get_q_func_filenames(experiment_dir)
#     nb_q_funcs = len(q_func_filenames)
#     xrange = get_q_func_xrange(q_func_filenames)
#
#     # Initialise grid tracking states
#     tracking_states = create_grid_tracking_states(env, n_tracking_dim)
#     nb_tracked = len(tracking_states)
#
#     vals = np.zeros(shape=(nb_tracked, nb_q_funcs))
#     for j, qfn in enumerate(q_func_filenames):
#         q = QValueFunctionLinear.load(qfn)
#         for i, ts in enumerate(tracking_states):
#             vals[i, j] = get_val(q, ts, nb_actions)
#
#     # Initialise figure
#     fig, axs = plt.subplots(2, gridspec_kw=dict(height_ratios=[1, 2]))
#     cmap = mpl.cm.get_cmap('tab10')
#     cmap_x = np.linspace(0, 1, nb_tracked)
#
#     # Plot tracked states
#     ax = axs[0]
#     ax.set_xlabel('State dimension 0')
#     ax.set_ylabel('State dimension 1')
#     for i, ts in enumerate(tracking_states):
#         ax.scatter(ts[0], ts[1], marker='x', c=cmap(cmap_x[i]))
#
#     # Plot evolution of q_values for tracked states
#     ax = axs[1]
#     ax.set_xlabel('Training steps')
#     ax.set_ylabel('Estimated values')
#     for i, val in enumerate(vals):
#         ax.plot(xrange, val, c=cmap(cmap_x[i]), lw=1.0)
#
#     fig.suptitle(experiment_name)
#     fig.tight_layout()
#     plt.savefig(os.path.join(save_dir, 'tracked_vals.png'))
#     plt.show()


def plot_q_vals_region_sampling_tracking_states(experiment_dir, env, save_dir=None, nb_regions=5,
                                                nb_samples_per_region=100):
    if save_dir is None:
        save_dir = experiment_dir

    n_obs = env.obs_dimension

    # Searching for training data
    q_func_filenames = get_q_func_filenames(experiment_dir)
    xrange = get_q_func_xrange(q_func_filenames)

    # Creating tracking regions
    nb_samples_per_region = 200
    rstep = 1 / nb_regions

    tracking_states = []
    for r in np.arange(nb_regions):
        for i in range(nb_samples_per_region):
            ts = nball_uniform_sample(n_obs, rlow=r * rstep, rhigh=r * rstep + rstep)
            tracking_states.append(ts)
    nb_tracked = len(tracking_states)

    # Estimating values at tracked states
    tracked_vals = [[] for _ in range(nb_tracked)]  # shape=(nb_tracked, len(q_func_filenames))
    for qfn in pbar(q_func_filenames):
        # q = QValueFunctionLinearEfficient.load(qfn)
        q = QValueFunctionLinear.load(qfn)
        for i, ts in enumerate(tracking_states):
            tracked_vals[i].append(get_val(q, ts, len(q.actions)))

    # Plotting
    fig, axs = plt.subplots(2, figsize=(15, 10))
    fig.suptitle(experiment_dir)
    cmap = mpl.cm.get_cmap('jet')

    # Gives color to each region. Regions are stored contiguously
    def region_color(idx):
        return cmap((idx // nb_samples_per_region)
                    * nb_samples_per_region / (nb_tracked - nb_samples_per_region))

    # Plot tracked states, color-coded per region
    ax = axs[0]
    ax.set_xlabel('State dimension 0')
    ax.set_ylabel('State dimension 1')
    for i, ts in enumerate(tracking_states):
        ax.scatter(ts[0], ts[1], c=region_color(i), marker='x')

    # Plot evolution of q values stats, color-coded per region
    ax = axs[1]
    ax.set_xlabel('Training steps')
    ax.set_ylabel('Estimated values')
    for val_idx in np.arange(0, nb_tracked, nb_samples_per_region):
        vals = tracked_vals[val_idx:val_idx + nb_samples_per_region]  # split per region
        vals_min = np.min(vals, axis=0)
        vals_max = np.max(vals, axis=0)
        vals_mean = np.mean(vals, axis=0)
        vals_std = np.std(vals, axis=0)
        vals_lower = vals_mean - vals_std
        vals_upper = vals_mean + vals_std
        label = 'Mean' if val_idx == 0 else None
        ax.plot(xrange, vals_mean, c=region_color(val_idx), lw=1.0, label=label)
        c = region_color(val_idx)
        label = '$min\\rightarrow -\sigma$' if val_idx == 0 else None
        ax.fill_between(xrange, vals_min, vals_lower, edgecolor=c, facecolor='None', hatch='//', label=label, alpha=0.4)
        label = '$-\sigma\\rightarrow\sigma$' if val_idx == 0 else None
        ax.fill_between(xrange, vals_lower, vals_upper, facecolor=c, alpha=0.2, label=label)
        label = '$\sigma\\rightarrow max$' if val_idx == 0 else None
        ax.fill_between(xrange, vals_upper, vals_max, edgecolor=c, facecolor='None', hatch='\\\\', label=label,
                        alpha=0.4)
    ax.legend(loc='best')
    plt.savefig(os.path.join(save_dir, 'tracked_vals_per_region.png'))
    plt.show()


def load_agent_traning_evals(exp_path):
    evals_path = os.path.join(exp_path, 'evals')
    eval_files = [os.path.abspath(os.path.join(evals_path, item)) for item in os.listdir(evals_path)]

    eval_file = eval_files[0]
    with open(eval_file, 'rb') as f:
        eval_data = pkl.load(f)

    print([i for i in eval_data.keys()])
    eval_ep_lens = eval_data['eval_ep_lens']
    eval_ep_successes = eval_data['eval_ep_success']
    eval_ep_returns = eval_data['eval_ep_rets']
    eval_every = eval_data['eval_every']

    nb_evals = eval_ep_lens.shape[0]

    fig, axs = plt.subplots(3)
    xrange = np.arange(0, nb_evals * eval_every, eval_every)

    ax = axs[0]
    ave_er = np.mean(eval_ep_returns, axis=1)
    std_er = np.std(eval_ep_returns, axis=1)
    ax.fill_between(xrange, ave_er - std_er, ave_er + std_er, alpha=0.4, label='$1\sigma$')
    ax.plot(xrange, ave_er, ls='dashed', label='Mean')
    ax.legend(loc='best')
    ax.set_ylabel('Eval returns')

    ax = axs[1]
    ave_el = np.mean(eval_ep_lens, axis=1)
    std_el = np.std(eval_ep_lens, axis=1)
    ax.fill_between(xrange, ave_el - std_el, ave_el + std_el, alpha=0.4, label='$1\sigma$')
    ax.plot(xrange, ave_el, ls='dashed', label='Mean')
    ax.legend(loc='best')
    ax.set_ylabel('Eval ep. lengths')

    ax = axs[2]

    ax.plot(xrange, eval_ep_successes)
    ax.set_ylabel('Eval successes')

    for ax in axs:
        ax.set_xlabel('Training steps')

    plt.show()


if __name__ == '__main__':
    exp_dir = get_latest_experiment('..', 'qupulseenv_sarsa')

    qc = QubitCircuit(N=1)
    qc.add_gate('X', targets=0)
    env = QuPulseEnv(qc)

    # # train_step = 108800
    # train_step = 49
    # plot_episodes(exp_dir=exp_dir, train_step=train_step, env=env, save_dir=exp_dir, nrows=2, ncols=2)

    # # Plot weight evolition during training
    # plot_weight_evolution(exp_dir, save_dir=exp_dir)
    #
    # # Create fresh evaluation episodes using saved Q-tables from file during training
    # create_training_stats(exp_dir, env=env, eval_eps=4)
    # # Plot the training performance from the new evaluation episodes
    # plot_experiment_training_stats(exp_dir, 'Linear RL on QuPulseEnv')

    # Plot training evaluation stats
    load_agent_traning_evals(exp_dir)
