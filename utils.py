import os
from abc import ABC
import numpy as np
from copy import deepcopy
import yaml

from matplotlib.ticker import MultipleLocator


def argmax(arr):
    return max((x, i) for i, x in enumerate(arr))[1]


class QFuncBaseClass(ABC):
    def __init__(self, *args, **kwargs):
        self.n_discrete_actions = None

    def value(self, state, action_idx: int):
        raise NotImplementedError

    def update(self, state, action_idx, target, lr):
        raise NotImplementedError

    def greedy_action(self, state):
        return argmax([self.value(state, a_) for a_ in range(self.n_discrete_actions)])

    def save(self, save_path):
        raise NotImplementedError

    @staticmethod
    def load(load_path):
        raise NotImplementedError


def eval_agent(eval_env, q: QFuncBaseClass, nb_eps: int, init_func=None, get_obses=False):
    max_steps = 5
    if get_obses:
        init_obses = np.empty(shape=(0, eval_env.obs_dimension))
        terminal_obses = np.empty(shape=(0, eval_env.obs_dimension))

    if init_func is None:
        init_func = eval_env.reset

    # opt_env = deepcopy(eval_env)

    # regrets = np.zeros(nb_eps)
    ep_lens = np.zeros(nb_eps)
    returns = np.zeros(nb_eps)
    successes = np.zeros(nb_eps)

    actions = q.actions
    for ep in range(nb_eps):
        o = eval_env.reset(options=dict(init_state=init_func()))
        if get_obses:
            init_obses = np.vstack([init_obses, o. copy()])
        d = False
        t = 0
        g = 0
        # regret = 0
        while (not d) and t < max_steps:
            a = q.greedy_action(o)
            otp1, r, d, info = eval_env.step(actions[a])

            # opt_env.reset(o.copy())
            # _, ropt, *_ = opt_env.step(opt_env.get_optimal_action(o))

            # regret += ropt - r

            g += r
            o = otp1
            t += 1
        if get_obses:
            terminal_obses = np.vstack([terminal_obses, o])
        ep_lens[ep] = t
        returns[ep] = g
        successes[ep] = info['success']
        # regrets[ep] = regret
    if get_obses:
        obses = dict(initial=init_obses, terminal=terminal_obses)
        return dict(obses=obses, successes=successes, returns=returns, ep_lens=ep_lens)#, regrets=regrets)
    else:
        # return dict(returns=returns, ep_lens=ep_lens, regrets=regrets)
        return dict(successes=successes, returns=returns, ep_lens=ep_lens)


def init_label(label):
    if label:
        return f'{label}_'
    else:
        return ''


class LinearDecay(yaml.YAMLObject):
    yaml_tag = '!LinearDecay'

    def __init__(self, init, final, decay_steps, label=None):
        self.init = init
        self.final = final
        self.decay_steps = decay_steps
        self.label = init_label(label)

    def __call__(self, t):
        return self.final + (self.init - self.final) * max(0, 1 - t/self.decay_steps)

    def __repr__(self):
        return f'{self.label}LinearDecay_init-{self.init}_final-{self.final}_decaysteps-{self.decay_steps}'

    @classmethod
    def to_yaml(cls, dumper, data):
        return dumper.represent_mapping(cls.yaml_tag, {
            'init': data.init,
            'final': data.final,
            'decay_steps': data.decay_steps,
            'label': data.label
        })

    @classmethod
    def from_yaml(cls, loader, node):
        d = loader.construct_mapping(node)
        return LinearDecay(init=d['init'], final=d['final'],
                           decay_steps=d['decay_steps'], label=d['label'])


yaml.add_representer(LinearDecay, LinearDecay.to_yaml)
yaml.add_constructor(LinearDecay.yaml_tag, LinearDecay.from_yaml)


def eps_greedy(state: np.ndarray, qfunc: QFuncBaseClass, epsilon: float) -> int:
    nb_actions = qfunc.n_discrete_actions
    if np.random.rand() < epsilon:
        return np.random.choice(nb_actions)
    else:
        return qfunc.greedy_action(state)


def boltzmann(state: np.ndarray, qfunc: QFuncBaseClass, tau: float) -> int:
    nb_actions = qfunc.n_discrete_actions
    qvals_exp = np.exp([qfunc.value(state, a_) / tau for a_ in range(nb_actions)])
    qvals_exp_sum = np.sum(qvals_exp)

    cum_probas = np.cumsum(qvals_exp / qvals_exp_sum)
    return np.searchsorted(cum_probas, np.random.rand())


def recolor_yaxis(ax, c):
    ax.spines.right.set_color(c)
    ax.yaxis.label.set_color(c)
    ax.tick_params(axis='y', colors=c)


def grid_on(ax, axis='y', major_loc=None, minor_loc=None, major_grid=True, minor_grid=True):
    if axis == 'y':
        axis_ = ax.yaxis
    else:
        axis_ = ax.xaxis

    if major_loc is not None:
        axis_.set_major_locator(MultipleLocator(major_loc))
    if minor_loc is not None:
        axis_.set_minor_locator(MultipleLocator(minor_loc))
    ax.minorticks_on()
    if major_grid:
        ax.grid(which='major', c='gray', axis=axis, alpha=0.5)
    if minor_grid:
        ax.grid(which='minor', c='gray', ls='--', alpha=0.5, axis=axis)


def get_q_func_step(fn):
    """
        Every experiment contains q_func directory which stores the QValueFunctionTiles3
        instance at a specified training step. Training step X obtained from filename fn
        in the form q_step_X.pkl.
    """
    return int(os.path.splitext(fn)[0].split('_')[-1])


def get_q_func_filenames(experiment_dir):
    """
        Every experiment has q_func directory. Get sorted filenames found within.
    """
    q_func_dir = os.path.join(experiment_dir, 'q_func')
    q_func_filenames = [fn for fn in os.listdir(q_func_dir)]

    q_func_filenames = sorted(q_func_filenames, key=get_q_func_step)
    q_func_filenames = [os.path.join(q_func_dir, item) for item in q_func_filenames]

    return q_func_filenames


def get_q_func_xrange(q_func_filenames):
    """
        Given a sorted list of q-table files stored during an experiment, return
        the xrange to be used for the plotting x-axis.
    """
    return [int(os.path.splitext(item)[0].split('_')[-1]) for item in q_func_filenames]


def get_val(qvf: QFuncBaseClass, state, nb_actions):
    """
        Value is defined as the expected return given a state. QValueFunctionTiles3
        only gives us Q-values. I'm assuming the value is the average of all q-values
        obtained from all possible actions.
    """
    return np.max([qvf.value(state, a_) for a_ in range(nb_actions)])


def get_latest_experiment(lab_dir, pattern='sarsa', offset=0):
    experiments = []
    for fn in os.listdir(lab_dir):
        if pattern in fn:
            experiments.append(fn)
    experiments = sorted(experiments)

    experiment_name = experiments[-1-offset]

    return os.path.join(lab_dir, experiment_name)