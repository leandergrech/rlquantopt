import os
import numpy as np
import pickle as pkl
from pulses_workspace.utils import argmax, QFuncBaseClass


class FeatureExtractor:
    def __init__(self, env):
        self.n_obs = env.observation_space.shape[0]
        self.n_act = len(env.action_space)
        self._n_features = None

    def _get_feature(self, state):
        r = np.sqrt(np.sum(np.square(state)))

        return np.array([item for item in (*state, r)])

    @property
    def n_features(self):
        if self._n_features is None:
            test_state = np.zeros(self.n_obs)
            self._n_features = len(self._get_feature(test_state))
        return self._n_features

    def __call__(self, state, action, **kwargs):
        return self._get_feature(state)


class QValueFunctionLinear(QFuncBaseClass):
    def __init__(self, feature_fn: FeatureExtractor, actions: list):
        """
        param feature_fn: FeatureExtractor instance which converts continuous state to features
        param actions: List of all possible actions
        param lr: Generator that yields learning rate
        """
        super(QValueFunctionLinear, self).__init__()
        self.feature_fn = feature_fn
        self.n_features = feature_fn.n_features

        self.actions = actions
        self.n_discrete_actions = len(actions)
        self.n_act = len(actions[0])

        # self.w = np.zeros(self.n_features)
        self.w = np.ones((self.n_discrete_actions, self.n_features))
        # self.w = np.random.normal(0.0, 0.1, size=(self.n_discrete_actions, self.n_features))

        self.nb_updates = 0

    def reset_weights(self):
        self.w = np.random.normal(0.0, 0.1, size=self.w.shape)

    def greedy_action(self, state):
        features = self.feature_fn(state, action=None).reshape(1, -1)
        features = np.repeat(features, self.n_discrete_actions, axis=0)
        # def npdot(w):
        #     return w.dot(features)
        # with Pool(4) as p:
        #     return argmax(p.map(npdot, self.w))
        return argmax([item for item in map(np.dot, self.w, features)])

    def value(self, state, action_idx):
        features = self.feature_fn(state, self.actions[action_idx])
        # return self.w.dot(features)
        return self.w[action_idx].dot(features)

    def update(self, state, action_idx, target, lr):
        self.nb_updates += 1

        features = self.feature_fn(state, self.actions[action_idx])
        error = (target - self.value(state, action_idx)) * features

        self.w[action_idx] += lr * error

        return np.mean(error)

    def save(self, save_path):
        with open(save_path, 'wb') as f:
            pkl.dump(dict(
                feature_fn=self.feature_fn,
                actions=self.actions,
                w=self.w,
                nb_updates=self.nb_updates
            ), f)

    @classmethod
    def load(cls, load_path):
        if os.path.exists(load_path):
            with open(load_path, 'rb') as f:
                d = pkl.load(f)
                feature_fn = d['feature_fn']
                actions = d['actions']
                self = cls(feature_fn, actions)
                self.w = d['w']
                self.nb_updates = d['nb_updates']
                return self
        else:
            raise FileNotFoundError(f'Path passed: {load_path}, does not exist.')
