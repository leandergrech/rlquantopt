from functools import reduce
from operator import mul

import gymnasium as gym
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Independent, Normal

from collections import OrderedDict

import torch
import torch.nn as nn


def weight_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()


class Policy(nn.Module):
    def __init__(self, input_size, output_size):
        super(Policy, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        # For compatibility with Torchmeta
        self.named_meta_parameters = self.named_parameters
        self.meta_parameters = self.parameters

    def update_params(self, loss, params=None, step_size=0.5, first_order=False):
        """Apply one step of gradient descent on the loss function `loss`, with
        step-size `step_size`, and returns the updated parameters of the neural
        network.
        """
        if params is None:
            params = OrderedDict(self.named_meta_parameters())

        grads = torch.autograd.grad(loss, params.values(), create_graph=not first_order)

        updated_params = OrderedDict()
        for (name, param), grad in zip(params.items(), grads):
            updated_params[name] = param - step_size * grad

        return updated_params


class NormalMLPPolicy(Policy):
    """Policy network based on a multi-layer perceptron (MLP), with a
    `Normal` distribution output, with trainable standard deviation. This
    policy network can be used on tasks with continuous action spaces.
    """

    def __init__(
        self,
        input_size,
        output_size,
        hidden_sizes=(),
        nonlinearity=F.relu,
        init_std=0.5,
        min_std=1e-6,
    ):
        super(NormalMLPPolicy, self).__init__(
            input_size=input_size, output_size=output_size
        )
        self.hidden_sizes = hidden_sizes
        self.nonlinearity = nonlinearity
        self.min_log_std = math.log(min_std)
        self.num_layers = len(hidden_sizes) + 1

        layer_sizes = (input_size,) + hidden_sizes
        for i in range(1, self.num_layers):
            self.add_module(
                "layer{0}".format(i), nn.Linear(layer_sizes[i - 1], layer_sizes[i])
            )

        self.mu = nn.Linear(layer_sizes[-1], output_size)
        self.sigma = nn.Parameter(torch.Tensor(output_size))
        self.sigma.data.fill_(math.log(init_std))

        self.apply(weight_init)

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        output = input
        for i in range(1, self.num_layers):
            output = F.linear(
                output,
                weight=params["layer{0}.weight".format(i)],
                bias=params["layer{0}.bias".format(i)],
            )
            output = self.nonlinearity(output)

        mu = F.linear(output, weight=params["mu.weight"], bias=params["mu.bias"])
        scale = torch.exp(torch.clamp(params["sigma"], min=self.min_log_std))

        return Independent(Normal(loc=mu, scale=scale), 1)



def get_policy_for_env(env, hidden_sizes=(100, 100), nonlinearity="relu"):
    continuous_actions = isinstance(env.action_space, gym.spaces.Box)
    input_size = get_input_size(env)
    nonlinearity = getattr(torch, nonlinearity)

    if continuous_actions:
        output_size = reduce(mul, env.action_space.shape, 1)
        policy = NormalMLPPolicy(
            input_size,
            output_size,
            hidden_sizes=tuple(hidden_sizes),
            nonlinearity=nonlinearity,
        )
    else:
        output_size = env.action_space.n
        policy = CategoricalMLPPolicy(
            input_size,
            output_size,
            hidden_sizes=tuple(hidden_sizes),
            nonlinearity=nonlinearity,
        )
    return policy


def get_input_size(env):
    return reduce(mul, env.observation_space.shape, 1)
