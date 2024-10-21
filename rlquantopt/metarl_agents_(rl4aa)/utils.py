import numpy as np
import torch
from torch.distributions import Categorical, Independent, Normal
from torch.nn.utils.convert_parameters import _check_param_device


def conjugate_gradient(f_Ax, b, cg_iters=10, residual_tol=1e-10):
    p = b.clone().detach()
    r = b.clone().detach()
    x = torch.zeros_like(b).float()
    rdotr = torch.dot(r, r)

    for _ in range(cg_iters):
        z = f_Ax(p).detach()
        v = rdotr / torch.dot(p, z)
        x += v * p
        r -= v * z
        newrdotr = torch.dot(r, r)
        mu = newrdotr / rdotr
        p = r + mu * p

        rdotr = newrdotr
        if rdotr.item() < residual_tol:
            break

    return x.detach()


def weighted_mean(tensor, lengths=None):
    if lengths is None:
        return torch.mean(tensor)
    if tensor.dim() < 2:
        raise ValueError(
            "Expected tensor with at least 2 dimensions "
            "(trajectory_length x batch_size), got {0}D "
            "tensor.".format(tensor.dim())
        )
    for i, length in enumerate(lengths):
        tensor[length:, i].fill_(0.0)

    extra_dims = (1,) * (tensor.dim() - 2)
    lengths = torch.as_tensor(lengths, dtype=torch.float32)

    out = torch.sum(tensor, dim=0)
    out.div_(lengths.view(-1, *extra_dims))

    return out


def weighted_normalize(tensor, lengths=None, epsilon=1e-8):
    mean = weighted_mean(tensor, lengths=lengths)
    out = tensor - mean.mean()
    for i, length in enumerate(lengths):
        out[length:, i].fill_(0.0)

    std = torch.sqrt(weighted_mean(out**2, lengths=lengths).mean())
    out.div_(std + epsilon)

    return out


def detach_distribution(pi):
    if isinstance(pi, Independent):
        distribution = Independent(
            detach_distribution(pi.base_dist), pi.reinterpreted_batch_ndims
        )
    elif isinstance(pi, Categorical):
        distribution = Categorical(logits=pi.logits.detach())
    elif isinstance(pi, Normal):
        distribution = Normal(loc=pi.loc.detach(), scale=pi.scale.detach())
    else:
        raise NotImplementedError(
            "Only `Categorical`, `Independent` and "
            "`Normal` policies are valid policies. Got "
            "`{0}`.".format(type(pi))
        )
    return distribution


def to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, (tuple, list)):
        return np.stack([to_numpy(t) for t in tensor], axis=0)
    else:
        raise NotImplementedError()


def vector_to_parameters(vector, parameters):
    param_device = None

    pointer = 0
    for param in parameters:
        param_device = _check_param_device(param, param_device)

        num_param = param.numel()
        param.data.copy_(vector[pointer : pointer + num_param].view_as(param).data)

        pointer += num_param


# def value_iteration(transitions, rewards, gamma=0.95, theta=1e-5):
#     rewards = np.expand_dims(rewards, axis=2)
#     values = np.zeros(transitions.shape[0], dtype=np.float32)
#     delta = np.inf
#     while delta >= theta:
#         q_values = np.sum(transitions * (rewards + gamma * values), axis=2)
#         new_values = np.max(q_values, axis=1)
#         delta = np.max(np.abs(new_values - values))
#         values = new_values
#     return values
#
#
# def value_iteration_finite_horizon(transitions, rewards, horizon=10, gamma=0.95):
#     rewards = np.expand_dims(rewards, axis=2)
#     values = np.zeros(transitions.shape[0], dtype=np.float32)
#     for k in range(horizon):
#         q_values = np.sum(transitions * (rewards + gamma * values), axis=2)
#         values = np.max(q_values, axis=1)
#
#     return values


# def get_returns(episodes):
#     return to_numpy([episode.rewards.mean(dim=0) for episode in episodes])
def get_returns(episodes):
    return to_numpy([episode.rewards.sum(dim=0) for episode in episodes])


def get_final_returns(episodes):
    return to_numpy([episode.rewards[-1] for episode in episodes])


def get_episode_lengths(episodes):
    return [episode.lengths for episode in episodes]


def reinforce_loss(policy, episodes, params=None):
    pi = policy(
        episodes.observations.view((-1, *episodes.observation_shape)), params=params
    )

    log_probs = pi.log_prob(episodes.actions.view((-1, *episodes.action_shape)))
    log_probs = log_probs.view(len(episodes), episodes.batch_size)

    losses = -weighted_mean(log_probs * episodes.advantages, lengths=episodes.lengths)

    return losses.mean()
