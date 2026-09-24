"""Load a Stable-Baselines3 (v1) MlpPolicy checkpoint into the JAX ActorCritic.

    model, params = load_sb3_policy("…/rl_model_12566528_steps.zip")
    out = evaluate(model, params, EnvConfig())

Only torch is needed (CPU is fine); SB3 itself is not imported.
"""
import io
import re
import zipfile

import numpy as np
import jax.numpy as jnp

from rlquantopt.jx.agents.common import ActorCritic


def _dense(sd, prefix):
    return {"kernel": jnp.asarray(sd[prefix + ".weight"].T, jnp.float32),
            "bias": jnp.asarray(sd[prefix + ".bias"], jnp.float32)}


def sb3_activation(zip_path):
    """'relu' or 'tanh', read from the policy_kwargs stored in the SB3 zip (SB3's default is tanh)."""
    with zipfile.ZipFile(zip_path) as z:
        data = z.read("data").decode()
    m = re.search(r"activation_fn[^}]*?(ReLU|Tanh)", data)
    return m.group(1).lower() if m else "tanh"


# --8<-- [start:load_sb3_policy]
def load_sb3_policy(zip_path):
    import torch
    with zipfile.ZipFile(zip_path) as z:
        sd = torch.load(io.BytesIO(z.read("policy.pth")), map_location="cpu", weights_only=True)
    sd = {k: v.numpy() for k, v in sd.items()}
    hidden = []
    i = 0
    while f"mlp_extractor.policy_net.{i}.weight" in sd:
        hidden.append(sd[f"mlp_extractor.policy_net.{i}.weight"].shape[0])
        i += 2      # Linear, Tanh, Linear, Tanh, ...
    n_layers = len(hidden)

    def mlp(net, head):
        layers = {f"Dense_{j}": _dense(sd, f"mlp_extractor.{net}.{2 * j}") for j in range(n_layers)}
        layers[f"Dense_{n_layers}"] = _dense(sd, head)
        return {"params": layers}

    act_dim = sd["action_net.weight"].shape[0]
    model = ActorCritic.make(act_dim, tuple(hidden), sb3_activation(zip_path))
    params = dict(actor=dict(net=mlp("policy_net", "action_net"), log_std=jnp.asarray(sd["log_std"], jnp.float32)),
                  critic=mlp("value_net", "value_net"))
    return model, params
# --8<-- [end:load_sb3_policy]


def sb3_predict(zip_path, obs):
    """Reference deterministic action from the torch weights (for tests)."""
    import torch
    act = torch.relu if sb3_activation(zip_path) == "relu" else torch.tanh
    with zipfile.ZipFile(zip_path) as z:
        sd = torch.load(io.BytesIO(z.read("policy.pth")), map_location="cpu", weights_only=True)
    x = torch.as_tensor(np.asarray(obs, np.float32))
    i = 0
    while f"mlp_extractor.policy_net.{i}.weight" in sd:
        x = act(x @ sd[f"mlp_extractor.policy_net.{i}.weight"].T + sd[f"mlp_extractor.policy_net.{i}.bias"])
        i += 2
    return (x @ sd["action_net.weight"].T + sd["action_net.bias"]).numpy()
