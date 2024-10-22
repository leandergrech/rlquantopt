import numpy as np
from rlquantopt.rl_envs.zcqubits import setup_ZCQubits4MKrauss_params
from rlquantopt.rl_envs.zc_qpee import ZCQPEE


def sample_zcqpee_tasks(n_tasks, eps=1e-1, **env_kwargs):
    excluded_kw = ["n_levels", "num_qubits", "coupler_dims", "qubit_dims"]
    envs = []
    for _ in range(n_tasks):
        model_params = setup_ZCQubits4MKrauss_params()
        for k, v in model_params.items():
            if k not in excluded_kw:
                v = np.array(v)
                bound = np.abs(v * eps)
                v = np.array(np.random.uniform(v-bound, v+bound))
                model_params[k] = v.tolist()

        envs.append(ZCQPEE(model_params=model_params, **env_kwargs))
    return envs


if __name__ == '__main__':
    print(sample_zcqpee_tasks(3))


