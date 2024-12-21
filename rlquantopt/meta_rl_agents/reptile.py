import os
import copy
from sys import prefix

import torch
from sb3_contrib import TRPO

from rlquantopt.meta_rl_agents.sampler import sample_zcqpee_tasks
from rlquantopt.meta_rl_agents.train_utils import train_agent, save_checkpoint, copy_params
from rlquantopt.meta_rl_agents.eval_reptile_agent import evaluate_meta_policy, log_metrics


def reptile_meta_learning_zcqpee(initial_policy_params,
                                 trpo_kw,
                                 save_dir,
                                 writer,
                                 trpo_n_envs=8,
                                 save_every=10,
                                 n_tasks=10,
                                 n_eval_envs=5,
                                 eval_eps=5e-2,  # Let's make validation easier for now
                                 meta_iterations=1000,
                                 inner_steps=10,
                                 meta_step_size=0.1,
                                 eps=1e-1,
                                 device='cpu',
                                 **env_kwargs):
    meta_policy_params = copy.deepcopy(initial_policy_params)

    networks = ('pi', 'vf')

    try:
        for it in range(meta_iterations):
            print(f'-> Iteration {it}')
            print(f'\t`-> Sampling {n_tasks} tasks')
            task_batch = sample_zcqpee_tasks(n_tasks, eps, **env_kwargs)

            meta_policy_update = dict()
            for network in networks:
                meta_policy_update[network] = {}
                for key in meta_policy_params[network]:
                    meta_policy_update[network][key] = torch.zeros_like(meta_policy_params[network][key], dtype=float).to(device)

            print(f'\t`-> Training inner loops:')
            for env in task_batch:
                print(f'\t\t`-> {env.model_str()}')
                # Copy initial policy parameters and train agent for `inner_steps` steps
                task_policy_params = copy_params(meta_policy_params)

                task_policy_params = train_agent(policy_params=task_policy_params,
                                                 env=env,
                                                 algo=TRPO,
                                                 algo_kw=trpo_kw,
                                                 steps=inner_steps,
                                                 n_envs=trpo_n_envs)

                # Calculate the parameter differences for the meta-update
                for network in networks:
                    for key in meta_policy_params[network]:
                        meta_policy_update[network][key] += task_policy_params[network][key] - meta_policy_params[network][key]

            # # Regularise and update meta policy parameters
            for network in networks:
                for key in meta_policy_params[network]:
                    meta_policy_update[network][key] /= n_tasks
                    meta_policy_params[network][key] += meta_step_size * meta_policy_update[network][key]

            print(f'\t`-> Updated meta policy:')

            if it % save_every == 0:
                chkpt_dir = os.path.join(save_dir, f'chkpt_{it}')
                os.makedirs(chkpt_dir, exist_ok=False)
                save_checkpoint(meta_policy_params, chkpt_dir)
                print(f'\t\t`-> Saved checkpoint in: {chkpt_dir}')
                print(f'\t\t`-> Validating meta policy:')
                eval_env = sample_zcqpee_tasks(n_tasks=1,
                                               eps=eval_eps,
                                               **env_kwargs)[0]
                eval_before, eval_after = evaluate_meta_policy(meta_policy_params=meta_policy_params,
                                                               algo=TRPO,
                                                               algo_kw=trpo_kw,
                                                               eval_env=eval_env,
                                                               verbose=True)
                log_metrics(eval_before, step=it, writer=writer, prefix='eval/before-tuning')
                log_metrics(eval_after, step=it, writer=writer, prefix='eval/after-tuning')

    except KeyboardInterrupt:
        pass

    return meta_policy_params


