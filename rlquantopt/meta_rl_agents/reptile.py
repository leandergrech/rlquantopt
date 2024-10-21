import argparse
import copy
import numpy as np

from rlquantopt.meta_rl_agents.sampler import sample_zcqpee_tasks
from rlquantopt.meta_rl_agents.train_utils import train_agent, validate_meta_policy, save_checkpoint


def reptile_meta_learning_zcqpee(initial_policy_params,
                                 trpo_kw,
                                 save_dir,
                                 save_every=10,
                                 n_tasks=10,
                                 val_n_envs=5,
                                 val_eps=5e-2,  # Let's make validation easier for now
                                 meta_iterations=1000,
                                 inner_steps=10,
                                 meta_step_size=0.1,
                                 eps=1e-1,
                                 **env_kwargs):
    meta_policy_params = copy.deepcopy(initial_policy_params)

    val_envs = sample_zcqpee_tasks(n_tasks=val_n_envs, eps=val_eps, **env_kwargs)

    try:
        for it in range(meta_iterations):
            task_batch = sample_zcqpee_tasks(n_tasks, eps, **env_kwargs)

            meta_policy_update = {
                'pi': np.zeros_like(meta_policy_params['pi']),
                'vf': np.zeros_like(meta_policy_params['vf'])
            }

            for env in task_batch:
                # Copy initial policy parameters and train agent for `inner_steps` steps
                task_policy_params = copy.deepcopy(meta_policy_params)
                task_policy_params = train_agent(task_policy_params, env, trpo_kw, inner_steps)

                # Calculate parameter difference from initial and add to task batch policy parameters update
                # Calculate the parameter differences for the meta-update
                meta_policy_update['pi'] += task_policy_params['pi'] - meta_policy_params['pi']
                meta_policy_update['vf'] += task_policy_params['vf'] - meta_policy_params['vf']

            # Regularise and update meta policy parameters
            meta_policy_update['pi'] /= n_tasks
            meta_policy_update['vf'] /= n_tasks

            # Update the meta-policy parameters
            meta_policy_params['pi'] += meta_step_size * meta_policy_update['pi']
            meta_policy_params['vf'] += meta_step_size * meta_policy_update['vf']

            print(f"Iteration {it + 1}/{meta_iterations} - Meta-policy updated")

            if it % save_every == 0:
                chkpt_dir = os.path.join(save_dir, f'chkpt_{it}')
                os.makedirs(chkpt_dir, exist_ok=False)
                save_checkpoint(meta_policy_params, chkpt_dir)
                validate_meta_policy(meta_policy_params, val_envs, trpo_kw, writer)

    except KeyboardInterrupt:
        pass

    return meta_policy_params


