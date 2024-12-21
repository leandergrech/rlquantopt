import os
from collections import defaultdict
import torch
from sb3_contrib import TRPO

from rlquantopt.meta_rl_agents.sampler import sample_zcqpee_tasks
from rlquantopt.meta_rl_agents.vec_train_utils import vec_train_agents
from rlquantopt.meta_rl_agents.train_utils import copy_params, save_checkpoint
from rlquantopt.meta_rl_agents.eval_reptile_agent import evaluate_meta_policy, log_metrics, evaluate_params_zcqpee


def vec_reptile_meta_learning_zcqpee(   initial_policy_params,
                                        trpo_kw,
                                        save_dir,
                                        writer,
                                        trpo_n_envs=1,
                                        save_every=1,
                                        eval_every=20,
                                        n_tasks=9,
                                        cur_chkpt=0,
                                        eval_eps=5e-2,
                                        n_parallel_tasks=3,
                                        meta_iterations=1000,
                                        inner_steps=10,
                                        meta_step_size=0.1,
                                        eps=1e-1,
                                        device='cuda',
                                        **env_kwargs):

    assert n_tasks >= n_parallel_tasks and n_tasks % n_parallel_tasks == 0, "Choose `n_tasks` to be a multiple of `n_parallel_tasks` to avoid useless inefficiency"

    meta_policy_params = copy_params(initial_policy_params)

    networks = ('pi', 'vf')

    try:
        for it in range(meta_iterations):
            if it <= cur_chkpt:
                continue

            print(f'-> Iteration {it}')
            print(f'\t`-> Sampling {n_tasks} tasks')
            task_batch = sample_zcqpee_tasks(n_tasks, eps, **env_kwargs)

            meta_policy_update = dict()
            for network in networks:
                meta_policy_update[network] = {}
                for key in meta_policy_params[network]:
                    meta_policy_update[network][key] = torch.zeros_like(meta_policy_params[network][key], dtype=torch.float).to(device)

            print(f'\t`-> Training inner loops:')
            all_task_updated_params = {}
            for batch_idx in range(n_tasks // n_parallel_tasks):
                print(f'\t\t`-> Batch #{batch_idx + 1}')
                tasks = task_batch[batch_idx * n_parallel_tasks:(batch_idx + 1) * n_parallel_tasks]

                task_policy_params = copy_params(meta_policy_params)

                vec_params = vec_train_agents(policy_params=task_policy_params,
                                              envs=tasks,
                                              algo=TRPO,
                                              algo_kw=trpo_kw,
                                              steps=inner_steps,
                                              n_envs=trpo_n_envs,
                                              inner_loop_tb_dir=save_dir)
                all_task_updated_params.update(vec_params.copy())
                for params in vec_params.values():
                    for network in networks:
                        for key in meta_policy_params[network]:
                            meta_policy_update[network][key] += params[network][key] - meta_policy_params[network][key]

            # # Regularise and update meta policy parameters
            n_keys = 0
            for network in networks:
                for key in meta_policy_params[network]:
                    meta_policy_update[network][key] /= n_tasks
                    param_update = meta_step_size * meta_policy_update[network][key]
                    meta_policy_params[network][key] += param_update
                    n_keys += 1

            print(f'\t`-> Updated meta policy:')

            if it % save_every == 0 or it == 0:
                chkpt_dir = os.path.join(save_dir, f'chkpt_{it}')
                os.makedirs(chkpt_dir, exist_ok=False)
                save_checkpoint(meta_policy_params, chkpt_dir)
                print(f'\t\t`-> Saved checkpoint in: {chkpt_dir}')

            if it % eval_every == 0:
                print(f'\t\t`-> Evaluating latest task policies starting from meta policy:')
                eval_envs = sample_zcqpee_tasks(n_tasks=1,
                                               eps=eval_eps,
                                               **env_kwargs)[0]
                n_evals = len(all_task_updated_params)
                all_eval_data = defaultdict(float)
                for params in all_task_updated_params.values():
                    eval_data = evaluate_params_zcqpee(meta_policy_params=params,
                                                       algo=TRPO,
                                                       algo_kw=trpo_kw,
                                                       eval_envs=eval_envs,
                                                       deterministic=True,)
                    for k, v in eval_data.items():
                        all_eval_data[k] += v
                for k, v in all_eval_data.items():
                    all_eval_data[k] = v / n_evals
                log_metrics(all_eval_data, step=it, writer=writer, prefix='eval-meta')
                # eval_before, eval_after = evaluate_meta_policy(meta_policy_params=meta_policy_params,
                #                                                algo=TRPO,
                #                                                algo_kw=trpo_kw,
                #                                                eval_env=eval_env,
                #                                                fine_tuning_steps=20,
                #                                                fine_tuning_n_envs=1,
                #                                                verbose=True)
                # log_metrics(eval_before, step=it, writer=writer, prefix='eval/before-tuning')
                # log_metrics(eval_after, step=it, writer=writer, prefix='eval/after-tuning')

    except KeyboardInterrupt:
        pass

    return meta_policy_params


