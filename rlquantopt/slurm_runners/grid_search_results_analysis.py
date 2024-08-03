import os
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt

from rlquantopt.rl_agents.eval_zcqpee_sb3 import main, parse_args as eval_parse_args
from rlquantopt.rl_envs.zc_qpee import ZCQPEE


def get_env_yml(model_dir):
    for item in os.listdir(model_dir):
        if item.endswith(".yml") or item.endswith(".yaml"):
            return os.path.join(model_dir, item)


def parse_info_txt(info_txt_path):
    with open(info_txt_path, "r") as f:
        content = f.read()

    patterns = {
    'batch_size': r'"batch_size"\s*:\s*(\d+)',
    'n_steps': r'"n_steps"\s*:\s*(\d+)',
    'seed': r'"seed"\s*:\s*(\d+)',
    }
    extracted_data = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            extracted_data[key] = match.group(1)
    return extracted_data


def main():
    # par_dir = os.path.abspath('.')
    par_dir = os.path.abspath('/home/leander/code/rlquantopt/rlquantopt/rl_agents/')
    print(par_dir)
    model_dirs = []
    for env_type_dir in os.listdir(par_dir):
        if os.path.isdir(env_type_dir) and os.path.basename(env_type_dir).startswith('ZCQPEE_pl-'):
            env_type_path = os.path.join(par_dir, env_type_dir)
            for model_dir in os.listdir(env_type_path):
                model_dirs.append(os.path.join(env_type_path, model_dir))

    hparams = pd.DataFrame(columns=['model_dir', 'algo', 'pulse_length', 'T', 'fid_thresh', 'n_steps', 'batch_size', 'seed', 'a_scale', 'a_norm_max'])
    for model_dir in model_dirs:
        env = ZCQPEE.from_yaml(get_env_yml(model_dir))
        run_data = parse_info_txt(os.path.join(model_dir, 'info.txt'))
        run_data['model_dir'] = model_dir
        run_data['pulse_length'] = env.pulse_length
        run_data['T'] = env.T
        run_data['fid_thresh'] = env.FID_THRESH
        run_data['a_scale'] = env.action_scaling['z']
        run_data['a_norm_max'] = env.A_norm_max
        run_data_df = pd.DataFrame([run_data])#, orient='index')
        hparams = pd.concat([hparams, run_data_df], ignore_index=False)

    hparams.to_csv(os.path.join(par_dir, 'analysis_03082024_191500.csv'))

    # Now estimate best performance of each model


if __name__ == '__main__':
    main()
