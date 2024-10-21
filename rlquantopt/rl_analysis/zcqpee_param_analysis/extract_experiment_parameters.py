import os
import yaml
import pandas as pd

from rlquantopt.rl_envs.zc_qpee import ZCQPEE


def main():
    lab_dir = '../../rl_agents/'
    lab_name = 'ZCQPEE_pl-500_T-300ns_delta_mode-TRPO'
    lab_path = os.path.join(lab_dir, lab_name)

    save_dir = '/home/leander/code/rlquantopt/rlquantopt/rl_analysis/zcqpee_param_analysis'

    # List to store each env_kwargs dictionary
    env_kwargs_list = []

    # Iterate over each experiment directory
    for exp_dir in os.listdir(lab_path):
        exp_dir_path = os.path.join(lab_path, exp_dir)
        yml_path = ZCQPEE.find_yaml_in_dir(exp_dir_path)

        # Load YAML data from file
        with open(yml_path, 'r') as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)

        env_kw = data.get('env_kwargs', {})
        env_kw['exp_dirname'] = exp_dir  # Add exp_dirname as key
        env_kwargs_list.append(env_kw)

    # Create a DataFrame from the list of env_kwargs
    df = pd.DataFrame(env_kwargs_list)

    # Reorder columns to put 'exp_dirname' first
    columns = ['exp_dirname'] + [col for col in df.columns if col != 'exp_dirname']
    df = df[columns]

    # Sort the DataFrame by 'exp_dirname' assuming it is a date
    df['exp_datetime'] = pd.to_datetime(df['exp_dirname'], format='%d-%m-%y_%H%M%S', errors='coerce')
    df = df.sort_values(by='exp_datetime').reset_index(drop=True)

    # Fill in missing values with None
    df = df.fillna(value=pd.NA)

    # Save the DataFrame to a tab-delimited text file with all columns aligned
    save_name = os.path.join(save_dir, f'{lab_name}_parameters')
    with open(f'{save_name}.txt', 'w') as f:
        f.write(df.to_string(index=False, na_rep='None', col_space=15))

    df.to_csv(f'{save_name}.csv')

    # Display the resulting DataFrame as a table
    print(df)


if __name__ == "__main__":
    main()