import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm

from rlquantopt.utils.analysis_utils import get_pulse_data, get_pulse_spectrum, get_metrics_data


# Directory and pattern for CSV files
par_dir = "/home/leander/code/rlquantopt/rlquantopt/rl_agents"
exp_type = "ZCQPEE_pl-1000_T-50ns_delta_mode-TRPO"
exp_date = "05-12-24_201634"

# /home/leander/code/rlquantopt/rlquantopt/rl_agents/ZCQPEE_pl-1000_T-50ns_delta_mode-TRPO/05-12-24_201634

# exp_name = os.path.join(exp_date, "results/clip_to_fid_0.9900")
# exp_name = exp_date
save_suffix = '_no_term'

# exp_path = os.path.join(par_dir, exp_type, exp_date, 'results/no_term')
exp_path = f'{par_dir}/{exp_type}/{exp_date}/results/no_term'

vlines = {'Best agent': {'v': 12566528, 'c': 'g', 'ls': '--'},
          # 'Breakthrough': {'v': 200000, 'c': 'r', 'ls': '--'},
          }
# min_rl_step = int(7e6)
min_rl_step = 0
# max_rl_step = int(4e5)
max_rl_step = np.inf
cmap = 'Spectral'

# file_pattern = f"rl_model_*_steps_ZCQPEE_pl-500_T-300ns_delta_mode_ep0{save_suffix}.csv"

# Store step numbers and spectra
step_numbers = []
spectra = []
concurrences = []
unitarities = []
rewards = []


# Collecting all relevant CSV files and their spectra
for root, _, files in os.walk(exp_path):
    for file in tqdm(files):
        if file.endswith('metrics.csv'):
            continue
        elif file.startswith("rl_model_") and file.endswith(f"{save_suffix}.csv"):
            step_number = int(file.split("_")[2])  # Extracting step number from filename
            if step_number < min_rl_step or step_number > max_rl_step:
                continue

            pulse_file = os.path.join(root, file)
            metrics_file = os.path.join(root, f'{os.path.splitext(file)[0]}_metrics.csv')
            if not os.path.exists(metrics_file):
                continue

            tlist_fine, amps = get_pulse_data(pulse_file)
            tlist_coarse, concurs, unitars, rews = get_metrics_data(metrics_file)

            flist, fabsamps = get_pulse_spectrum(tlist_fine, amps)
            step_numbers.append(step_number)
            spectra.append(fabsamps)
            concurrences.append(concurs)
            unitarities.append(unitars)
            rewards.append(rews)

# Sort by step number
sorted_indices = np.argsort(step_numbers)
step_numbers = np.array(step_numbers)[sorted_indices]
spectra = np.array(spectra)[sorted_indices]
concurrences = np.array(concurrences)[sorted_indices]
concurrences = 1 - concurrences
unitarities = np.array(unitarities)[sorted_indices]
unitarities = 1 - unitarities
rewards = np.array(rewards)[sorted_indices]

# Plotting the spectral evolution
log_x = False
fig, ax = plt.subplots(figsize=(15, 10))
im = ax.imshow(
    spectra.T, aspect='auto', extent=[min(step_numbers), max(step_numbers),flist[0], flist[-1]],
    origin='lower', norm=LogNorm(), cmap=cmap
)
fig.suptitle("Spectral evolution during RL training")
ax.set_title(f'Experiment: {exp_type} - {exp_date}')
for k, v in vlines.items():
    if k == 'Best agent' and v['v'] > max_rl_step:
        continue
    ax.axvline(v['v'], color=v['c'], lw=1.4, linestyle=v['ls'], label=k)
ax.legend(loc='best')
fig.colorbar(im, label='Mag. [a.u]')
if log_x: ax.set_xscale('log')
ax.set_xlabel("RL training step")
ax.set_ylabel("Frequency [GHz]")
fig.tight_layout()
# fig.savefig(os.path.join(f"{exp_type}_{exp_date}_pulse_spectra_evolution_between_{min_rl_step:.1e}-{max_rl_step:.1e}_steps{'_logx' if log_x else ''}.pdf"))
# fig.savefig(os.path.join(f"{exp_type}_{exp_date}_pulse_spectra_evolution_steps_{min_rl_step}-{max_rl_step}{'_logx' if log_x else ''}.pdf"))
fig.savefig(os.path.join(f"{exp_type}_{exp_date}_pulse_spectra_evolution_lin.pdf"))
# plt.show()

# Plotting the concurrence evolution
log_x = False
fig, ax = plt.subplots(figsize=(15, 10))
im = ax.imshow(
    concurrences.T, aspect='auto', extent=[min(step_numbers), max(step_numbers), 0, tlist_coarse[-1]],
    origin='lower', norm=LogNorm(), cmap=cmap
)
fig.suptitle("Concurrence evolution during RL training")
ax.set_title(f'Experiment: {exp_type} - {exp_date}')
for k, v in vlines.items():
    if k == 'Best agent' and v['v'] > max_rl_step:
        continue
    ax.axvline(v['v'], color=v['c'], lw=1.4, linestyle=v['ls'], label=k)
ax.legend(loc='best')
fig.colorbar(im, label='Mag. [a.u]')
if log_x: ax.set_xscale('log')
ax.set_xlabel("RL training step")
ax.set_ylabel("Pulse time [ns]")
fig.tight_layout()
# fig.savefig(os.path.join(f"{exp_type}_{exp_date}_pulse_concurrence_evolution_between_{min_rl_step:.1e}-{max_rl_step:.1e}_steps{'_logx' if log_x else ''}.pdf"))
# fig.savefig(os.path.join(f"{exp_type}_{exp_date}_pulse_concurrence_evolution_steps_{min_rl_step}-{max_rl_step}{'_logx' if log_x else ''}.pdf"))
fig.savefig(os.path.join(f"{exp_type}_{exp_date}_pulse_concurrence_evolution_lin_inv.pdf"))
# plt.show()

# Plotting the unitarities evolution
log_x = False
fig, ax = plt.subplots(figsize=(15, 10))
im = ax.imshow(
    unitarities.T, aspect='auto', extent=[min(step_numbers), max(step_numbers), 0, tlist_coarse[-1]],
    origin='lower', norm=LogNorm(), cmap=cmap
)
fig.suptitle("Unitarity evolution during RL training")
ax.set_title(f'Experiment: {exp_type} - {exp_date}')
for k, v in vlines.items():
    if k == 'Best agent' and v['v'] > max_rl_step:
        continue
    ax.axvline(v['v'], color=v['c'], lw=1.4, linestyle=v['ls'], label=k)
ax.legend(loc='best')
fig.colorbar(im, label='Mag. [a.u]')
if log_x: ax.set_xscale('log')
ax.set_xlabel("RL training step")
ax.set_ylabel("Pulse time [ns]")
fig.tight_layout()
# fig.savefig(os.path.join(f"{exp_type}_{exp_date}_pulse_concurrence_evolution_between_{min_rl_step:.1e}-{max_rl_step:.1e}_steps{'_logx' if log_x else ''}.pdf"))
# fig.savefig(os.path.join(f"{exp_type}_{exp_date}_pulse_unitarity_evolution_steps_{min_rl_step}-{max_rl_step}{'_logx' if log_x else ''}.pdf"))
fig.savefig(os.path.join(f"{exp_type}_{exp_date}_pulse_unitarity_evolution_steps_lin_inv.pdf"))

# Plotting the rewards evolution
log_x = False
fig, ax = plt.subplots(figsize=(15, 10))
im = ax.imshow(
    rewards.T, aspect='auto', extent=[min(step_numbers), max(step_numbers), 0, tlist_coarse[-1]],
    origin='lower', cmap=cmap
)
fig.suptitle("Reward evolution during RL training")
ax.set_title(f'Experiment: {exp_type} - {exp_date}')
for k, v in vlines.items():
    if k == 'Best agent' and v['v'] > max_rl_step:
        continue
    ax.axvline(v['v'], color=v['c'], lw=1.4, linestyle=v['ls'], label=k)
ax.legend(loc='best')
fig.colorbar(im, label='Mag. [a.u]')
if log_x: ax.set_xscale('log')
ax.set_xlabel("RL training step")
ax.set_ylabel("Pulse time [ns]")
fig.tight_layout()
# fig.savefig(os.path.join(f"{exp_type}_{exp_date}_pulse_concurrence_evolution_between_{min_rl_step:.1e}-{max_rl_step:.1e}_steps{'_logx' if log_x else ''}.pdf"))
# fig.savefig(os.path.join(f"{exp_type}_{exp_date}_pulse_unitarity_evolution_steps_{min_rl_step}-{max_rl_step}{'_logx' if log_x else ''}.pdf"))
fig.savefig(os.path.join(f"{exp_type}_{exp_date}_pulse_reward_evolution_steps_lin.pdf"))
plt.show()


