import os
import numpy as np
import pandas as pd

from rlquantopt.utils.rl_utils import get_pulse_data


# Function to compute spectrum from CSV
def get_pulse_spectrum(tlist, amps):
    # Perform FFT
    famps = np.fft.fft(amps)

    # Fix freq axis & calculate magnitude
    famps = np.fft.fftshift(famps)
    fabsamps = np.abs(famps)

    # Get freq axis
    flist = np.fft.fftfreq(len(famps), d=tlist[-1] / len(famps))
    flist = np.fft.fftshift(flist)

    return flist, fabsamps


def get_model_zip_files(exp_path):
    model_zips = {}
    # Collecting all relevant CSV files and their spectra
    for root, _, files in os.walk(exp_path):
        for file in files:
            if file.startswith("rl_model_") and file.endswith(".zip"):
                step_number = int(file.split("_")[2])  # Extracting step number from filename
                model_zips[step_number] = os.path.join(root, file)

    return model_zips

def get_metrics_data(metrics_path, verbose=False):
    data = pd.read_csv(metrics_path)
    c = data['concurrences'].to_numpy()
    u = data['unitarities'].to_numpy()
    r = data['rewards']
    t = data['tlist'].to_numpy()

    return t, c, u, r

if __name__ == '__main__':
    exp_path = '/home/leander/code/rlquantopt/rlquantopt/rl_agents/ZCQPEE_pl-1000_T-50ns_delta_mode-TRPO/05-12-24_201634'
    model_zips = get_model_zip_files(exp_path)
    model_training_steps = list(model_zips.keys())

    nb_val_steps = 100
    max_step = max(model_training_steps)
    val_steps = np.linspace(0, max_step, nb_val_steps).astype(int)

    ret_model_zips = {}
    for val_step in val_steps:
        idx = np.argmin(np.abs(np.subtract(model_training_steps, val_step)))
        train_step = model_training_steps[idx]
        ret_model_zips[train_step] = model_zips[train_step]

    print(' '.join(list(ret_model_zips.values())))


