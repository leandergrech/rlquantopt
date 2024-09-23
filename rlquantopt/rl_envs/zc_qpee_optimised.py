import numpy as np
import pandas as pd
from rlquantopt.rl_envs.zc_qpee import ZCQPEE


class ZCQPEEOptimised(ZCQPEE):
    def __init__(self, optimised_pulse_path, model_params, **env_kwargs):
        super().__init__(model_params, **env_kwargs)

        self.optimised_pulse_path = optimised_pulse_path
        self.ideal_pulses = None

        coeff, tlist = self.get_optimal_pulse()
        self.ideal_pulses = {'labels': [self.channel_label],
                             'coeff': [coeff],
                             'tlist': [tlist],
                             'max_len': len(coeff),
                             'max_time': tlist[-1]}

    def get_optimal_pulse(self):
        data = pd.read_csv(self.optimised_pulse_path)
        n_keys = len(data.keys())
        if n_keys == 2:
            tkey, vkey = data.keys()
        elif n_keys == 3:
            tkey, vkey = data.keys()[1:]
        else:
            raise Exception("There's something fucked with the CSV pulse file.")

        tlist = data[tkey].to_numpy()
        coeff = data[vkey].to_numpy()

        return coeff, tlist

    def get_reconstructed_pulses_with_uniform_time(self):
        ideal_pulses = self.ideal_pulses
        ideal_amps = ideal_pulses['coeff']
        tlists = ideal_pulses['tlist']
        T = ideal_pulses['max_time']
        N = self.pulse_length
        global_tlist = np.linspace(0, T, N)

        if N == ideal_pulses['max_len']:
            return np.array(ideal_amps), global_tlist

            # Obtain ideal amplitudes reconstructions
        recons_amps = np.zeros(shape=(len(ideal_amps), N))
        for i, t in enumerate(global_tlist[:-1]):
            for j, tli in enumerate(tlists):
                idx = np.argmin(np.square(tli - t))
                if tli[idx] < t:
                    idx1 = idx - 1
                    idx2 = idx
                else:
                    idx1 = idx
                    idx2 = idx + 1

                recons_amp = ideal_amps[j][idx1] + ((t - tli[idx1]) / (tli[idx2] - tli[idx1])) * (
                        ideal_amps[j][idx2] - ideal_amps[j][idx1])
                recons_amps[j][i] = recons_amp

        return np.array(recons_amps), global_tlist
