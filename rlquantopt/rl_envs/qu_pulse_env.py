from itertools import product
from typing import *
import numpy as np
import gym
from qutip import basis
from qutip_qip.device import SCQubits
import scipy.interpolate as interp
import matplotlib.pyplot as plt
from qutip_qip.circuit import QubitCircuit

ObsType = TypeVar("QuantumPulse", gym.spaces.Box, list, np.ndarray)
ActType = TypeVar("QuantumPulseCorrection", list, np.ndarray)


def get_discrete_actions(n_act, act_dim=3):
    all_actions = [list(item) for item in product(*np.repeat([[item for item in range(act_dim)]], n_act, axis=0))]
    if n_act == 1:
        all_actions = [item[0] for item in all_actions]
    return all_actions


def get_reduced_discrete_actions(n_act, act_dim=3):
    all_actions = np.vstack([np.zeros(n_act),
                      np.diag(-np.ones(n_act)),
                      np.diag(np.ones(n_act))]).astype(int) + 1
    return all_actions.tolist()


class QuPulseEnv(gym.Env):
    ABS_REW_IMPROVEMENT = 1e-4  # Added to fidelity then scaled by self.rew_scale
    def __init__(self, qc=None, processor_params=None):
        if qc is None:
            qc = QubitCircuit(N=1)
            qc.add_gate('X', targets=0)
        # self.action_eps = 0.001
        self.action_eps = 0.01
        self.rew_scale = 10.
        self.max_steps = 20

        self.max_obs_scale = 10.

        self.it = 0

        self.thresh_r = None # Initialised on reset
        self.current_state = None

        if processor_params is None: processor_params = dict(t1=50e3, t2=20e3)

        self.processor = SCQubits(num_qubits=qc.N, **processor_params)
        self.orig_tlist, self.orig_coeff = self.processor.load_circuit(qc)

        self.n_channels = len(self.orig_coeff)
        self.action_space = gym.spaces.MultiDiscrete(np.repeat(3, self.n_channels * 2))
        # # Only amplitudes or duration
        # self.observation_space = gym.spaces.Box(np.zeros(self.n_channels, dtype=float),
        #                                         np.repeat(np.inf, self.n_channels))
        # Amplitudes and duration
        self.observation_space = gym.spaces.Box(np.zeros(self.n_channels * 2, dtype=float),
                                                np.ones(self.n_channels * 2, dtype=float) * 2)

        self.basis0 = basis(3)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> ObsType:
        init_state = np.ones(self.observation_space.shape[0])
        if options is not None:
            init_state = options.get('init_state', init_state)

        self.thresh_r = (np.diagonal(np.abs(np.array(self.processor.run_state(self.basis0).states[-1])))[1] + \
                         self.ABS_REW_IMPROVEMENT)

        self.current_state = init_state
        self.it = 0

        return self.current_state

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        # print('action=', action)
        centered_action = np.array(action) - 1
        delta_action = centered_action * self.action_eps

        self.current_state += delta_action
        self.current_state = np.clip(self.current_state, a_min=-self.max_obs_scale, a_max=self.max_obs_scale)

        # Amplitudes and durations
        for i in range(self.n_channels):
            self.scale_pulse_amp(self.current_state[i], i)  # 1st half for amplitude
            self.scale_pulse_duration(self.current_state[i + self.n_channels], i)   # 2nd half for duration

        final_state = np.array(self.processor.run_state(self.basis0).states[-1])
        final_state = np.abs(final_state)
        prob_ket_one = np.diagonal(final_state)[1]

        r = prob_ket_one

        success = r > self.thresh_r

        self.it += 1
        d = success or self.it >= self.max_steps

        return self.current_state, r * self.rew_scale, d, {'success': success}

    def scale_pulse_amp(self, scale, chidx):
        ogp = [item for item in self.orig_coeff.values()][chidx]
        self.processor.pulses[chidx].coeff = ogp * scale


    def scale_pulse_duration(self, scale, chidx):
        ogp = [item for item in self.orig_coeff.values()][chidx]
        sz = ogp.size
        nsz = int(scale * sz)
        coeff_interp = interp.interp1d(np.arange(sz), ogp)
        self.processor.pulses[chidx].coeff = coeff_interp(np.linspace(0, sz - 1, nsz))
        self.processor.pulses[chidx].tlist = np.arange(nsz).astype(float)

    def render(self, mode='human'):
        if 'plt' not in globals(): import matplotlib.pyplot as plt
        fig, axs = plt.subplots(self.n_channels)
        for ax, pulse in zip(axs, self.processor.pulses):
            ax.plot(pulse.coeff)
            ax.set_xlabel('Time')
            ax.set_ylabel('Amplitude')
            ax.set_title(pulse.label)
            fig.tight_layout()
        plt.show(block=False)

    def __repr__(self):
        return f'QuPulseEnv_{self.observation_space.shape[0]}obsx{self.action_space.shape[0]}act'

