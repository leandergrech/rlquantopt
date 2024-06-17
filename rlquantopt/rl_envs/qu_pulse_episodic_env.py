import os.path
from itertools import product
from datetime import datetime as dt

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm
from matplotlib import cm
from gymnasium.core import RenderFrame, ActType, ObsType
from tqdm import trange, tqdm

from qutip.core import ket2dm, basis, ptrace, ket
from qutip_qip.circuit import QubitCircuit
from qutip_qip.device import SCQubits
# from qutip import basis, ptrace, ket, Bloch
# from krotov.functionals import F_avg
# from qutip.metrics import fidelity

from rlquantopt.utils.utils import generate_random_alphanumeric
from rlquantopt.rl_envs.zcqubits import ZCQubits

from stable_baselines3.ppo.policies import MlpPolicy


class QuPulseEpisodicEnv(gym.Env):
    # Action scaling symmetric around 0
    action_channel_scaling = {
        'zx': 50e-5,
        'sx': 5e-3,
        'sy': 5e-3,
        'sz': 1e-4
    }

    FID_THRESH = 0.995
    PREC = 5e-4
    A_norm_max = 10
    PLOT_LOG_EPS = 1e-3
    cmap = 'CMRmap'

    def __init__(self, pulse_length=300, sparse_reward=False, use_full_liouville=False, inc_off_diag=False):
        super(QuPulseEpisodicEnv, self).__init__()
        self.REW_THRESH = self.fid2rew(self.FID_THRESH)

        self.dt = 1
        self.sparse = sparse_reward
        self.use_full_liouville = use_full_liouville
        self.inc_off_diag = inc_off_diag
        self.cur_idx = 0
        self.rewards = None

        # Initialise simulator
        self.n_levels = n_levels = 2
        self.qc = qc = QubitCircuit(N=2)
        qc.add_gate("CNOT", controls=0, targets=1)
        # qc.add_gate("SWAP", targets=[0, 1])

        self.simulator = SCQubits(num_qubits=2, dims=[n_levels, n_levels], wq=[5.15, 5.09], wr=[5.96, 5.96], g=[0.1, 0.1], alpha=[-0.3, -0.3], omega_single=[0.01, 0.01], omega_cr=[0.01, 0.01], t1=50.e3, t2=20.e3)
        # print(processor.get_control('sx0'))
        self.simulator.pulse_mode = 'discrete'
        self.simulator.load_circuit(qc)
        pulses = self.simulator.pulses
        if pulse_length:
            N = pulse_length
        else:
            N = max([len(item.coeff) for item in pulses])
        T = max([item.tlist[-1] for item in pulses])
        self.ideal_pulses = {'labels': [item.label for item in pulses],
                             'coeff': [item.coeff.copy() for item in pulses],
                             'tlist': [item.tlist.copy() for item in pulses],
                             'max_len': N,
                             'max_time': T}
        # Set simulation delta time and maximum pulse length based on ideal pulse for the gate being implemented
        self.dt = T / (N - 1)
        self.pulse_length = N

        # `channel_labels` will define the order of the dimensions in the action space
        self.channel_labels = [item.label for item in self.simulator.pulses]
        self.n_channels = n_channels = len(self.channel_labels)

        # Action space holds amplitude deltas for all channels
        self.action_space = spaces.Box(low=-1, high=1, shape=(n_channels,), dtype=np.float32)
        self.n_act = n_channels

        # Action space holds channel amplitudes throughout episode
        self.pulse_amplitudes_norm = self.init_pulse_amplitudes()
        self.amps_cur = np.zeros(self.n_act)

        # Observation space holds 4 basis vectors of a 2 qudit system + previous pulse amplitudes
        self.nb_basis_states = self.n_levels ** 2
        # self.states_in_episode = None
        self.current_state = None
        self.final_states_all = None

        # Prepare evaluation initial states
        self.psi00 = basis([self.n_levels, self.n_levels], [0, 0])
        self.psi01 = basis([self.n_levels, self.n_levels], [0, 1])
        self.psi10 = basis([self.n_levels, self.n_levels], [1, 0])
        self.psi11 = basis([self.n_levels, self.n_levels], [1, 1])
        self.basis_states_ket = [self.psi00, self.psi01, self.psi10, self.psi11]
        self.full_liouville_basis = [psi * phi.dag() for psi, phi in product(self.basis_states_ket, self.basis_states_ket)]
        self.basis_states = [ket2dm(b) for b in self.basis_states_ket]

        self.unitary = u = qc.compute_unitary()


        self.mapped_basis_states = [sum(complex(self.unitary[i, j]) * self.basis_states_ket[i]
                                        for i in range(self.unitary.shape[0])) for j in range(self.unitary.shape[1])]

        # Lots of gates just rearrange the basis states, and we can avoid some complexity by identifying
        # this and setting the mapped_basis_states to the identical objects as the original basis_states
        for i, state in enumerate(self.mapped_basis_states):
            for j, basis_state in enumerate(self.basis_states_ket):
                if state == basis_state:
                    self.mapped_basis_states[i] = basis_state

        self.full_liouville_targets = [u * b for b in self.full_liouville_basis]
        if use_full_liouville:
            self.initial_states = self.full_liouville_basis.copy()
            self.target_states = self.full_liouville_targets
        else:
            self.initial_states = self.basis_states.copy()

        self.target_states = [u * i * u.dag() for i in self.initial_states]

        if self.inc_off_diag:
            init_state = self.get_tril_elements(np.zeros(shape=(self.nb_basis_states, self.nb_basis_states, self.nb_basis_states), dtype=np.complex64))
            N = init_state.shape[0]
        else:
            N = self.nb_basis_states ** 2

        self.n_obs = N + n_channels
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.n_obs,), dtype=np.float32)

    def __call__(self):
        return self

    def __str__(self):
        return (f'Quantum circuit: {self.qc}\n'
                f'Nb. observations: {self.n_obs}\n'
                f'Nb. Actions: {self.n_act}\n'
                f'Action channels: {self.channel_labels}\n'
                f'Pulse length: {self.pulse_length}\n'
                f'Time delta: {self.dt}')

    @staticmethod
    def _rho1(basis_states):
        results = sum(((4 - i) / 10) * psi * psi.dag() for i, psi in enumerate(basis_states))
        return results

    @staticmethod
    def _rho2(basis_states):
        return 0.25 * sum(psi_i * psi_j.dag() for psi_i, psi_j in product(basis_states, repeat=2))

    @staticmethod
    def _rho3(basis_states):
        return 0.25 * sum(psi * psi.dag() for psi in basis_states)

    def get_reconstructed_pulses_with_uniform_time(self):
        ideal_pulses = self.ideal_pulses
        ideal_amps = ideal_pulses['coeff']
        tlists = ideal_pulses['tlist']
        T = ideal_pulses['max_time']
        N = self.pulse_length
        global_tlist = np.linspace(0, T, N + 1)

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

    def init_pulse_amplitudes(self):
        # return np.zeros((self.n_channels, self.pulse_length))  # First row for u01, second row for d1
        buffer = {}
        for channel_label in self.channel_labels:
            buffer[channel_label] = np.zeros(self.pulse_length + 1)  # +1 since first amp must be zero

        return buffer

    def denorm_action(self, action):
        for a_idx, channel_label_action in enumerate(self.channel_labels):    # For all channel labels
            for channel_label_norm, scale in self.action_channel_scaling.items():
                if channel_label_norm in channel_label_action:  # Find the correct scaling
                    action[a_idx] *= scale
        return action

    def norm_action(self, action):
        for a_idx, channel_label in enumerate(self.channel_labels):
            for channel_type, scale in self.action_channel_scaling.items():
                if channel_type in channel_label:
                    action[a_idx] /= scale
                    break
        return np.array(action)

    def reset(self, seed=None):
        # super().reset(self, seed)
        # print('Resetting here')
        if seed is not None:
            np.random.seed(seed)

        self.cur_idx = 0
        self.final_states_all = [self.initial_states.copy()]
        self.pulse_amplitudes_norm = self.init_pulse_amplitudes()
        self.amps_cur = np.zeros(self.n_act)
        self.step_states = self.initial_states.copy()
        self.current_state = self.extract_current_state(self.step_states, 0).astype(np.float32)
        # self.current_state = np.zeros(self.n_obs, dtype=np.float32)
        self.rewards = []
        return self.current_state, {}

    def step(self, action):
        """
        Add a pulse amplitude delta vector (action) on the previous value of the pulse amplitude.
        :param action: Must be list-like with `self.n_abs` dimensions
        :return: observation_tp1, reward, terminated, truncated, info
        """
        # Absolute amplitude conversion required for rendering
        prev_abs_action = np.zeros(self.n_channels)
        oob_pulse = False   # Out of bounds absolute pulse
        if self.cur_idx > 0:
            prev_abs_action = [self.pulse_amplitudes_norm[lbl][self.cur_idx - 1] for lbl in self.channel_labels]
        for i, channel_label in enumerate(self.channel_labels):
            abs_action = prev_abs_action[i] + action[i]
            if abs(abs_action) > self.A_norm_max:
                oob_pulse = True
                abs_action = np.sign(abs_action) * self.A_norm_max
            self.pulse_amplitudes_norm[channel_label][self.cur_idx] = abs_action

        # Environment dynamics
        action_denorm = self.denorm_action(action.copy())
        self.step_states = self.forward_dynamics(from_states=self.step_states, amp_deltas=action_denorm)
        self.final_states_all.append(self.step_states.copy())

        # terminated is only True when reward threshold is exceeded
        # truncated is only True when maximum pulse length is reached
        terminated, truncated = False, False

        # Check if episode is done
        if self.cur_idx >= self.pulse_length - 1:
            truncated = True

        # Calculate reward for current action
        if (self.sparse and not truncated) or oob_pulse:
            reward = 0.
        else:
            reward = self.reward_function(self.step_states)

        if reward > self.REW_THRESH:
            terminated = True
            # reward = self.REW_THRESH * 10
            # reward = 1.
        self.rewards.append(reward)

        # Construct observation for agent using state probabilities and normed actions
        self.current_state = self.extract_current_state(self.step_states, self.cur_idx)

        self.cur_idx += 1
        return self.current_state, reward, terminated, truncated, {}

    def get_tril_elements(self, rho_matrices):
        state_probas = []
        for state in rho_matrices:
            for i in range(self.nb_basis_states):
                for j in range(i + 1):
                    state_probas.append(state[i, j].real)
                    if i != j:
                        state_probas.append(state[i, j].imag)
        state_probas = np.array(state_probas)

        return state_probas

    def extract_current_state(self, step_states, idx):
        if self.use_full_liouville:
            observable_states = [step_states[i * self.nb_basis_states + i] for i in range(self.nb_basis_states)]
        else:
            observable_states = step_states

        if self.inc_off_diag:
            state_probas = self.get_tril_elements(observable_states)
        else:
            state_probas = np.real(np.concatenate([np.diag(item) for item in observable_states]))

        normed_latest_amplitudes = np.divide([self.pulse_amplitudes_norm[lbl][idx] for lbl in self.channel_labels], self.A_norm_max)

        return np.concatenate([state_probas, normed_latest_amplitudes])

    def forward_dynamics(self, from_states, amp_deltas):
        for i, _ in enumerate(self.simulator.pulses):
            prev_amp = self.amps_cur[i]
            self.amps_cur[i] = prev_amp + amp_deltas[i]
            self.simulator.pulses[i].coeff = np.array([self.amps_cur[i]])
            self.simulator.pulses[i].tlist = np.array([0, self.dt])

        final_states = []
        for state in from_states:
            result = self.simulator.run_state(init_state=state, analytical=False)
            final_state = result.states[-1]
            final_states.append(final_state)

        return final_states

    def reward_function(self, step_states):
        # f = F_avg(self.final_states_all[-1], self.basis_states_ket, self.unitary, self.mapped_basis_states, prec=self.PREC)
        # f = F_avg(self.final_states_all[-1], self.initial_states, self.unitary, self.target_states, prec=self.PREC)
        # f = np.mean([fidelity(f, t) for f, t in zip(step_states, self.target_states)])
        f = self.fidelity_fn(step_states, self.target_states)
        # f = F_avg(self.final_states_all[-1], self.basis_states_ket, self.unitary, self.mapped_basis_states,
        #           prec=self.PREC)
        return self.fid2rew(f)

    @staticmethod
    def fidelity_fn(from_states, to_states):
        # print(from_states, to_states)
        if isinstance(from_states[0], np.ndarray) or isinstance(to_states[0], np.ndarray):
            if isinstance(from_states[0], np.ndarray):
                from_states = [s.data.todense() for s in from_states]
            if isinstance(to_states[0], np.ndarray):
                to_states = [s.data.todense() for s in to_states]
            return np.mean([np.abs(s @ t) ** 2 for s, t in zip(from_states, to_states)])

        return np.mean([np.abs(s.overlap(t)) ** 2 for s, t in zip(from_states, to_states)])

    @staticmethod
    def fid2rew(fid):
        return -np.log10(1 - fid)

    @staticmethod
    def rew2fid(rew):
        return 1 - np.power(10, -rew)

    def get_ep_fidelities(self):
        # fidelities = [[self.fidelity_fn(f, self.target_states) for f in self.final_states_all[i]] for i in range(self.cur_idx)]
        # fidelities = np.mean(fidelities, axis=1)
        # fidelities = np.clip([self.rew2fid(r) for r in self.rewards], a_min=1e-5, a_max=1) * 100.
        fidelities = [self.rew2fid(r) for r in self.rewards]

        return fidelities

    def render(self, mode='human', save_path=None, fps=80, **kwargs):
        mpl.rcParams['font.size'] = 10

        states = self.final_states_all
        actions = [self.pulse_amplitudes_norm[lbl] for lbl in self.channel_labels]

        fidelities = self.get_ep_fidelities()
        recons_amps, global_tlist = self.get_reconstructed_pulses_with_uniform_time()
        recons_amps_norm = self.norm_action(recons_amps.copy())

        # Setup figure and axes
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle(f'CNOT\nReal F_avg={self.use_full_liouville}')
        gs = mpl.gridspec.GridSpec(5, self.nb_basis_states)

        ax_state_titles = []
        for i in range(self.n_levels):
            for j in range(self.n_levels):
                ax_state_titles.append(f'$|{i}{j}\\rangle$')

        ax_state = []
        for i in range(self.nb_basis_states):
            ax_state.append(fig.add_subplot(gs[0, i]))
            ax_state[i].set_title(ax_state_titles[i])

        ax_current_state = fig.add_subplot(gs[1, :])

        ax_action = fig.add_subplot(gs[2, :])
        ax_action_deltas = fig.add_subplot(gs[3, :])
        ax_reward = fig.add_subplot(gs[4, :])

        # Initialize the plot
        EPS = self.PLOT_LOG_EPS
        norm = LogNorm(vmin=EPS, vmax=1)

        def real_norm_mat(mat, EPS):
            mat = np.clip(np.abs(np.flipud(mat)), a_min=EPS, a_max=1)
            return mat

        tick_labels = []
        for i in range(self.n_levels):
            for j in range(self.n_levels):
                tick_labels.append(f'$|{i}{j}\\rangle$')

        # Adjust for aesthetics
        for i, ax in enumerate(ax_state):
            ax.set_xlim(-0.5, self.nb_basis_states - 0.5)
            ax.set_ylim(-0.5, self.nb_basis_states - 0.5)
            ax.set_xticks(np.arange(self.nb_basis_states), tick_labels)
            ax.set_yticks(np.arange(self.nb_basis_states), reversed(tick_labels))

        ax_current_state.set_title('Current state components')

        ax_action.set_xlim(0, self.cur_idx - 1)
        # if self.cur_idx == 0:
        #     A = 1.
        # else:
        #     A = np.max(np.abs(recons_amps_norm[:, :self.cur_idx]))
        A = self.A_norm_max
        ax_action.set_ylim(-A, A)  # Adjust based on action range
        ax_action.set_ylabel('Pulse amplitude')  # Adjust based on action range
        ax_action.set_xlabel('Steps')  # Adjust based on action range

        ax_action_deltas.set_xlim(0, self.cur_idx - 1)
        ax_action_deltas.set_ylim(-1.1, 1.1)  # Adjust based on action range
        ax_action_deltas.set_ylabel('Pulse deltas')  # Adjust based on action range
        ax_action_deltas.set_xlabel('Steps')  # Adjust based on action range

        # print(self.rewards)
        # ax_reward.set_ylim(min(self.rewards), max(self.rewards))  # Adjust based on expected reward range
        ax_reward.set_ylim(min(fidelities)*0.9, max(fidelities)*1.1)  # Adjust based on expected reward range
        # ax_reward.set_yscale('log')
        ax_reward.set_ylabel('Fidelity (%)')
        ax_reward.set_xlabel('Steps')

        # Initialize plots
        ims = []
        s=None
        for i, ax in enumerate(ax_state):
            s = real_norm_mat(states[0][i], EPS)
            im = ax.imshow(s, animated=True, cmap=cm.plasma, norm=norm, origin='upper')
            ims.append(im)
            fig.colorbar(im, ax=ax)

            # Explicitly build grid
            for j in np.arange(self.nb_basis_states + 1):
                ax.axhline(j - 0.5, color='w', linestyle='-', linewidth=1)
                ax.axvline(j - 0.5, color='w', linestyle='-', linewidth=1)
        img_cur_state = ax_current_state.imshow(self.get_tril_elements(states[0]).reshape(self.nb_basis_states, -1), animated=True, cmap=cm.plasma, origin='upper')

        # Plot ideal pulses - dash-cross
        for lbl, pulse in zip(self.channel_labels, recons_amps_norm):
            ax_action.plot(pulse, ls='dashed', alpha=0.7, marker='x', label=f'Ideal {lbl}')
        ax_reward.set_xlim(0, self.cur_idx - 1)

        action_lines = [ax_action.plot([], [], lw=1.2, label=ch, marker='.')[0] for _, ch in enumerate(self.channel_labels)]
        action_delta_lines = [ax_action_deltas.plot([], [], lw=1.2, label=ch, marker='.')[0] for _, ch in enumerate(self.channel_labels)]
        ax_action.legend(loc='upper right', ncol=2)
        ax_action_deltas.legend(loc='upper right')
        reward_line, = ax_reward.plot([], [], 'g-', lw=2, marker='.')

        def init():
            for i, ax in enumerate(ax_state):
                if self.use_full_liouville:
                    idx = i * self.nb_basis_states + i
                else:
                    idx = i
                im = ax.imshow(real_norm_mat(states[0][idx], EPS), animated=True, cmap=cm.plasma, norm=norm, origin='upper')
                ims[i] = im

            img_cur_state = ax_current_state.imshow(self.get_tril_elements(states[0]).reshape(self.nb_basis_states, -1), animated=True, cmap=cm.plasma, origin='upper')

            for line in action_lines:
                line.set_data([], [])
            for line in action_delta_lines:
                line.set_data([], [])
            reward_line.set_data([], [])

            return *ims, img_cur_state, *action_lines, *action_delta_lines, reward_line

        def update(frame):
            for i, ax in enumerate(ax_state):
                ax.set_xlim(-0.5, self.nb_basis_states - 0.5)
                ax.set_ylim(-0.5, self.nb_basis_states - 0.5)
                if self.use_full_liouville:
                    idx = i * self.nb_basis_states + i
                else:
                    idx = i
                s = real_norm_mat(states[frame][idx], EPS)
                im = ax.imshow(s, animated=True, cmap=cm.plasma, norm=norm, origin='upper')
                ims[i] = im
                ax.set_title(f'Real(Basis {ax_state_titles[i]}) Trace={np.sum(np.diag(np.abs(states[frame][i].data.todense()))):.2f}')

            img_cur_state = ax_current_state.imshow(self.get_tril_elements(states[frame]).reshape(self.nb_basis_states, -1),
                                      animated=True, cmap=cm.plasma, origin='upper')

            for line, action in zip(action_lines, actions):
                line.set_data(range(frame), action[:frame])

            for line, action in zip(action_delta_lines, actions):
                line.set_data(range(frame-1), np.diff(action[:frame]))

            reward_line.set_data(range(frame), fidelities[:frame])
            ax_reward.set_title(f'Time: {global_tlist[frame]}ns  Reward: {fidelities[frame]}')

            return *ims, img_cur_state, *action_lines, *action_delta_lines, reward_line

        frames = list(range(self.cur_idx)) + [self.cur_idx - 1] * 5
        ani = FuncAnimation(fig, update, frames=frames, interval=5, init_func=init, blit=False)

        if save_path is None:
            save_path = generate_random_alphanumeric(8) + '.mp4'
        print(f'Saving to: {save_path}')
        ffwriter = mpl.animation.FFMpegWriter(fps=10)
        ani.save(save_path, dpi=60, writer=ffwriter)


if __name__ == '__main__':
    env = QuPulseEpisodicEnv(pulse_length=300)
    print(str(env))
    env = env()
    print(str(env))