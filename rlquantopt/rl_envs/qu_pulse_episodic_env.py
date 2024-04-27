import os.path
from itertools import product
from datetime import datetime as dt
from typing import SupportsFloat, Any

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

from qutip_qip.circuit import QubitCircuit
from qutip_qip.device import SCQubits
from qutip import basis, ptrace, ket, Bloch
from krotov.functionals import F_avg


class QuPulseEpisodicEnv(gym.Env):
    # Action scaling symmetric around 0
    action_channel_scaling = {
        'zx': 25e-5,
        'sx': 15e-4,
        'sy': 15e-4,
        'sz': 5e-5
    }
    # action_channel_scaling = {
    #     'zx': 0.05,
    #     'sx': 0.05,
    #     'sy': 0.05,
    #     'sz': 0.005
    # }
    dt = 1
    REW_THRESH = 0.9

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

    def __init__(self, pulse_length=None):
        super(QuPulseEpisodicEnv, self).__init__()
        self.cur_idx = 0
        self.rewards = None

        # Initialise simulator
        self.n_levels = n_levels = 2
        self.qc = qc = QubitCircuit(N=2)
        # qc.add_gate("CNOT", controls=0, targets=1)
        qc.add_gate("SWAP", targets=[0, 1])

        self.simulator = SCQubits(num_qubits=2, dims=[n_levels, n_levels], wq=[5.15, 5.09], wr=[5.96, 5.96], g=[0.1, 0.1], alpha=[-0.3, -0.3], omega_single=[0.01, 0.01], omega_cr=[0.01, 0.01], t1=50.e3, t2=20.e3)
        # print(processor.get_control('sx0'))
        self.simulator.pulse_mode = 'discrete'
        self.simulator.load_circuit(qc)
        pulses = self.simulator.pulses
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
        self.action_space = spaces.Box(low=-1, high=1, shape=(n_channels,), dtype=float)
        self.n_act = n_channels

        # Action space holds channel amplitudes throughout episode
        self.pulse_amplitudes = self.init_pulse_amplitudes()

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
        self.basis_states = [self.psi00, self.psi01, self.psi10, self.psi11]
        self.full_liouville_basis = [psi * phi.dag() for psi, phi in product(self.basis_states, self.basis_states)]

        self.unitary = qc.compute_unitary()
        self.mapped_basis_states = [sum(complex(self.unitary[i, j]) * self.basis_states[i]
                                        for i in range(self.unitary.shape[0])) for j in range(self.unitary.shape[1])]

        # Lots of gates just rearrange the basis states, and we can avoid some complexity by identifying
        # this and setting the mapped_basis_states to the identical objects as the original basis_states
        for i, state in enumerate(self.mapped_basis_states):
            for j, basis_state in enumerate(self.basis_states):
                if state == basis_state:
                    self.mapped_basis_states[i] = basis_state

        self.initial_states = [self._rho1(self.basis_states), self._rho2(self.basis_states),
                               self._rho3(self.basis_states)]
        # self.initial_states = self.basis_states
        self.initial_state_purities = [s.purity() for s in self.initial_states]
        self.weights = np.array([1. / (3 * p) for p in self.initial_state_purities])
        self.target_states = [self._rho1(self.mapped_basis_states), self._rho2(self.mapped_basis_states),
                              self._rho3(self.mapped_basis_states)]
        # self.target_states = [qc.run(init_state) for init_state in self.initial_states]

        self.n_obs = self.nb_basis_states * len(self.initial_states) + n_channels
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.n_obs,), dtype='float')

    def __str__(self):
        return (f'Quantum circuit: {self.qc}\n'
                f'Action channels: {self.channel_labels}\n'
                f'Pulse length: {self.pulse_length}\n'
                f'Time delta: {self.dt}')

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
        return np.array(action)

    def reset(self, *params):
        self.cur_idx = 0
        self.final_states_all = []
        self.pulse_amplitudes = self.init_pulse_amplitudes()
        self.amps_cum = np.zeros(self.n_act)
        self.step_states = self.full_liouville_basis.copy()
        # TODO: initialise state probas. appropriately - i.e. sum up to one
        self.current_state = np.zeros(self.n_obs)
        self.rewards = []
        return self.current_state

    def step(self, action):
        return self.step_direct(action)
        # return self.step_naive(action)

    def step_direct(self, action):
        action_denorm = self.denorm_action(action)

        self.step_states = self.directed_forward_dynamics(from_states=self.step_states, amp_deltas=action_denorm)

        done, success = False, False

        # Check if episode is done
        if self.cur_idx >= self.pulse_length - 1:
            done = True

        # Calculate reward for current action
        reward = self.reward_function(self.step_states)
        # if done:
        #     reward = self.reward_function()
        # else:
        #     reward = 0

        self.rewards.append(reward)

        if reward > self.REW_THRESH:
            success = True
            done = True

        # Construct observation for agent using state probabilities and normed actions
        state_probas = np.real(np.concatenate([np.diag(item) for item in self.step_states]))
        normed_latest_amplitudes = self.norm_action(
            [self.pulse_amplitudes[lbl][self.cur_idx] for lbl in self.channel_labels])
        self.current_state = np.concatenate([state_probas, normed_latest_amplitudes])

        self.cur_idx += 1
        return self.current_state, reward, done, success

    def directed_forward_dynamics(self, from_states, amp_deltas):
        for i, _ in enumerate(self.simulator.pulses):
            prev_amp = self.amps_cum[i]
            self.amps_cum[i] = prev_amp + amp_deltas[i]
            self.simulator.pulses[i].coeff = np.array([prev_amp, self.amps_cum[i]])
            self.simulator.pulses[i].tlist = np.array([0, 1e-3, self.dt])

        final_states = []
        for state in from_states:
            result = self.simulator.run_state(init_state=state)
            final_state = result.states[-1]
            final_states.append(final_state)

        return final_states

    def step_naive(self, action):
        # print(action)
        prev_action_denorm = np.zeros(self.n_channels)
        if self.cur_idx > 0:
            prev_action_denorm = [self.pulse_amplitudes[lbl][self.cur_idx - 1] for lbl in self.channel_labels]

        action_denorm = np.add(prev_action_denorm, self.denorm_action(action))

        # Save current action to full, absolute pulse history
        for i, channel_label in enumerate(self.channel_labels):
            self.pulse_amplitudes[channel_label][self.cur_idx] = action_denorm[i]

        done = False
        success = False

        # Simulate pulses and get final states
        final_states = self.forward_dynamics()
        self.final_states_all.append(final_states)

        # Check if episode is done
        if self.cur_idx >= self.pulse_length - 1:
            done = True

        # Calculate reward for current action
        # reward = self.reward_function()
        if done:
            reward = self.reward_function()
        else:
            reward = 0

        self.rewards.append(reward)

        if reward > self.REW_THRESH:
            success = True
            done = True

        # Obtain basis vector probabilities for rho1/2/3 initial state evolutions
        state = np.real(np.concatenate([np.diag(item) for item in final_states]))

        # Obtain latest normalised amplitudes
        normed_latest_amplitudes = self.norm_action([self.pulse_amplitudes[lbl][self.cur_idx] for lbl in self.channel_labels])

        # Concatenate the 3 states obtained when starting from rho1/2/3 and running pulse up to current time AND the latest pulse amplitudes
        self.current_state = np.concatenate([state, normed_latest_amplitudes])

        self.cur_idx += 1
        return self.current_state, reward, done, success

    def forward_dynamics(self, amplitudes=None):
        if not amplitudes:
            # amplitudes = self.get_cur_normed_pulse_amplitudes(return_dict=False)
            amplitudes = [self.pulse_amplitudes[lbl][:self.cur_idx + 1] for lbl in self.channel_labels]

        for i, _ in enumerate(self.simulator.pulses):
            self.simulator.pulses[i].coeff = np.asarray(amplitudes[i], dtype=float)
            n_bins = len(amplitudes[i])
            tlist = np.linspace(0, self.dt * (n_bins), n_bins + 1, dtype='float')
            self.simulator.pulses[i].tlist = tlist

        # if self.cur_idx % 25 == 0:
        #     self.simulator.plot_pulses(show_axis=True)
        #     plt.show()

        # Use qiskit.pulse to simulate the system's forward dynamics based on the amplitudes
        final_states = []
        # for init_state in self.initial_states:
        for init_state in self.basis_states:
            result = self.simulator.run_state(init_state=init_state)
            final_state = result.states[-1]
            final_states.append(final_state)
        # return result.states[-1]
        return final_states

    def reward_function(self, step_states=None) -> float:
        # TODO: expand fidelity computation to cover all basis vectors
        next_states = []
        if step_states is None:
            step_states = self.full_liouville_basis
        # for full_louis in self.full_liouville_basis:
        for step_state in step_states:
            next_states.append(self.simulator.run_state(step_state).states[-1])

        f = F_avg(next_states, self.basis_states, self.unitary, self.mapped_basis_states, prec=1e-4)
        return f

    def render(self, **kwargs) -> RenderFrame | list[RenderFrame] | None:
        mpl.rcParams['font.size'] = 10

        states = self.final_states_all

        # Setup figure and axes
        fig = plt.figure(tight_layout=True, figsize=(20, 12))
        fig.suptitle('CNOT')
        gs = mpl.gridspec.GridSpec(3, len(self.basis_states))

        ax_state_titles = []
        for i in range(self.n_levels):
            for j in range(self.n_levels):
                ax_state_titles.append(f'$|{i}{j}\\rangle$')

        ax_state = []
        for i in range(len(self.basis_states)):
            ax_state.append(fig.add_subplot(gs[0, i]))
            ax_state[i].set_title(ax_state_titles[i])

        ax_action = fig.add_subplot(gs[1, :])
        ax_reward = fig.add_subplot(gs[2, :])

        # Initialize the plot
        matrix = states[0][0]
        EPS = 1e-4
        norm = LogNorm(vmin=EPS, vmax=1)

        def norm_mat(mat, EPS):
            # lmat = np.log10(mat)
            # return (lmat + 6)/ 6
            mat = np.clip(mat, a_min=EPS, a_max=1)
            return mat

        tick_labels = []
        for i in range(self.n_levels):
            for j in range(self.n_levels):
                tick_labels.append(f'$|{i}{j}\\rangle$')

        ims = []
        for i, ax in enumerate(ax_state):
            im = ax.imshow(norm_mat(np.real(matrix), EPS), animated=True, cmap=cm.plasma, norm=norm)
            fig.colorbar(im, ax=ax)
            for j in np.arange(self.nb_basis_states + 1):
                ax.axhline(j - 0.5, color='w', linestyle='-', linewidth=1)
                ax.axvline(j - 0.5, color='w', linestyle='-', linewidth=1)


        # Adjust for aesthetics
        # ax_action.set_xlim(0, self.cur_idx - 1)  # Assuming 100 timesteps, adjust as necessary
        # ax_action.set_ylim(-0.05, 0.05)  # Adjust based on action range
        ax_action.set_ylabel('Pulse amplitude')  # Adjust based on action range
        ax_action.set_xlabel('Steps')  # Adjust based on action range

        ax_reward.set_xlim(0, self.cur_idx - 1)  # Assuming 100 timesteps, adjust as necessary
        ax_reward.set_ylim(min(self.rewards), max(self.rewards))  # Adjust based on expected reward range
        ax_reward.set_ylabel('Fidelity')
        ax_reward.set_xlabel('Steps')

        # Initialize plots
        action_lines = [ax_action.plot([], [], lw=1.2, label=ch)[0] for _, ch in enumerate(self.channel_labels)]
        ax_action.legend(loc='upper right')
        reward_line, = ax_reward.plot([], [], 'g-', lw=1.5)

        # Initialize data
        time = np.arange(self.pulse_length)  # Placeholder for timesteps

        def init():
            ims = []
            for basis, ax in zip(self.basis_states, ax_state):
                im = ax.imshow(norm_mat(np.real(basis), EPS), animated=True, cmap=cm.plasma, norm=norm, origin='upper')
                ims.append(im)
                ax.set_xlim(-0.5, self.nb_basis_states - 0.5)
                ax.set_ylim(-0.5, self.nb_basis_states - 0.5)
                ax.set_xticks(np.arange(self.nb_basis_states), tick_labels)
                ax.set_yticks(np.arange(self.nb_basis_states), tick_labels)

            for lbl, pulse in zip(self.channel_labels, self.ideal_pulses['coeff']):
                ax_action.plot(pulse)

            for line in action_lines:
                line.set_data([], [])
            reward_line.set_data([], [])

            return *ims, *action_lines, reward_line

        actions = [self.pulse_amplitudes[lbl] for lbl in self.channel_labels]
        def update(frame):
            for i, (im, ax) in enumerate(zip(ims, ax_state)):
                # ax.set_title(f'{(frame + 1) * 100. / self.pulse_length:.2f}%')
                # ax.set_title(f'$\\rho {i+1}$')
                im.set_array(norm_mat(np.real(states[frame][i]), EPS))

            # actions = self.get_cur_normed_pulse_amplitudes(return_dict=False)

            for line, action in zip(action_lines, actions):
                line.set_data(time[:frame], action[:frame])

            rewards = self.rewards
            reward_line.set_data(time[:frame], rewards[:frame])
            ax_reward.set_title(rewards[frame])
            # ax_reward.set_ylim((min(rewards)/2, max(rewards)))

            return *ims, *action_lines, reward_line

        frames = list(range(self.cur_idx)) + [self.cur_idx - 1] * 30
        ani = FuncAnimation(fig, update, frames=frames, interval=20, init_func=init, blit=True)

        save_path = kwargs.get('save_path', '')
        if os.path.exists(os.path.dirname(save_path)):
            ani.save(save_path, dpi=80, writer='imagemagick')

        plt.show()
