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
from gymnasium.core import RenderFrame
from tqdm import trange, tqdm

from qutip_qip.circuit import QubitCircuit
from qutip_qip.device import SCQubits
from qutip import basis, ptrace, ket, Bloch
from krotov.functionals import F_avg


class QuPulseEpisodicEnv(gym.Env):
    # Action scaling symmetric around 0
    action_channel_scaling = {
        'zx': 0.01,
        'sx': 0.01,
        'sy': 0.01,
        'sz': 0.001
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
        qc.add_gate("CNOT", controls=0, targets=1)

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
        self.final_states = None

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

    # def get_cur_normed_pulse_amplitudes(self, return_dict=False):
    #     ret = {k: v[:self.cur_idx + 1] for k, v in self.pulse_amplitudes.items()}
    #     if return_dict:
    #         return ret
    #     else:
    #         return list(ret.values())

    def reset(self, *params):
        self.cur_idx = 0
        self.final_states = []
        self.pulse_amplitudes = self.init_pulse_amplitudes()
        # TODO: initialise state probas. appropriately - i.e. sum up to one
        self.current_state = np.zeros(self.n_obs)
        self.rewards = []
        return self.current_state

    def step(self, action):
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
        self.final_states.append(final_states)

        # Check if episode is done
        if self.cur_idx >= self.pulse_length - 1:
            done = True

        # Calculate reward for current action
        reward = 0
        if done:
            reward = self.reward_function()

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
            final_states.append(result.states[-1])
        # return result.states[-1]
        return final_states

    def reward_function(self):
        # TODO: expand fidelity computation to cover all basis vectors
        full_louis_results = []
        for full_louis in self.full_liouville_basis:
            full_louis_results.append(self.simulator.run_state(full_louis).states[-1])

        f = F_avg(full_louis_results, self.basis_states, self.unitary, self.mapped_basis_states)
        return f

    # def get_bloch_sphere_coordinates(self, qobj):
    #     # Ensure qobj is a density matrix
    #     if not qobj.isoper:# or qobj.dims[0][0] != 2:
    #         raise ValueError("Qobj must be a density matrix of a qubit.")
    #
    #     # Extract diagonal elements
    #     p0 = qobj[0, 0].real  # Probability of being in |00>
    #     p1 = qobj[4, 4].real  # Probability of being in |11>
    #
    #     # Calculate z from probabilities
    #     z = p0 - p1
    #
    #     # Extract off-diagonal elements for x and y if possible
    #     coherence = qobj[1, 1]  # Coherence between |00> and |11> seen as |01>
    #     x = 2 * coherence.real
    #     y = 2 * coherence.imag
    #
    #     return x, y, z

    def render(self, **kwargs) -> RenderFrame | list[RenderFrame] | None:
        mpl.rcParams['font.size'] = 10

        states = self.final_states

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


from cycler import cycler
mpl.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')


def testing_actions():
    env = QuPulseEpisodicEnv()
    print(env)
    ch_lbls = env.channel_labels

    ideal_pulses = env.ideal_pulses
    T = ideal_pulses['max_time']
    N = ideal_pulses['max_len']
    global_tlist = np.linspace(0, T, N)
    ideal_actions = ideal_pulses['coeff']
    tlists = ideal_pulses['tlist']

    init_state = env.reset()

    action = prev_action = np.zeros(env.n_act)

    fig, axs = plt.subplots(2)
    ax = axs[0]
    cmap = mpl.cm.get_cmap('tab10')

    for i, (lbl, ideal_act, tl) in enumerate(zip(ch_lbls, ideal_actions, tlists)):
        ax.plot(tl, ideal_act, c=cmap(i / (env.n_act - 1)), label=lbl, marker='.')

    # For pulse duration
    all_actions = np.empty(shape=(N, env.n_act))
    for i, t in enumerate(tqdm(global_tlist)):
        cur_action = np.zeros(env.n_act)
        for j, (ideal_act, tli) in enumerate(zip(ideal_actions, tlists)):
            idx = np.argmin(np.square(tli - t))
            if idx > 0:
                cur_action[j] = ideal_act[idx] - ideal_act[idx-1]

        all_actions[i] = env.norm_action(cur_action)

    ax = axs[1]
    for j, (acts, ch_lbl) in enumerate(zip(all_actions.T, ch_lbls)):
        ax.scatter(global_tlist, acts, c=cmap(j / (env.n_act - 1)), marker='^', label=ch_lbl)

    for ax in axs:
        ax.legend(loc='best')

    fig.tight_layout()
    plt.show()


def main():
    env = QuPulseEpisodicEnv()
    print(env)

    ideal_pulses = env.ideal_pulses
    T = ideal_pulses['max_time']
    N = ideal_pulses['max_len']
    global_tlist = np.linspace(0, T, N)
    ideal_actions = ideal_pulses['coeff']
    tlists = ideal_pulses['tlist']

    init_state = env.reset()

    for i, t in enumerate(tqdm(global_tlist)):
        if i > 2:
            break
        cur_action = np.zeros(env.n_act)
        for j, (ideal_act, tli) in enumerate(zip(ideal_actions, tlists)):
            idx = np.argmin(np.square(tli - t))
            if idx > 0:
                cur_action[j] = ideal_act[idx] - ideal_act[idx - 1]

        # cur_action = env.norm_action(cur_action)
        cur_action = env.norm_action(cur_action)
        # cur_action = np.random.uniform(-1, 1, env.n_act)
        env.step(cur_action)

    # env.render(save_path=f'QuPulseEpisodicEnv_ideal_pulse_{dt.now().strftime("%m%d%yT%H%M%S")}.gif')
    env.render()


if __name__ == '__main__':
    # testing_actions()
    main()
