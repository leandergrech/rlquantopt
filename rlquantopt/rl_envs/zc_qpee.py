from gymnasium import spaces
import numpy as np
from functools import reduce
from tqdm import tqdm
import pandas as pd
from qutip import Qobj, ket, QobjEvo, SESolver, expect
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from gymnasium import Env

# from rlquantopt.rl_envs.qu_pulse_episodic_env import QuPulseEpisodicEnv as QPEE
from rlquantopt.rl_envs.zcqubits import ZCQubits, fidelity
# from rlquantopt.utils import generate_random_alphanumeric


class ZCQPEE(Env):
    action_channel_scaling = {'z': 1e-1}
    # optimised_pulse_path = '/home/leander/code/rlquantopt/rlquantopt/rl_envs/configs/mc_optimised_pulse.csv'
    optimised_pulse_path = '/home/leander/code/rlquantopt/rlquantopt/rl_envs/configs/data_lilmc.csv'

    FID_THRESH = 0.995
    PREC = 5e-4
    A_norm_max = 10
    PLOT_LOG_EPS = 1e-3
    cmap = cm.CMRmap
    REW_THRESH = None

    def __init__(self, pulse_length):
        ZCQPEE.REW_THRESH = self.fid2rew(ZCQPEE.FID_THRESH)

        # Set up model
        self.n_levels = n_levels = 3
        params = {
            "omega_s": [5.8899, 5.0311],
            "alpha_s": [-324e-3, -235e-3],
            "g": [100e-3, 71.4e-3]
        }
        qubit_dims = [n_levels, n_levels]
        coupler_dims = 3
        full_dims = qubit_dims.copy()
        full_dims.append(coupler_dims)

        self.psi00 = ket((0, 0, 0), dim=full_dims)
        self.psi01 = ket((0, 1, 0), dim=full_dims)
        self.psi10 = ket((1, 0, 0), dim=full_dims)
        self.psi11 = ket((1, 1, 0), dim=full_dims)
        self.basis_states_ket = [self.psi00, self.psi01, self.psi10, self.psi11]

        self.simulators = [ZCQubits(2, qubit_dims=qubit_dims, coupler_dims=coupler_dims, params=params) for _ in range(len(self.basis_states_ket))]

        self.channel_labels = ['z']

        self.T = T = 200 # ns
        self.pulse_length = pulse_length
        self.tlist = np.linspace(0, T, pulse_length)
        self.dt = T/(self.pulse_length - 1) # dt obtained after np.linspace(0, T, pulse_length)
        self.pulse_amplitudes_norm = self.init_pulse_amplitudes()

        coeff, tlist = self.get_optimal_pulse()
        self.ideal_pulses = {'labels': self.channel_labels,
                             'coeff': [coeff],
                             'tlist': [tlist],
                             'max_len': len(coeff),
                             'max_time': tlist[-1]}

        self.unitary = unitary = Qobj(np.array([
            [1, 0, 0, 0],
            [0, 0, 1j, 0],
            [0, 1j, 0, 0],
            [0, 0, 0, 1]
        ]), dims=[[2, 2], [2, 2]])  # iSWAP

        # Set up state vectors

        self.basis_states = basis_states = self.basis_states_ket.copy()
        self.basis_states_str = ('$|000\\rangle$', '$|010\\rangle$','$|100\\rangle$','$|110\\rangle$',)
        self.initial_states = self.basis_states.copy()
        # self.solvers = [SESolver(self.simulator.H) for _ in self.basis_states]
        self.solvers = [SESolver(self.simulators[i].H) for i in range(len(self.basis_states))]

        mapped_basis_states = [sum(unitary[i, j] * basis_states[i]
                                   for i in range(unitary.shape[0])) for j in range(unitary.shape[1])]
        # Lots of gates just rearrange the basis states, and we can avoid some complexity by identifying
        # this and setting the mapped_basis_states to the identical objects as the original basis_states
        for i, state in enumerate(mapped_basis_states):
            for j, basis_state in enumerate(basis_states):
                if state == basis_state:
                    mapped_basis_states[i] = basis_state
        self.target_states = mapped_basis_states.copy()

        # Set up action space
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.n_channels = 1
        self.amps_cur = np.array([0.])
        self.n_act = 1

        # Set up observation space
        self.n_obs = len(self.basis_states) * 2 * reduce(lambda x, y: x*y, full_dims) + self.n_act + 1  # +1 is the time dimension
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.n_obs,), dtype=np.float32)

        # All initialised after first call to reset method
        self.cur_idx = 0
        self.final_states_all = None
        self.pulse_amplitudes_norm = None
        self.amps_cur = None
        self.step_states = None
        self.current_state = None
        self.rewards =None
        self.actions_all = None

    def get_optimal_pulse(self):
        data = pd.read_csv(self.optimised_pulse_path)
        tlist = data['tlist'].to_numpy()
        coeff = data['amplist'].to_numpy()

        return coeff, tlist

    def init_pulse_amplitudes(self):
        # return np.zeros((self.n_channels, self.pulse_length))  # First row for u01, second row for d1
        buffer = {}
        for channel_label in self.channel_labels:
            buffer[channel_label] = np.zeros(self.pulse_length + 1)  # +1 since first amp must be zero

        return buffer

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.cur_idx = 0
        self.final_states_all = [self.initial_states.copy()]
        self.pulse_amplitudes_norm = self.init_pulse_amplitudes()
        self.amps_cur = np.zeros(self.n_act)
        self.step_states = self.initial_states.copy()
        self.current_state = self.extract_current_state(self.step_states, idx=0).astype(np.float32)
        self.actions_all = []
        self.rewards = []

        for solver, basis_state in zip(self.solvers, self.basis_states):
            solver.start(basis_state, 0.)

        # for k, solver in enumerate(self.solvers):
        #     self.solvers[k].start(from_states[k], 0.)
        # self.fidelities = []
        return self.current_state, {}

    def extract_current_state(self, states, action=None, idx=None, state_only=False):
        obs = []
        for state in states:
            if isinstance(state, Qobj):
                state = state.full()
            obs.append(np.concatenate([state.real.copy(), state.imag.copy()]))
        if not state_only:
            if action is None:
                obs.append([self.amps_cur / self.A_norm_max])
            else:
                obs.append(np.array(action).reshape(-1, 1))
            # Add also the time of current step
            obs.append(np.array(self.cur_idx/self.pulse_length).reshape(-1, 1))
        obs = np.concatenate(obs).squeeze()
        return obs.astype(np.float32)

    def norm_action(self, action):
        SCALE = self.action_channel_scaling['z']
        return action / SCALE

    def denorm_action(self, action):
        SCALE = self.action_channel_scaling['z']
        return action * SCALE

    def step(self, action):
        """
        Add a pulse amplitude delta vector (action) on the previous value of the pulse amplitude.
        :param action: Must be list-like with `self.n_abs` dimensions
        :return: observation_tp1, reward, terminated, truncated, info
        """
        # Absolute amplitude conversion required for rendering
        self.actions_all.append(action[0])
        prev_abs_action = np.zeros(self.n_channels)
        oob_pulse = False   # Out of bounds absolute pulse
        if self.cur_idx > 0:
            prev_abs_action = [self.pulse_amplitudes_norm[lbl][self.cur_idx - 1] for lbl in self.channel_labels]
        for i, channel_label in enumerate(self.channel_labels):
            # Delta formalism
            abs_action = prev_abs_action[i] + action[i]
            if abs(abs_action) > self.A_norm_max:
                oob_pulse = True
                abs_action = np.sign(abs_action) * self.A_norm_max
            # Absolute formalism
            # abs_action = action[i]
            self.pulse_amplitudes_norm[channel_label][self.cur_idx] = abs_action

        # Environment dynamics
        action_denorm = self.denorm_action(action.copy())
        from_states = self.step_states.copy()
        self.step_states = self.forward_dynamics(from_states=from_states, amp_deltas=action_denorm)
        self.final_states_all.append(self.step_states.copy())

        # terminated is only True when reward threshold is exceeded
        # truncated is only True when maximum pulse length is reached
        terminated, truncated = False, False

        # Check if episode is done
        if self.cur_idx >= self.pulse_length - 1:
            truncated = True

        # Calculate reward for current action
        if oob_pulse:
            reward = 0.
        else:
            reward = self.reward_function(self.step_states)
        if reward > self.REW_THRESH:
            terminated = True
        self.rewards.append(reward)

        # Construct observation for agent using state probabilities and normed actions
        self.current_state = self.extract_current_state(self.step_states, self.cur_idx)

        self.cur_idx += 1
        return self.current_state, reward, terminated, truncated, {}

    def forward_dynamics(self, from_states, amp_deltas):
        # prev_amp = self.amps_cur.copy()
        # self.amps_cur = np.add(prev_amp, amp_deltas)
        # self.amps_cur += amp_deltas
        self.amps_cur = np.array(amp_deltas).reshape(-1)
        final_states = []
        for k, solver in enumerate(self.solvers):
            # self.solvers[k].start(from_states[k], 0.)
            # s = self.solvers[k].step((self.cur_idx + 1) * self.dt, args={'A': self.amps_cur[0]})

            s = self.solvers[k].step(self.tlist[self.cur_idx], args={'A': self.amps_cur[0]})
            final_states.append(s.copy())

        return final_states

    def reward_function(self, step_states):
        # f = sum([np.abs(step_states[i].overlap(self.target_states[i])) ** 2 for i in range(4)])
        # f = np.mean([np.abs(s.overlap(t)) ** 2 for s, t in zip(step_states, self.target_states)])
        # SCALE = 10.
        f = np.mean([fidelity(s, t) for s, t in zip(step_states, self.target_states)])
        rew = self.fid2rew(f)
#         if rew > self.REW_THRESH:
#             rew *= SCALE
        return rew

    @staticmethod
    def fid2rew(fid):
        return -np.log10(1 - fid)

    @staticmethod
    def rew2fid(rew):
#         SCALE = 10.
#         if rew > ZCQPEE.REW_THRESH * SCALE:
#             rew /= SCALE
        return 1 - np.power(10, -rew)

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

    def render(self, mode='human', save_path=None, fps=80, **kwargs):
        mpl.rcParams['font.size'] = 10

        states = self.final_states_all
        actions = [self.pulse_amplitudes_norm[lbl] for lbl in self.channel_labels]
        infidelities = (1 - np.clip([self.rew2fid(r) for r in self.rewards], a_min=1e-4, a_max=1)) * 100.
        recons_amps, global_tlist = self.get_reconstructed_pulses_with_uniform_time()
        # recons_amps = recons_amps[0]
        recons_amps_norm = self.norm_action(recons_amps.copy())

        # Setup figure and axes
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle(f'ISWAP')
        gs = mpl.gridspec.GridSpec(4, 2)

        ax_state_titles = []
        for i in range(self.n_levels):
            for j in range(self.n_levels):
                ax_state_titles.append(f'$|{i}{j}\\rangle$')

        ax_state = fig.add_subplot(gs[0, :])
        ax_action = fig.add_subplot(gs[1, :])
        ax_action_deltas = fig.add_subplot(gs[2, :])
        ax_reward = fig.add_subplot(gs[3, :])

        # Initialize the plot
        EPS = self.PLOT_LOG_EPS
        norm = LogNorm(vmin=EPS, vmax=1)

        xtick_labels = []
        for i in range(self.n_levels):
            for j in range(self.n_levels):
                for k in range(self.n_levels):
                    xtick_labels.append(f'$|{i}{j}{k}\\rangle$')

        # Adjust for aesthetics
        len_state_ket = len(self.initial_states[0].full())
        nb_basis_states = len(self.basis_states)

        ax = ax_state
        ax.set_xlim(-0.5, len_state_ket - 0.5)
        ax.set_ylim(-0.5, nb_basis_states - 0.5)
        ax.set_xticks(np.arange(len_state_ket), xtick_labels)
        ax.set_yticks(np.arange(nb_basis_states), reversed(self.basis_states_str))

        ax = ax_action
        ax.axhline(y=0, linestyle='dashed', color='gray')
        ax.set_xlim(0, self.cur_idx - 1)
        A = self.A_norm_max
        ax.set_ylim(-A, A)  # Adjust based on action range
        ax.set_ylabel('Pulse amplitude')  # Adjust based on action range
        ax.set_xlabel('Steps')  # Adjust based on action range

        ax = ax_action_deltas
        ax.axhline(y=0, linestyle='dashed', color='gray')
        ax.set_xlim(0, self.cur_idx - 1)
        ax.set_ylim(-1.1, 1.1)  # Adjust based on action range
        ax.set_ylabel('Pulse deltas')  # Adjust based on action range
        ax.set_xlabel('Steps')  # Adjust based on action range

        ax = ax_reward
        ax.set_ylim(10**int(np.log10(min(infidelities))), 10**(int(np.log10(max(infidelities))) + 1))  # Adjust based on expected reward range
        ax.set_yscale('log')
        ax.set_ylabel('Infidelity (%)')
        ax.set_xlabel('Steps')

        # s = np.repeat(self.PREC, self.n_obs - self.n_act - 1).reshape(4, -1)
        s = np.abs(np.concatenate([s.full() for s in states[0]])).reshape(4, -1)
        im = ax_state.imshow(s, animated=True, cmap=self.cmap, norm=norm, origin='upper')
        texts = []
        ax = ax_state
        for i, srow in enumerate(s):
            for j, selem in enumerate(srow):
                text = ax.text(j, i, f'{selem:.2f}', horizontalalignment='center', verticalalignment='center', fontsize=6, c='k')
                texts.append(text)
        fig.colorbar(im, ax=ax_state)

        # Plot ideal pulses - dash-cross
        for lbl, pulse in zip(self.channel_labels, recons_amps_norm):
            ax_action.plot(pulse, ls='dashed', alpha=0.7, marker='x', label=f'Ideal {lbl}')

        ax_reward.set_xlim(0, self.cur_idx - 1)

        # action_lines = [ax_action.plot([], [], lw=1.2, label=ch, marker='.')[0] for _, ch in enumerate(self.channel_labels)]
        ch = self.channel_labels[0]
        action_line, = ax_action.plot([], [], lw=1.2, label=ch, marker='.')
        # action_delta_lines = [ax_action_deltas.plot([], [], lw=1.2, label=f'Δ{ch}', marker='.')[0] for _, ch in enumerate(self.channel_labels)]
        action_delta_line, = ax_action_deltas.plot([], [], lw=1.2, label=f'Δ{ch}', marker='.')
        ax_action.legend(loc='upper right', ncol=2)
        ax_action_deltas.legend(loc='upper right')
        reward_line, = ax_reward.plot([], [], 'g-', lw=2, marker='.')
        ax_reward.set_title('Time:  Fidelity:')
        fig.tight_layout()

        def init():
            s = np.abs(np.concatenate([s.full() for s in states[0]])).reshape(4, -1)
            s = np.flipud(s)
            im = ax_state.imshow(s, animated=True, cmap=self.cmap, norm=norm, origin='upper')

            # for line in action_lines:
            action_line.set_data([], [])
            # for line in action_delta_lines:
            action_delta_line.set_data([], [])
            reward_line.set_data([], [])

            for i, srow in enumerate(s):
                for j, selem in enumerate(srow):
                    texts[i*len(srow) + j].set_text(f'{selem:.2f}')

            # return *ims, *action_lines, *action_delta_lines, reward_line
            return im, action_line, action_delta_line, reward_line, texts

        def update(frame):
            s = np.clip(np.abs(np.concatenate([s_.full() for s_ in states[frame]])), EPS, 1).reshape(4, -1)
            s = np.flipud(s)
            im = ax_state.imshow(s, animated=True, cmap=self.cmap, norm=norm, origin='upper')

            # for line, action in zip(action_lines, [self.pulse_amplitudes_norm[self.channel_labels[0]]]):
            action_line.set_data(range(frame), self.pulse_amplitudes_norm[self.channel_labels[0]][:frame])

            # for line, action in zip(action_delta_lines, [self.actions_all]):
            action_delta_line.set_data(range(frame), self.actions_all[:frame])

            reward_line.set_data(range(frame), infidelities[:frame])
            ax_reward.set_title(f'Time: {global_tlist[frame]:.2f}ns  Fidelity: {100 - infidelities[frame]:.2f}:%')

            for i, srow in enumerate(s):
                for j, selem in enumerate(srow):
                    texts[i*len(srow) + j].set_text(f'{selem:.3f}')

            # return *ims, *action_lines, *action_delta_lines, reward_line
            return im, action_line, action_delta_line, reward_line, texts

        frames = list(np.arange(0, self.cur_idx, 2)) + [self.cur_idx - 1] * 5
        ani = FuncAnimation(fig, update, frames=frames, interval=5, init_func=init, blit=False)
        # plt.show()
        # return ani
        if save_path is None:
            print(f'Returning animation')
            return ani
        #     save_path = generate_random_alphanumeric(8) + '.mp4'
        print(f'Saving to: {save_path}')
        ffwriter = mpl.animation.FFMpegWriter(fps=10)

        pbar = tqdm(total=self.pulse_length)

        def progress_callback(current_frame: int, total_frames: int):
            nonlocal pbar
            pbar.update(1)
        print(f'Saving to: {save_path}')
        ani.save(save_path, dpi=100, writer=ffwriter, progress_callback=progress_callback)


    @staticmethod
    def evaluate_pulse(pulse_file, save_path):
        assert pulse_file.endswith('.csv')
        assert save_path.endswith('.mp4')

        data = pd.read_csv(pulse_file)
        pulse = data['amplist'].to_numpy()
        tlist = data['tlist'].to_numpy()
        pulse_length = len(tlist)
        env = ZCQPEE(pulse_length=pulse_length)
        print(env.dt)
        # exit(23)
        assert tlist[1] == env.dt

        env.reset()
        truncated = False
        terminated = False
        idx = 0
        while not terminated and not truncated:
            if idx == 0:
                action = np.array([pulse[0]])
            else:
                action = np.array([pulse[idx] - pulse[idx - 1]])

            action = env.norm_action(action)

            _, _, terminated, truncated, _ = env.step(action)
            idx += 1

        env.render(save_path=save_path)


if __name__ == '__main__':
    import os
    # par_dir = '/home/leander/code/rlquantopt/rlquantopt/rl_agents/ZCQPEE120pl-PPO/13-06-24_115241_ZCQPEE120pl/best_model/pulses'
    par_dir = '/home/leander/code/rlquantopt/rlquantopt/rl_envs/configs'
    pulse_file = os.path.join(par_dir, 'data_lilmc.csv')
    save_path = os.path.join(par_dir, 'data_lilmc.mp4')
    ZCQPEE.evaluate_pulse(pulse_file, save_path)


