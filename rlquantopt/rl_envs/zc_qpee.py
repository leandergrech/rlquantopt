from gymnasium import spaces
import yaml
import numpy as np
from tqdm import tqdm
import pandas as pd
from qutip import Qobj, ket, QobjEvo, SESolver, expect
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from gymnasium import Env

from rlquantopt.rl_envs.zcqubits import ZCQubits, fidelity


class ZCQPEE(Env):
    action_scaling = {'z': 1e-1}
    # optimised_pulse_path = '/home/leander/code/rlquantopt/rlquantopt/rl_envs/configs/mc_optimised_pulse.csv'
    # optimised_pulse_path = 'configs/data_lilmc/data_lilmc.csv'

    PREC = 5e-4
    A_norm_max = 10
    PLOT_LOG_EPS = 1e-3
    cmap = cm.CMRmap

    def to_yaml(self, save_path=None):
        data = dict(pulse_length=self.pulse_length,
                    delta_mode=self.delta_mode,
                    T=self.T,
                    fid_thresh=self.FID_THRESH,
                    action_scaling=self.action_scaling,
                    a_norm_max=self.A_norm_max,
                    optimised_pulse_path=self.optimised_pulse_path,
                    model_params=self.all_model_params)

        if save_path is not None:
            with open(save_path, 'w') as f:
                yaml.dump(data, f)

        return data

    @classmethod
    def from_yaml(cls, load_path):
        with open(load_path, 'r') as f:
            kw = yaml.load(f, yaml.SafeLoader)
        self = cls(**kw)
        return self

    def __init__(self, pulse_length=120, delta_mode=True, T=50, fid_thresh=0.995, action_scaling=None, a_norm_max=None, optimised_pulse_path=None, model_params=None):
        self.FID_THRESH = fid_thresh
        self.REW_THRESH = self.fid2rew(self.FID_THRESH)
        self.delta_mode = delta_mode
        if not delta_mode:
            self.A_norm_max = 1

        if action_scaling is not None:
            self.action_scaling = action_scaling

        if a_norm_max is not None:
            self.A_norm_max = a_norm_max

        # optimised_pulse_path = 'configs/data_lilmc/data_lilmc.csv'
        self.optimised_pulse_path = None
        self.ideal_pulses = None
        self.channel_label = 'z'
        if optimised_pulse_path is not None:
            self.optimised_pulse_path = optimised_pulse_path
            coeff, tlist = self.get_optimal_pulse()
            self.ideal_pulses = {'labels': [self.channel_label],
                                 'coeff': [coeff],
                                 'tlist': [tlist],
                                 'max_len': len(coeff),
                                 'max_time': tlist[-1]}

        # Set up model
        self.model_params = {}
        if model_params is not None:
            self.model_params["omega_s"] = model_params["omega_s"]
            self.model_params["alpha_s"] = model_params["alpha_s"]
            self.model_params["g"] = model_params["g"]
            self.model_params["alpha_c"] = model_params["alpha_c"]
            self.model_params["omega_r"] = model_params["omega_r"]
            self.model_params["omega_c_0"] = model_params["omega_c_0"]
            self.n_levels = n_levels = model_params["n_levels"]
            coupler_dims = model_params["coupler_dims"]
            num_qubits = model_params["num_qubits"]
        else:
            # self.model_params["omega_s"] = [5.8899, 5.0311]
            # self.model_params["alpha_s"] = [-324e-3, -235e-3]
            # self.model_params["g"] = [100e-3, 71.4e-3]
            # self.model_params["omega_s"] = [5.9, 6.0]
            # self.model_params["alpha_s"] = [-310e-3, -290e-3]
            self.model_params["omega_s"] = [6.0, 5.9]
            self.model_params["alpha_s"] = [-290e-3, -310e-3]
            self.model_params["g"] = [70e-3, 70e-3]
            self.model_params["alpha_c"] = -200e-3
            self.model_params["omega_r"] = 6.2
            self.model_params["omega_c_0"] = 6.7
            self.n_levels = n_levels = 3
            coupler_dims = 3
            num_qubits = 2

        qubit_dims = [n_levels, n_levels]
        full_dims = qubit_dims.copy()
        full_dims.append(coupler_dims)

        self.psi00 = ket((0, 0, 0), dim=full_dims)
        self.psi01 = ket((0, 1, 0), dim=full_dims)
        self.psi10 = ket((1, 0, 0), dim=full_dims)
        self.psi11 = ket((1, 1, 0), dim=full_dims)
        self.basis_states_ket = [self.psi00, self.psi01, self.psi10, self.psi11]

        # self.simulators = [ZCQubits(2, qubit_dims=qubit_dims, coupler_dims=coupler_dims, params=self.model_params) for _ in range(len(self.basis_states_ket))]
        self.simulator = ZCQubits(num_qubits, qubit_dims=qubit_dims, coupler_dims=coupler_dims, **self.model_params)
        self.all_model_params = self.model_params.copy()
        self.all_model_params.update({'n_levels': n_levels,
                                      'coupler_dims': coupler_dims,
                                      'num_qubits': num_qubits})

        self.T = T  # ns
        self.pulse_length = pulse_length
        self.tlist = np.linspace(0, T, pulse_length)
        self.dt = T/(self.pulse_length - 1) # dt obtained after np.linspace(0, T, pulse_length)
        assert self.tlist[1] == self.dt
        self.pulse_amplitudes_denorm = self.init_pulse_amplitudes()

        self.unitary_iswap = U = self.get_iswap_u()

        # Set up state vectors

        self.basis_states = basis_states = self.basis_states_ket.copy()
        self.basis_states_str = ('$|000\\rangle$', '$|010\\rangle$','$|100\\rangle$','$|110\\rangle$',)
        self.initial_states = self.basis_states.copy()
        # self.solvers = [SESolver(self.simulator.H) for _ in self.basis_states]
        self.solvers = [SESolver(self.simulator.H) for i in range(len(self.basis_states))]

        mapped_basis_states = [sum(U[i, j] * basis_states[i]
                                   for i in range(U.shape[0])) for j in range(U.shape[1])]
        # Lots of gates just rearrange the basis states, and we can avoid some complexity by identifying
        # this and setting the mapped_basis_states to the identical objects as the original basis_states
        for i, state in enumerate(mapped_basis_states):
            for j, basis_state in enumerate(basis_states):
                if state == basis_state:
                    mapped_basis_states[i] = basis_state
        self.target_states = mapped_basis_states.copy()

        # Set up action space
        self.n_channels = len(self.action_scaling)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.n_channels,), dtype=np.float32)
        self.amps_cur = np.zeros(self.n_channels)
        self.n_act = self.n_channels

        # Set up observation space
        # self.n_obs = len(self.basis_states) * 2 * reduce(lambda x, y: x*y, full_dims) + self.n_act + 1  # +1 is the time dimension
        self.n_obs = 26
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.n_obs,), dtype=np.float32)

        # All initialised after first call to reset method
        self.cur_idx = 0
        self.final_states_all = None
        self.pulse_amplitudes_denorm = None
        self.amps_cur = None
        self.step_states = None
        self.current_state = None
        self.rewards = None
        self.fidelities = None
        self.actions_all = None

    @staticmethod
    def get_iswap_u():
        return Qobj(np.array([
            [1, 0, 0, 0],
            [0, 0, 1j, 0],
            [0, 1j, 0, 0],
            [0, 0, 0, 1]
        ]), dims=[[2, 2], [2, 2]])  # iSWAP

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

    def init_pulse_amplitudes(self):
        # return np.zeros((self.n_channels, self.pulse_length))  # First row for u01, second row for d1
        # buffer = {}
        # for channel_label in self.channel_labels:
        #     buffer[channel_label] = np.zeros(self.pulse_length)  # +1 since first amp must be zero
        # return buffer
        return np.zeros(self.pulse_length)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        if seed is not None:
            np.random.seed(seed)
        self.cur_idx = 0
        self.final_states_all = [self.initial_states.copy()]
        self.pulse_amplitudes_denorm = self.init_pulse_amplitudes()
        self.amps_cur = 0.
        self.step_states = self.initial_states.copy()
        # self.current_state = self.extract_current_state(self.step_states, idx=0).astype(np.float32)
        self.current_state = self.extract_current_cardinal_state(self.step_states).astype(np.float32)
        self.actions_all = []
        self.rewards = []
        self.fidelities = []

        for solver, initial_state in zip(self.solvers, self.initial_states):
            solver.start(initial_state, 0.)

        # for k, solver in enumerate(self.solvers):
        #     self.solvers[k].start(from_states[k], 0.)
        # self.fidelities = []
        return self.current_state, {}

    def get_current_amp_time_norm_tupe(self, action=None):
        obs_ampt = []
        # Add the action
        if action is None:
            obs_ampt.append(np.array([self.amps_cur / self.A_norm_max]).reshape(-1, 1))
        else:
            obs_ampt.append(np.array(action).reshape(-1, 1))
        # Add the time of current step
        obs_ampt.append(np.array(self.cur_idx / self.pulse_length).reshape(-1, 1))

        return np.concatenate(obs_ampt)

    def extract_current_state(self, states, action=None, idx=None, state_only=False):
        obs = []
        for state in states:
            if isinstance(state, Qobj):
                state = state.full()
            obs.append(np.concatenate([state.real.copy(), state.imag.copy()]))
        if not state_only:
            obs.extend(self.get_current_amp_time_norm_tupe(action))
        obs = np.concatenate(obs).squeeze()
        return obs.astype(np.float32)

    def extract_current_cardinal_state(self, states, action=None, idx=None, state_only=False):
        obs = []
        for i, state in enumerate(states):
            if i == 0:
                continue

            if isinstance(state, Qobj):
                state = state.full()

            if i < 3:
                for idx in (1, 3, 9):
                    s_ = state[idx]
                    obs.append(np.concatenate([s_.real.copy(), s_.imag.copy()]))
            elif i == 3:
                for idx in (2, 4, 6, 10, 12, 18):
                    s_ = state[idx]
                    obs.append(np.concatenate([s_.real.copy(), s_.imag.copy()]))

        if not state_only:
            obs.extend(self.get_current_amp_time_norm_tupe(action))

        obs = np.concatenate(obs).squeeze()
        return obs.astype(np.float32)

    def norm_action(self, action):
        SCALE = self.action_scaling['z']
        return action / SCALE

    def denorm_action(self, action):
        SCALE = self.action_scaling['z']
        return np.array(action) * SCALE

    def step(self, action, can_term=False):
        """
        Add a pulse amplitude delta vector (action) on the previous value of the pulse amplitude.
        :param action: Must be list-like with `self.n_abs` dimensions
        :return: observation_tp1, reward, terminated, truncated, info
        """
        # Action linear scaling wrt. episode time
        action = np.array(action)

        # Absolute amplitude conversion required for rendering
        self.actions_all.append(action[0])

        # Environment dynamics
        # amp_delta_denorm = self.denorm_action(action.copy())[0]
        from_states = self.step_states.copy()
        crash = False
        oob_pulse = False
        try:
            self.step_states, oob_pulse = self.forward_dynamics(amp_delta_norm=action)
        except Exception as e:
            print(e)
            crash = True

        self.final_states_all.append(self.step_states.copy())

        # terminated is only True when reward threshold is exceeded
        # truncated is only True when maximum pulse length is reached
        terminated, truncated = False, False

        # Check if episode is done
        if self.cur_idx >= self.pulse_length - 1:
            terminated = True

        # Calculate reward for current action
        reward = self.reward_function(self.step_states) - self.REW_THRESH
        if crash:
            reward = -10
            truncated = True

        if oob_pulse:
            reward = -10

        self.rewards.append(reward)

        if reward > 0 and can_term:
            terminated = True

        # Construct observation for agent using state probabilities and normed actions
        # self.current_state = self.extract_current_state(self.step_states, self.cur_idx)
        self.current_state = self.extract_current_cardinal_state(self.step_states)

        self.cur_idx += 1
        return self.current_state, reward, terminated, truncated, {}

    def forward_dynamics(self, amp_delta_norm):
        oob_pulse = False  # Out of bounds absolute pulse

        # Delta formalism
        if self.delta_mode:
            if self.cur_idx == 0:
                abs_action_denorm = 0.
            else:
                abs_action_denorm = self.pulse_amplitudes_denorm[self.cur_idx - 1]

            abs_action_denorm += self.denorm_action(amp_delta_norm[0])
            abs_action_norm = self.norm_action(abs_action_denorm)
            if abs(abs_action_norm) > self.A_norm_max:
                oob_pulse = True
                abs_action_denorm = self.denorm_action(np.sign(abs_action_norm) * self.A_norm_max)
        # Absolute formalism
        else:
            abs_action_denorm = self.denorm_action(amp_delta_norm * self.A_norm_max)

        # Save amplitude to pulse
        self.pulse_amplitudes_denorm[self.cur_idx] = abs_action_denorm

        # Step through the simulator with new amplitude for dt time
        self.amps_cur = abs_action_denorm
        final_states = []
        for k, solver in enumerate(self.solvers):
            t = self.tlist[self.cur_idx]
            s = self.solvers[k].step(t, args={'A': self.amps_cur})
            final_states.append(s.copy())

        return final_states, oob_pulse

    def reward_function(self, step_states):
        f = np.mean([fidelity(s, t) for s, t in zip(step_states, self.target_states)])
        self.fidelities.append(f)

        rew = self.fid2rew(f)   # - self.REW_THRESH
        return rew

    @staticmethod
    def fid2rew(fid):
        return -np.log10(1 - fid)

    @staticmethod
    def rew2fid(rew):
        rew = float(rew)
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
        # fidelities = (np.clip([self.rew2fid(r) for r in self.rewards], a_min=1e-4, a_max=1)) * 100.
        fidelities = np.multiply(self.fidelities, 100.)
        infidelities = [100 - item for item in fidelities]
        # recons_amps, global_tlist = self.get_reconstructed_pulses_with_uniform_time()

        fig, ax = plt.subplots()
        ax.plot(self.tlist, infidelities, color='k')
        ax.axhline((1 - self.FID_THRESH) * 100., color='g', ls='--')
        ax.set_title(f'Fidelity evolution  T={self.T}')
        ax.set_ylabel('Fidelity (%)')
        ax.set_ylabel('Step')
        fig.savefig(os.path.splitext(save_path)[0] + '_fidelity.pdf')
        ax.set_yscale('log')
        fig.savefig(os.path.splitext(save_path)[0] + '_fidelity_logscale.pdf')



        # Setup figure and axes
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle(f'iSWAP\naction scale={self.action_scaling[self.channel_label]:.2e}')
        if self.delta_mode:
            n_rows = 4
        else:
            n_rows = 3
        gs = mpl.gridspec.GridSpec(n_rows, 2)

        ax_state_titles = []
        for i in range(self.n_levels):
            for j in range(self.n_levels):
                ax_state_titles.append(f'$|{i}{j}\\rangle$')

        ax_state = fig.add_subplot(gs[0, :])
        ax_action = fig.add_subplot(gs[1, :])
        if self.delta_mode:
            ax_action_deltas = fig.add_subplot(gs[2, :])
        ax_reward = fig.add_subplot(gs[n_rows - 1, :])

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
        ax.set_xlim(0, self.T)
        K_delta = self.action_scaling['z']
        K =  K_delta * self.A_norm_max
        ax.set_ylim(-K, K)  # Adjust based on action range
        ax.set_ylabel('Pulse amplitude')  # Adjust based on action range
        ax.set_xlabel('Time [ns]')  # Adjust based on action range

        if self.delta_mode:
            ax = ax_action_deltas
            ax.axhline(y=0, linestyle='dashed', color='gray')
            ax.set_xlim(0, self.T)

            ax.set_ylim(-K_delta*1.1, K_delta*1.1)  # Adjust based on action range
            ax.set_ylabel('Pulse deltas')  # Adjust based on action range
            ax.set_xlabel('Time [ns]')  # Adjust based on action range

        nb_orders = lambda x: 10 ** int(np.log10(x))
        ax = ax_reward
        infid_thresh = (1. - self.FID_THRESH) * 100.
        rew_lim_min = min(nb_orders(min(infidelities)), nb_orders(infid_thresh)) / 10.
        rew_lim_max = nb_orders(max(infidelities)) * 10.
        ax.axhline(infid_thresh, ls='dashed', c='g', label=f'Threshold: {infid_thresh:.2f}%')
        ax.set_ylim(rew_lim_min, rew_lim_max)  # Adjust based on expected reward range
        ax.legend(loc='upper right')

        ax.set_yscale('log')
        ax.set_ylabel('Infidelity (%)')
        ax.set_xlabel('Time [ns]')

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
        # ax_action_ideal = ax_action.twinx()
        # ax_action_ideal.set_ylabel('Pulse amplitide')
        # ax_action_ideal.plot(self.tlist, recons_amps[0], ls='dashed', alpha=0.7, marker='x', label=f'Ideal {self.channel_label}')
        # ax_action_ideal.legend(loc='best')

        ax_reward.set_xlim(0, self.T)

        action_line, = ax_action.plot([], [], lw=1.2, c='tab:orange', label=self.channel_label, marker='.')
        ax_action.legend(loc='upper right', ncol=2)

        ax_reward.set_title('Time:  Fidelity:')
        ax_reward.grid(which='major', linestyle='--', color='grey', linewidth=1)
        ax_reward.grid(which='minor', linestyle=':', color='lightgrey', linewidth=0.5)
        reward_line, = ax_reward.plot([], [], 'g-', lw=2, marker='.')

        action_delta_line = None
        if self.delta_mode:
            action_delta_line, = ax_action_deltas.plot([], [], lw=1.2, label=f'Δ{self.channel_label}', marker='.')
            ax_action_deltas.legend(loc='upper right')
        fig.tight_layout()

        def init():
            s = np.abs(np.concatenate([s.full() for s in states[0]])).reshape(4, -1)
            s = np.flipud(s)
            im = ax_state.imshow(s, animated=True, cmap=self.cmap, norm=norm, origin='upper')

            action_line.set_data([], [])
            reward_line.set_data([], [])
            if self.delta_mode:
                action_delta_line.set_data([], [])

            for i, srow in enumerate(s):
                for j, selem in enumerate(srow):
                    texts[i*len(srow) + j].set_text(f'{selem:.2f}')

            return im, action_line, action_delta_line, reward_line, texts

        def update(frame):
            s = np.clip(np.abs(np.concatenate([s_.full() for s_ in states[frame]])), EPS, 1).reshape(4, -1)
            s = np.flipud(s)
            im = ax_state.imshow(s, animated=True, cmap=self.cmap, norm=norm, origin='upper')

            action_line.set_data(self.tlist[:frame], self.denorm_action(self.pulse_amplitudes_denorm[:frame]))
            reward_line.set_data(self.tlist[:frame], infidelities[:frame])
            ax_reward.set_title(f'Time: {self.tlist[frame]:.2f}ns  Fidelity: {100 - infidelities[frame]:.2f}%  (Best fidelity: {max(fidelities):.2f}%)')
            if self.delta_mode:
                action_delta_line.set_data(self.tlist[:frame], self.denorm_action(self.actions_all[:frame]))

            for i, srow in enumerate(s):
                for j, selem in enumerate(srow):
                    texts[i*len(srow) + j].set_text(f'{selem:.3f}')

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

        pbar = tqdm(total=len(frames))

        def progress_callback(current_frame: int, total_frames: int):
            nonlocal pbar
            pbar.update(1)
        print(f'Saving to: {save_path}')
        ani.save(save_path, dpi=100, writer=ffwriter, progress_callback=progress_callback)



    @staticmethod
    def evaluate_pulse(pulse_file, save_path, render=True, **env_kwargs):
        assert pulse_file.endswith('.csv')
        assert save_path.endswith('.mp4')

        data = pd.read_csv(pulse_file)
        pulse = data['pulse'].to_numpy()
        tlist = data['tlist'].to_numpy()
        pulse_length = len(tlist)
        env = ZCQPEE(**env_kwargs)
        # print(env.dt)
        # exit(23)
        assert int(tlist[1] * 1e6) == int(env.dt * 1e6)

        acts = []
        rews = []
        fids = []

        # obses.append(env.reset()[0])
        env.reset()

        truncated = False
        terminated = False
        idx = 0
        while not terminated and not truncated:
            if idx == 0:
                action = np.array([pulse[0]])
            else:
                action = np.array([pulse[idx] - pulse[idx - 1]])

            acts.append(pulse[idx])
            action = env.norm_action(action)

            _, rew, terminated, truncated, _ = env.step(action)

            rews.append(rew)
            fids.append(env.rew2fid(rew))
            idx += 1

        if render:
            env.render(save_path=save_path)
        states = np.array(env.final_states_all).T
        return states, np.array(acts), np.array(rews), np.array(fids)

    def __repr__(self):
        return (f"ZCQPEE:   pulse_length = {self.pulse_length}\n"
                f"          T = {self.T} ns \n"
                f"          ΔT = {self.dt:.3f} ns\n"
                f"          Fid. thresh. = {self.FID_THRESH*100.:.2f}%\n"
                f"          action scaling = {self.action_scaling}\n"
                f"          A_norm_max = {self.A_norm_max}\n"
                f"          ISWAP gate optimisation.\n"
                f"          Action delta_mode={self.delta_mode}\n")

    def __str__(self):
        return f"ZCQPEE_pl-{self.pulse_length}_T-{self.T:.1f}ns{'_delta_mode' if self.delta_mode else '_abs_mode'}"




if __name__ == '__main__':
    import os
    # par_dir = '/home/leander/code/rlquantopt/rlquantopt/rl_agents/ZCQPEE120pl-PPO/13-06-24_115241_ZCQPEE120pl/best_model/pulses'
    par_dir = '/home/leander/code/rlquantopt/rlquantopt/rl_envs/configs'
    pulse_file = os.path.join(par_dir, 'data_lilmc.csv')
    save_path = os.path.join(par_dir, 'data_lilmc.mp4')
    ZCQPEE.evaluate_pulse(pulse_file, save_path)


