import os

import setuptools.namespaces
from gymnasium import spaces
import yaml
import numpy as np
from tqdm import tqdm
import pandas as pd
from qutip import Qobj, ket, QobjEvo, SESolver, expect
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.animation import FuncAnimation
from gymnasium import Env
import random

from rlquantopt.rl_envs.zcqubits import ZCQubits, fidelity, setup_ZCQubits4MKrauss_params
# from rlquantopt.tests.zc_qpee_vs_zcqubits.zc_qpee_vs_zcqubits import env_kwargs

mpl.rcParams['font.size'] = 18


class ZCQPEE(Env):
    # Action scaling parameters
    action_scaling = {'z': 1}    # Only applicable in delta_mode==True
    A_norm_max = 5
    FID_THRESH = 0.99

    ACT_POLY_ORDER = 2          # Action transformation polynomial order, i.e. 1 = linear, 2 = quadratic, etc.

    ADD_PREV_OBS = False

    # Rendering parameters
    PREC = 5e-4
    PLOT_LOG_EPS = 1e-4
    cmap = mpl.colormaps["CMRmap"]
    FPS = 10
    DPI = 80

    RANDOM_START_PROB = 0.0
    RANDOM_START_STEPS = 10

    REW_SCALE = 1
    N_TIME_STEPS = 3
    TV_PENALTY_SCALE = 0.1

    def __init__(self, model_params=None, **env_kwargs):
        """

        :param model_params:
        :param env_kwargs:
        """
        self.env_kwargs = env_kwargs
        # Set up random starts
        if env_kwargs.get('start_basis_states', False):
            self.RANDOM_START_PROB = 0.
        self.RANDOM_START_PROB = env_kwargs.get('random_start_prob', self.RANDOM_START_PROB)
        self.RANDOM_START_STEPS = env_kwargs.get('random_start_steps', self.RANDOM_START_STEPS)

        # General parameters
        self.pulse_length = env_kwargs.get('pulse_length', 500)
        self.delta_mode = env_kwargs.get('delta_mode', True)
        self.T = env_kwargs.get('T', 300)

        # Reward rekated parameters
        self.FID_THRESH = env_kwargs.get('fid_thresh', self.FID_THRESH)
        self.REW_SCALE = env_kwargs.get('rew_scale', self.REW_SCALE)
        self.REW_THRESH = self._fid2rew(self.FID_THRESH) * self.REW_SCALE

        self.ACT_POLY_ORDER = env_kwargs.get('act_poly_order', self.ACT_POLY_ORDER)
        self.ADD_PREV_OBS = env_kwargs.get('add_prev_obs', self.ADD_PREV_OBS)
        self.TV_PENALTY_SCALE = env_kwargs.get('tv_penalty_scale', self.TV_PENALTY_SCALE)

        # Action related parameters
        self.N_TIME_STEPS = env_kwargs.get('n_time_steps', self.N_TIME_STEPS)
        self.action_scaling = env_kwargs.get('action_scaling', self.action_scaling)
        a_norm_max = env_kwargs.get('a_norm_max', None)
        if a_norm_max is not None and self.delta_mode:
            self.A_norm_max = a_norm_max
        elif not self.delta_mode:
            self.A_norm_max = 1

        # Set up model
        if model_params is None:
            model_params = dict()
        self.model_params = setup_ZCQubits4MKrauss_params(**model_params)
        self.n_levels = self.model_params.get("n_levels")
        self.channel_label = 'z'
        self.simulator = ZCQubits(**self.model_params)

        # Set up time axis
        self.tlist = np.linspace(0, self.T, self.pulse_length + 1)
        self.dt = self.T / self.pulse_length
        if self.tlist[1] != self.dt:
            raise ValueError("tlist diff and dt must be the same")

        # Set up gate
        self.unitary_iswap = U = self.get_iswap_u()

        # Set up initial states
        full_dims = self.model_params.get("qubit_dims").copy()
        full_dims.append(self.model_params.get("coupler_dims"))
        psi00 = ket((0, 0, 0), dim=full_dims)
        psi01 = ket((0, 1, 0), dim=full_dims)
        psi10 = ket((1, 0, 0), dim=full_dims)
        psi11 = ket((1, 1, 0), dim=full_dims)
        basis_states_ket = [psi00, psi01, psi10, psi11]
        self.basis_states = basis_states_ket.copy()
        self.initial_states = self.basis_states.copy()

        # Set up SE solvers
        self.solvers = [SESolver(self.simulator.H) for _ in range(len(self.basis_states))]

        self.basis_states_str = ('$|000\\rangle$', '$|010\\rangle$','$|100\\rangle$','$|110\\rangle$',)

        # Set up target states
        mapped_basis_states = [sum(U[i, j] * self.basis_states[i]
                                   for i in range(U.shape[0])) for j in range(U.shape[1])]
        # Lots of gates just rearrange the basis states, and we can avoid some complexity by identifying this and setting the mapped_basis_states to the identical objects as the original basis_states
        for i, state in enumerate(mapped_basis_states):
            for j, basis_state in enumerate(self.basis_states):
                if state == basis_state:
                    mapped_basis_states[i] = basis_state
        self.target_states = mapped_basis_states.copy()
        self.e_ops = [s_.proj() for s_ in self.target_states]

        # Set up action space
        self.n_channels = len(self.action_scaling)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.n_channels, self.N_TIME_STEPS), dtype=np.float32)
        # self.amps_cur = np.zeros(self.n_channels)
        self.n_act = self.n_channels * self.N_TIME_STEPS

        # Set up observation space
        # self.n_obs = 25 + self.N_TIME_STEPS
        if self.ADD_PREV_OBS:
            self.n_obs = (24 * 2) + 1 + self.N_TIME_STEPS
        else:
            self.n_obs = 25 + self.N_TIME_STEPS

        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.n_obs,), dtype=np.float32)

        # All initialised after first call to reset method
        self.cur_idx: int = 0
        self.final_states_all: list = []
        self.pulse_amplitudes_denorm: np.ndarray = np.array([])
        self.pulse_deltas_denorm: np.ndarray = np.array([])
        self.amps_cur: np.ndarray = np.array([])
        self.step_states: list = []
        self.prev_step_states: list = []
        self.current_obs: np.ndarray = np.array([])
        self.rewards: list = []
        self.fidelities: list = []

    @staticmethod
    def get_iswap_u():
        return Qobj(np.array([
            [1, 0, 0, 0],
            [0, 0, 1j, 0],
            [0, 1j, 0, 0],
            [0, 0, 0, 1]
        ]), dims=[[2, 2], [2, 2]])  # iSWAP

    def init_pulse_amplitudes(self, delta):
        if delta:
            return np.zeros(self.pulse_length)
        else:
            return np.zeros(self.pulse_length + 1)

    def reset(self, init_state=None, seed=None, random_starts=False, options=None):
        super().reset(seed=seed, options=options)
        if seed is not None:
            np.random.seed(seed)
        self.cur_idx = 0
        self.final_states_all = [self.initial_states.copy()]
        self.pulse_amplitudes_denorm = self.init_pulse_amplitudes(delta=False)
        if self.delta_mode:
            self.pulse_deltas_denorm = self.init_pulse_amplitudes(delta=True)
        self.amps_cur = np.zeros(self.N_TIME_STEPS)
        if init_state is not None:
            self.prev_step_states = init_state.copy()
            self.step_states = init_state.copy()
        else:
            self.prev_step_states = self.initial_states.copy()
            self.step_states = self.initial_states.copy()

        if self.ADD_PREV_OBS:
            self.current_obs = np.concatenate([self.extract_current_cardinal_state(self.prev_step_states, state_only=True), self.extract_current_cardinal_state(self.step_states).astype(np.float32)])
        else:
            self.current_obs = self.extract_current_cardinal_state(self.step_states).astype(np.float32)

        self.rewards = []
        self.fidelities = []

        for k, initial_state in enumerate(self.initial_states):
            self.solvers[k].start(initial_state, 0.)

        if np.random.random() < self.RANDOM_START_PROB and random_starts:
            info = None
            for _ in range(self.RANDOM_START_STEPS):
                *_, info = self.step(self.action_space.sample())
            self.reset(init_state=info['step_states'], random_starts=False, seed=seed, options=options)

        return self.current_obs, {}

    def get_current_amp_time_norm_tuple(self, action=None, include_time=True):
        obs_ampt = []
        # Add the action
        if action is None:
            obs_ampt.append(np.array([self.amps_cur / self.A_norm_max]).reshape(-1, 1))
        else:
            obs_ampt.append(np.array(action).reshape(-1, 1))
        if include_time:
            # Add the time of current step
            obs_ampt.append(np.array(self.cur_idx * 2 / self.pulse_length - 1).reshape(-1, 1))

        return np.concatenate(obs_ampt)

    def extract_current_cardinal_state(self, states, action=None, state_only=False, include_time=True):
        obs = []
        for i, state in enumerate(states):
            if i == 0:
                continue

            if isinstance(state, Qobj):
                state = state.full()

            # Cartesian
            # extract_complex = lambda z: np.concatenate([s_.real.copy(), s_.imag.copy()])
            # Polar form
            def extract_complex(z):
                return np.concatenate([(np.absolute(z) * 2) - 1, np.angle(z)/np.pi])
            # extract_complex = lambda z: np.concatenate([(np.absolute(z) * 2) - 1, np.angle(z)/np.pi])
            if i < 3:       # for states |000>, |010>, |100>
                for idx in (1, 3, 9):   # Extract only these elements from the state vector @ i < 3
                    s_ = state[idx]
                    obs.append(extract_complex(s_))
            elif i == 3:    # for state |110>
                for idx in (2, 4, 6, 10, 12, 18):   # Extract only these elements from the state vector @ i == 3
                    s_ = state[idx]
                    obs.append(extract_complex(s_))

        if not state_only:
            obs.extend(self.get_current_amp_time_norm_tuple(action, include_time=include_time))

        obs = np.concatenate(obs).squeeze()
        return obs.astype(np.float32)

    def norm_action(self, action):
        SCALE = self.action_scaling['z']
        return action / SCALE

    def denorm_action(self, action):
        SCALE = self.action_scaling['z']
        return np.array(action) * SCALE

    def transform_action(self, action):
        signs = np.sign(action)
        # action = np.clip(np.abs(action), 0, 1 - np.exp(-5))
        # action = -0.2 * np.log(np.ones_like(action) - action)
        action = np.power(np.abs(action), self.ACT_POLY_ORDER)
        return signs * action

    def step(self, action, can_early_term=False):
        """
        Add a pulse amplitude delta vector (action) on the previous value of the pulse amplitude.
        :param action: Must be list-like with `self.n_abs` dimensions
        :param can_early_term: If True, episode will terminate iff reward threshold is exceeded. Used during evaluation to get a shorter pulse ...
        :return: observation_tp1, reward, terminated, truncated, info
        """
        action = self.transform_action(action)  # More expressive around the 0 region
        action = np.array(action).reshape(self.n_channels, self.N_TIME_STEPS)
        action_denorm = self.denorm_action(action)

        N = self.N_TIME_STEPS
        if self.delta_mode:
            self.pulse_deltas_denorm[self.cur_idx:self.cur_idx+N] = action_denorm

        # Prepare action and append to memory
        amp_abs_denorm, oob_pulse = self.preprocess_action(action_denorm)
        self.pulse_amplitudes_denorm[self.cur_idx+1:self.cur_idx + N+1] = amp_abs_denorm

        # Environment dynamics
        crash = False
        try:
            self.step_states = self.forward_dynamics(amp_abs_denorm)
        except Exception as e:
            print(e)
            crash = True

        terminated, truncated, terminal = False, False, False

        # Calculate reward for current action
        reward, tv_penalty, fid = self._reward_function(self.step_states, self.amps_cur)
        reward *= self.REW_SCALE
        reward -= self.REW_THRESH   # so that any fidelity below the threshold obtains a negative reward, and positive otherwise

        # Assign penalty if pulse crashed the simulator & send trancation signal to stop the episode
        if crash:
            reward = -50
            truncated = True
        # Assign smaller penalty if the pulse os OOB
        elif oob_pulse:
            reward = -30 * (1 - (self.cur_idx / self.pulse_length))
            truncated = True

        if reward > 0 and can_early_term:
            terminated = True

        # Collect for rendering
        self.final_states_all.append(self.step_states.copy())
        self.rewards.append(reward)
        self.fidelities.append(fid)

        # Construct observation for agent using state probabilities and normed actions
        if self.ADD_PREV_OBS:
            self.current_obs = np.concatenate([self.extract_current_cardinal_state(self.prev_step_states, state_only=True), self.extract_current_cardinal_state(self.step_states, include_time=True)])
        else:
            self.current_obs = self.extract_current_cardinal_state(self.step_states, include_time=True)

        self.prev_step_states = self.step_states

        self.cur_idx += N

        # Check if episode is done
        if self.cur_idx + N - 1 >= self.pulse_length:
            terminated = True
            terminal = True

        return self.current_obs, reward, terminated, truncated, {'tv_penalty': tv_penalty, 'fidelity': fid, 'step_states': self.step_states, 'oob_pulse': oob_pulse, 'crash': crash, 'terminal_state': terminal}

    def preprocess_action(self, action_denorm):
        oob_pulse = False
        # Delta formalism
        if self.delta_mode:
            if self.cur_idx == 0:
                # First time-step assumes a prev. amplitude of 0 - start from zero-pulse
                prev_amp_abs_denorm = 0.
            else:
                # In delta formalism, we need the previous amplitude
                prev_amp_abs_denorm = self.amps_cur[-1]

            amp_abs_denorm = prev_amp_abs_denorm + np.cumsum(action_denorm)  # add the delta amp.
            # amp_abs_norm = amp_abs_denorm / self.A_norm_max # normalise absolute pulse wrt A_norm_max
            amp_abs_norm = self.norm_action(amp_abs_denorm)
            if max(abs(amp_abs_norm)) > self.A_norm_max:
                oob_pulse = True
                amp_abs_norm = np.where(abs(amp_abs_norm) > self.A_norm_max, np.sign(amp_abs_norm) * self.A_norm_max, amp_abs_norm)
                amp_abs_denorm = self.denorm_action(amp_abs_norm)
        # Absolute formalism
        else:
            # Treat the action as an absolute amplitude value
            amp_abs_denorm = action_denorm

        return amp_abs_denorm, oob_pulse

    def forward_dynamics(self, amp_abs_denorm):
        """
        Requires self.cur_idx==0 during the first step.
        This is to calculate the pulse time correctly.
        """

        # Step through the simulator with new amplitude until next time-step in simulation
        self.amps_cur = amp_abs_denorm
        final_states = []
        for k, _ in enumerate(self.step_states):
            for i in range(self.N_TIME_STEPS):
                t = self.tlist[self.cur_idx + i + 1]    # The +1 will skip the first time sample @ 0ns
                final_state = self.solvers[k].step(t, args=dict(A=amp_abs_denorm[i]))
            final_states.append(final_state)

        return final_states

    def _compute_total_variation(self, acts):
        return np.sum(np.abs(np.diff(acts)))

    def _reward_function(self, step_states, acts=None):
        f = np.mean([fidelity(s, t) for s, t in zip(step_states, self.target_states)])
        rew = self._fid2rew(f)
        if f >= self.FID_THRESH:    # remove tv penalty upon successful pulse
            return rew, 0.0, f

        if acts is None:
            return rew, 0.0, f

        total_variation = self._compute_total_variation(acts)
        tv_penalty = total_variation * self.TV_PENALTY_SCALE
        rew -= tv_penalty

        return rew, tv_penalty, f

    @staticmethod
    def _fid2rew(fid):
        return -np.log10(1 - fid)

    def to_yaml(self, save_path=None):
        data = dict(env_kwargs=self.env_kwargs,
                    model_params=self.model_params)

        if save_path is not None:
            with open(save_path, 'w') as f:
                yaml.dump(data, f)

        return data

    @classmethod
    def from_yaml(cls, load_path):
        with open(load_path, 'r') as f:
            kw = yaml.load(f, yaml.SafeLoader)
        self = cls(model_params=kw['model_params'], **kw['env_kwargs'])
        return self

    @staticmethod
    def find_yaml_in_dir(dir):
        for item in os.listdir(dir):
            if (item.endswith('.yaml') or item.endswith('.yml')) and item.startswith('ZCQPEE_pl-'):
                return os.path.join(dir, item)

    def render(self, mode='human', save_path=None, fps=80, **kwargs):
        mpl.rcParams['font.size'] = 10
        states = self.final_states_all
        fidelities = np.multiply(self.fidelities, 100.)
        infidelities = [100 - item for item in fidelities]

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
        ax_lim_border = np.ptp(self.pulse_amplitudes_denorm) * 0.1
        ax.set_ylim(min(self.pulse_amplitudes_denorm) - ax_lim_border, max(self.pulse_amplitudes_denorm) + ax_lim_border)  # Adjust based on action range
        ax.set_ylabel('Pulse amplitude')  # Adjust based on action range
        ax.set_xlabel('Time [ns]')  # Adjust based on action range

        # nb_orders = lambda x: 10 ** int(np.log10(x))
        def nb_orders(x):
            return 10 ** int(np.log10(x))

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

        # get_state_mat = lambda idx: np.square(np.abs(np.flipud(np.concatenate([s_.full() for s_ in states[idx]]).reshape(4, -1))))  # Transform state vector to probabilities
        def get_state_mat(idx):
            return np.square(np.abs(np.flipud(np.concatenate([s_.full() for s_ in states[idx]]).reshape(4, -1))))

        state_mat = get_state_mat(0)
        im = ax_state.imshow(state_mat, animated=True, cmap=self.cmap, norm=norm, origin='upper')
        texts = []
        ax = ax_state
        for i, srow in enumerate(state_mat):
            for j, selem in enumerate(srow):
                text = ax.text(j, i, f'{selem:.2f}', horizontalalignment='center', verticalalignment='center', fontsize=9, c='k')
                texts.append(text)
        fig.colorbar(im, ax=ax_state)

        ax_reward.set_xlim(0, self.T)

        action_line, = ax_action.plot([], [], lw=1, c='tab:orange', label=self.channel_label)
        ax_action.legend(loc='upper right', ncol=2)

        ax_reward.set_title('Time:  Fidelity:')
        ax_reward.grid(which='major', linestyle='--', color='grey', linewidth=1)
        ax_reward.grid(which='minor', linestyle=':', color='lightgrey', linewidth=0.5)
        reward_line, = ax_reward.plot([], [], 'g', lw=1.2)

        # Action delta plot if self.delta_mode==True
        if self.delta_mode:
            k_delta = self.action_scaling['z']
            ax_action_deltas = fig.add_subplot(gs[2, :])
            ax = ax_action_deltas
            ax.axhline(y=0, linestyle='dashed', color='gray')
            ax.set_xlim(0, self.T)

            ax.set_ylim(-k_delta * 1.1, k_delta * 1.1)  # Adjust based on action range
            ax.set_ylabel('Pulse deltas')  # Adjust based on action range
            ax.set_xlabel('Time [ns]')  # Adjust based on action range
            action_delta_line, = ax_action_deltas.plot([], [], lw=1, label=f'Δ{self.channel_label}')
            ax_action_deltas.legend(loc='upper right')

        fig.tight_layout()

        def init():
            s = get_state_mat(0)
            im_ = ax_state.imshow(s, animated=True, cmap=self.cmap, norm=norm, origin='upper')

            action_line.set_data([], [])
            reward_line.set_data([], [])
            if self.delta_mode:
                action_delta_line.set_data([], [])

            for i, srow in enumerate(s):
                for j, selem in enumerate(srow):
                    texts[i * len(srow) + j].set_text(f'{selem:.2f}')

            return im_, action_line, action_delta_line, reward_line, texts

        def update(frame):
            state_mat = get_state_mat(frame)
            state_mat = np.clip(state_mat, EPS, 1)
            im_ = ax_state.imshow(state_mat, animated=True, cmap=self.cmap, norm=norm, origin='upper')

            idx_ = frame * self.N_TIME_STEPS
            action_line.set_data(self.tlist[:idx_], self.denorm_action(self.pulse_amplitudes_denorm[:idx_]))
            reward_line.set_data(self.tlist[0:idx_:self.N_TIME_STEPS], infidelities[:frame])
            ax_reward.set_title(f'Time: {self.tlist[idx_]:.2f}ns  Fidelity: {100 - infidelities[frame - 1]:.2f}%  (Best fidelity: {max(fidelities):.2f}%)')
            if self.delta_mode:
                action_delta_line.set_data(self.tlist[:idx_], self.pulse_deltas_denorm[:idx_])

            for i, srow in enumerate(state_mat):
                for j, selem in enumerate(srow):
                    texts[i*len(srow) + j].set_text(f'{selem:.3f}')

            return im_, action_line, action_delta_line, reward_line, texts

        # frames = list(np.arange(0, self.cur_idx, max(1, int(self.cur_idx/120)))) + [self.cur_idx-1]*3
        max_render_steps = len(fidelities)
        frames = np.arange(max_render_steps).tolist() + [max_render_steps - 1] * 2
        ani = FuncAnimation(fig, update, frames=frames, interval=5, init_func=init, blit=False)

        if save_path is None:
            print(f'Returning animation')
            return ani

        print(f'Saving to: {save_path}')

        pbar = tqdm(total=len(frames))
        def progress_callback(current_frame: int, total_frames: int):
            nonlocal pbar
            pbar.update(1)
        print(f'Saving to: {save_path}')

        ffwriter = mpl.animation.FFMpegWriter(fps=self.FPS)
        ani.save(save_path, dpi=self.DPI, writer=ffwriter, progress_callback=progress_callback)

        pbar.close()
        plt.close()
        return

    @staticmethod
    def evaluate_pulse(pulse_file, save_path, render=True, **env_kwargs):
        assert pulse_file.endswith('.csv')
        # assert save_path.endswith('.mp4')

        data = pd.read_csv(pulse_file)
        pulse = data['pulse'].to_numpy()
        pulse_diff = np.insert(np.diff(pulse), 0, 0.)
        # pulse_diff = np.diff(pulse)
        tlist = data['time'].to_numpy()

        env_kwargs['pulse_length'] = len(pulse)
        env_kwargs['T'] = tlist[-1]
        N = env_kwargs['n_time_steps']

        # N = env_kwargs.get('n_time_steps')
        env = ZCQPEE(**env_kwargs)

        dt = env.dt
        assert np.round(tlist[1] * 1e4) == np.round(env.dt * 1e4)

        acts = []
        rews = []

        # obses.append(env.reset()[0])
        env.reset()

        truncated = False
        terminated = False
        idx = 0
        while not terminated and not truncated:
            action = pulse_diff[idx:idx+N]
            # if idx == 0:
            #     action = np.array([pulse_diff[:N]])
            # else:
            #     action = np.array([pulse[idx] - pulse[idx - 1]])

            acts.append(action.copy())
            action = env.norm_action(action)

            _, rew, terminated, truncated, _ = env.step(action, can_early_term=False)

            rews.append(rew)
            idx += N

        if render:
            env.render(save_path=save_path)
        states = np.array(env.final_states_all).T
        return states, np.array(acts), np.array(rews), env.fidelities

    def __repr__(self):
        return (f"ZCQPEE:   Pulse length = {self.pulse_length}\n"
                f"          T = {self.T} ns \n"
                f"          ΔT = {self.dt:.3f} ns\n"
                f"          Fidelity threshold = {self.FID_THRESH*100.:.2f}%\n"
                f"          Reward scale = {self.REW_SCALE}\n"
                f"          TV penalty scale = {self.TV_PENALTY_SCALE}\n"
                f"          Action scaling = {self.action_scaling}\n"
                f"          A_norm_max = {self.A_norm_max}\n"
                f"          ISWAP gate optimisation.\n"
                f"          Action delta_mode={self.delta_mode}\n"
                f"          Action polynomial order={self.ACT_POLY_ORDER}")

    def __str__(self):
        return f"ZCQPEE_pl-{self.pulse_length}_T-{int(self.T)}ns{'_delta_mode' if self.delta_mode else '_abs_mode'}"

    def model_str(self):
        p = self.model_params
        return f"ω_s={p['omega_s']}, α_s={p['alpha_s']}, g={p['g']}, α_c={p['alpha_c']}, ω_r={p['omega_r']}, ω_c0={p['omega_c_0']} GHz"


def test_actions(axs, pl, N, actions, title=None, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    env = ZCQPEE(pulse_length=pl, delta_mode=True, T=100, fid_thresh=0.995, n_time_steps=N)
    env.reset()
    rews = []
    tv_penalties = []
    for a in actions:
        _, r, _, _, info = env.step(a)
        tv_penalty = info['tv_penalty']
        tv_penalties.append(tv_penalty)
        rews.append(r)

    ax = axs[0]
    ax.set_title(title)
    short_tlist = np.linspace(0, env.T, len(rews))
    ax.plot(short_tlist, rews, c='g', marker='.')
    ax.set_ylabel('Rew.')

    ax = axs[1]
    ax.plot(short_tlist, tv_penalties, c='tab:orange', marker='.')
    ax.set_ylabel('TV pen.')

    ax = axs[2]
    ax.plot(env.tlist, env.pulse_amplitudes_denorm, c='r', marker='.')
    ax.set_ylabel('Amp.')

    for ax in axs:
        ax.set_xlabel('Time [ns]')
    # fig.tight_layout()
    # plt.show()


def test_tv_penalty():
    # import numpy as np
    PL = 4000

    fig, axs = plt.subplots(9, 4, figsize=(35, 20))
    col_idx = 0
    for N in (5, 10):
        row_idx = 0
        for scale in (1, 0.1, 1e-2):
            const_batch_actions = np.repeat(np.random.uniform(-1, 1, PL//N).reshape(-1, 1), N, axis=1) * scale

            axs_ = axs[row_idx: row_idx+3, col_idx]
            test_actions(axs_, pl=PL, N=N, actions=const_batch_actions, title=f'Constant batch N={N} scale={scale}')
            row_idx += 3
        col_idx += 1

    for N in (5, 10):
        row_idx = 0
        for scale in (1, 0.1, 1e-2):
            actions = np.random.uniform(-1, 1, (PL//N, N)) * scale

            axs_ = axs[row_idx: row_idx + 3, col_idx]
            test_actions(axs_, pl=PL, N=N, actions=actions,
                         title=f'Random batch N={N} scale={scale}')
            row_idx += 3
        col_idx += 1

    fig.tight_layout()

    plt.show()


def test_mkrauss_pulse():
    pulse_path = '/home/leander/code/rlquantopt/rlquantopt/rl_envs/configs/data_mkrauss/good_guess_analytical_correct.csv'
    data = pd.read_csv(pulse_path)
    pulse = data['pulse'].to_numpy()
    pulse_diff = np.insert(np.diff(pulse), 0, 0.)
    # pulse_diff = np.diff(pulse)
    tlist = data['time'].to_numpy()
    pulse_length = len(pulse)

    obses, acts, rews, fids = ZCQPEE.evaluate_pulse(pulse_path, save_path='/home/leander/code/rlquantopt/rlquantopt/rl_envs', render=False, n_time_steps=2, T=tlist[-1], action_scaling=dict(z=1), pulse_length=pulse_length, fid_thresh=0.99, tv_penalty_scale=0)

    tlist2 = np.linspace(tlist[0], tlist[-1], len(rews))
    print(len(rews), len(pulse))
    fig, axs = plt.subplots(3)
    ax = axs[0]
    t_max = tlist2[np.argmax(fids)]
    ax.set_title(f'Max fidelity = {max(fids) * 100:.2f}%  @ {t_max:.2f}ns')
    ax.plot(tlist2, rews, c='g', label='Reward', marker='.')
    ax.axvline(t_max, color='k', ls='--')
    ax.axhline(0.0, c='k')
    ax = axs[1]
    ax.plot(tlist2, fids, c='k', label='Fidelity', marker='.')
    ax.axvline(t_max, color='k', ls='--')
    ax = axs[2]
    ax.plot(tlist, pulse, c='r', label='Pulse amplitude', marker='.')
    for ax in axs:
        ax.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    test_mkrauss_pulse()