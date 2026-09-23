import os

import setuptools.namespaces
from gymnasium import spaces
import yaml
import numpy as np
# from keras.src.backend import shape
from tqdm import tqdm
import pandas as pd
from qutip import Qobj, ket, QobjEvo, SESolver, expect, MESolver
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.animation import FuncAnimation
from gymnasium import Env
import random
from weylchamber import c1c2c3, concurrence
# from weylchamber.coordinates import c1c2c3
# from weylchamber.perfect_entanglers import concurrence

from rlquantopt.rl_envs.zcqubits import ZCQubits, setup_ZCQubits4MKrauss_params#, fidelity
# from rlquantopt.tests.zc_qpee_vs_zcqubits.zc_qpee_vs_zcqubits import env_kwargs

mpl.rcParams['font.size'] = 18


class ZCQPEE(Env):
    """
    Z-Control Quantum Pulse Episodic Environment (ZCQPEE)

    This environment is designed for reinforcement learning (RL)-based optimization of quantum gates and perfect
    entanglers, using a three-qutrit system with a central coupler. ZCQPEE allows RL agents to learn optimal
    control strategies for generating high-fidelity pulses under varying Hamiltonian dynamics.

    Key Features
    ------------
        - Interaction with a quantum simulator to evolve state vectors based on pulse amplitudes.
        - Support for delta-based and absolute amplitude control.
        - Reward functions emphasizing either fidelity for the iSWAP gate or a weighted sum of the concurrence and
            unitarity of the realised gate, and pulse smoothness via a total variation-based penalty.
        - Action transformations to test policies under different smoothness conditions.
        - Configurable action and observation spaces tailored for RL training.

    Attributes
    ----------
        pulse_length (int): Number of discrete time steps for the pulse.
        T (float): Total duration of the pulse in nanoseconds.
        delta_mode (bool): If True, actions represent delta amplitudes; otherwise, absolute values.
        action_scaling (dict): Scaling factors for action amplitudes per channel.
        ADD_PREV_OBS (bool): If True, includes the previous state in the observation space.
        FID_THRESH (float): Target fidelity threshold for reward computation.
        REW_SCALE (float): Scaling factor for rewards.
        TV_PENALTY_SCALE (float): Regularization penalty for abrupt pulse transitions.
        TERMINAL_PENALTY (float): Penalty for terminal conditions like state crashes.
        RANDOM_START_PROB (float): Probability of random initialization for the episode.
        RANDOM_START_STEPS (int): Number of random initializations for the episode (Depends on RANDOM_START_PROB)
    """
    # Pulse parameters
    pulse_length = 1000
    T = 50.
    delta_mode = True

    # Action parameters
    N_TIME_STEPS = 3
    action_scaling = {'z': 20.}    # Only applicable in delta_mode==True
    A_norm_max = 1
    ACT_POLY_ORDER = 1          # Action transformation polynomial order, i.e. 1 = linear, 2 = quadratic, etc.

    # Observable parameters
    OBS_SCALE = 0.9
    ADD_PREV_OBS = False

    # step method return info dictionary
    DUMP_INFO = False

    # Reward parameters
    FID_THRESH = 0.99
    REW_SCALE = 1
    TV_PENALTY_SCALE = 1e-3
    USE_FIDELITY = False
    LOG_LIM = 1e-2
    CONCURRENCE_WEIGHT = 1
    UNITARITY_WEIGHT = 3
    TERMINAL_PENALTY = -20

    # Rendering parameters
    PREC = 5e-4
    PLOT_LOG_EPS = 1e-4
    cmap = mpl.colormaps["CMRmap"]
    FPS = 10
    DPI = 80

    # Random-walk episode initialisation - off by default
    RANDOM_START_PROB = 0.0
    RANDOM_START_STEPS = 10

    def __init__(self, model_params=None, **env_kwargs):
        """
        Initializes the ZCQPEE environment with specified Hamiltonian and control settings.

        Args:
            model_params (dict, optional): Parameters for configuring the qutrit Hamiltonian and simulator.
            env_kwargs (dict): Additional configuration for the environment, including pulse properties,
                reward parameters, and initialization settings.

        Raises:
            ValueError: If the time step spacing in `tlist` does not match the expected value.
        """
        self.env_kwargs = env_kwargs
        # Set up random starts
        if env_kwargs.get('start_basis_states', False):
            self.RANDOM_START_PROB = 0.
        self.RANDOM_START_PROB = env_kwargs.get('random_start_prob', self.RANDOM_START_PROB)
        self.RANDOM_START_STEPS = env_kwargs.get('random_start_steps', self.RANDOM_START_STEPS)

        # Pulse parameters
        self.pulse_length = env_kwargs.get('pulse_length', self.pulse_length)
        self.delta_mode = env_kwargs.get('delta_mode', self.delta_mode)
        self.T = env_kwargs.get('T', self.T)

        # Reward related parameters
        self.FID_THRESH = env_kwargs.get('fid_thresh', self.FID_THRESH)
        self.REW_SCALE = env_kwargs.get('rew_scale', self.REW_SCALE)
        self.USE_FIDELITY = env_kwargs.get('use_fidelity', self.USE_FIDELITY)
        self.TV_PENALTY_SCALE = env_kwargs.get('tv_penalty_scale', self.TV_PENALTY_SCALE)
        self.TERMINAL_PENALTY = env_kwargs.get('terminal_penalty', self.TERMINAL_PENALTY)
        self.CONCURRENCE_WEIGHT = env_kwargs.get('concurrence_weight', self.CONCURRENCE_WEIGHT)
        self.UNITARITY_WEIGHT = env_kwargs.get('unitarity_weight', self.UNITARITY_WEIGHT)

        # Calculate reward threshold
        self.REW_THRESH = 0
        '''
        if self.USE_FIDELITY:
            pass
            # self.REW_THRESH = self._fid2rew(self.FID_THRESH) * self.REW_SCALE
        else:
            self.REW_THRESH = self._fid2rew(self.LOG_LIM) * self.REW_SCALE
        '''
        # Observable related parameters
        self.ADD_PREV_OBS = env_kwargs.get('add_prev_obs', self.ADD_PREV_OBS)
        self.OBS_SCALE = env_kwargs.get('obs_scale', self.OBS_SCALE)

        # step method info dictionary
        self.DUMP_INFO = env_kwargs.get('dump_info', self.DUMP_INFO)

        # Action related parameters
        self.ACT_POLY_ORDER = env_kwargs.get('act_poly_order', ZCQPEE.ACT_POLY_ORDER)
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
        # self.unitary_iswap = U = self.get_iswap_u()
        self.unitary_iswap = U = self.get_sqrtiswap_u()

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
        # options = dict(method='adams')#, atol=1e-10, rtol=1e-8, order=5)
        options = dict(method='adams', atol=1e-10, rtol=1e-8)
        self.solvers = [SESolver(self.simulator.H, options=options) for _ in range(len(self.basis_states))]
        # self.solvers = [MESolver(self.simulator.H, options=options) for _ in range(len(self.basis_states))]

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
            self.n_obs = (24 * 2)
        else:
            self.n_obs = 24
        # Including time and previous amplitudes
        self.n_obs += 1 + self.N_TIME_STEPS

        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.n_obs,), dtype=np.float32)

        # All initialised after first call to reset method
        self.cur_idx: int = 0
        self.final_states_all: list = []
        self.pulse_amplitudes_denorm: np.ndarray = np.array([])
        self.pulse_deltas_denorm: np.ndarray = np.array([])
        self.forward_dynamics_amps_cur: np.ndarray = np.array([])
        self.step_states: list = []
        self.prev_step_states: list = []
        self.current_obs: np.ndarray = np.array([])
        self.rewards: list = []
        self.fidelities: list = []
        self.concurrences: list = []
        self.unitarities: list = []

    @staticmethod
    def get_iswap_u():
        """
        Provides the unitary matrix for the iSWAP gate.

        This matrix represents the desired quantum operation that the RL agent aims to achieve.

        Returns:
            Qobj: The unitary matrix of the iSWAP gate.
        """
        return Qobj(np.array([
            [1, 0, 0, 0],
            [0, 0, 1j, 0],
            [0, 1j, 0, 0],
            [0, 0, 0, 1]
        ]), dims=[[2, 2], [2, 2]])  # iSWAP

    @staticmethod
    def get_sqrtswap_u():
        return Qobj(np.array([
            [1, 0, 0, 0],
            [0, (1+1j)/2, (1-1j)/2, 0],
            [0, (1-1j)/2, (1+1j)/2, 0],
            [0, 0, 0, 1]
        ]), dims=[[2, 2], [2, 2]])

    @staticmethod
    def get_sqrtiswap_u():
        inv_sqrt2 = 1 / np.sqrt(2)
        return Qobj(np.array([
            [1, 0, 0, 0],
            [0, inv_sqrt2, inv_sqrt2*1j, 0],
            [0, inv_sqrt2*1j, inv_sqrt2, 0],
            [0, 0, 0, 1]
        ]), dims=[[2, 2], [2, 2]])

    def init_pulse_amplitudes(self, delta):
        if delta:
            return np.zeros(self.pulse_length)
        else:
            return np.zeros(self.pulse_length + 1)

    def reset(self, init_state=None, seed=None, random_starts=False, options=None):
        """
        Resets the environment to its initial state, including quantum system initialization
        and episode-specific configurations.

        Args:
            init_state (list, optional): Custom initial state for the quantum system.
            seed (int, optional): Seed for reproducible randomization.
            random_starts (bool): If True, initializes after a random walk.
            options (dict, optional): Additional options for reset behavior.

        Returns:
            tuple: Initial observation and a dictionary with additional information.
        """
        super().reset(seed=seed, options=options)
        if seed is not None:
            np.random.seed(seed)
        self.cur_idx = 0
        self.final_states_all = [self.initial_states.copy()]
        self.pulse_amplitudes_denorm = self.init_pulse_amplitudes(delta=False)
        if self.delta_mode:
            self.pulse_deltas_denorm = self.init_pulse_amplitudes(delta=True)
        self.forward_dynamics_amps_cur = np.zeros(self.N_TIME_STEPS)
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
        self.concurrences = []
        self.unitarities = []

        for k, initial_state in enumerate(self.initial_states):
            self.solvers[k].start(initial_state, 0.)

        # Included just in case initial exploration is very hard
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
            norm_amps_cur = self.norm_action(self.forward_dynamics_amps_cur)
            obs_ampt.append(np.array([norm_amps_cur / self.A_norm_max]).reshape(-1, 1))
        else:
            obs_ampt.append(np.array(action).reshape(-1, 1))

        if include_time:
            # Add the time of current step
            obs_ampt.append(np.array(self.cur_idx * 2 / self.pulse_length - 1).reshape(-1, 1))

        return np.concatenate(obs_ampt)

    def extract_current_cardinal_state(self, states, action=None, state_only=False, include_time=True):
        """
        Extracts and processes the system's current state vector.

        Converts the state vector into a polar coordinate representation and
        normalizes the relevant quantum states. Adds additional information
        like normalized pulse amplitude and time, depending on flags.

        Args:
            states (list): List of current quantum state vectors.
            action (array-like, optional): Action to include in the observation.
            state_only (bool): If True, extracts only the state information.
            include_time (bool): If True, includes the current time in the observation.

        Returns:
            np.ndarray: Normalized observation vector representing the system state.
        """
        obs = []
        for i, state in enumerate(states):
            if i == 0:  # omit |000>
                continue

            if isinstance(state, Qobj):
                state = state.full()

            # Cartesian
            # extract_complex = lambda z: np.concatenate([s_.real.copy(), s_.imag.copy()])
            # Polar form
            def extract_complex(z):
                return np.concatenate([(np.absolute(z) * 2) - 1, np.angle(z)/(np.pi)])
            # extract_complex = lambda z: np.concatenate([(np.absolute(z) * 2) - 1, np.angle(z)/np.pi])
            if i < 3:       # for states |010>, |100>
                for idx in (1, 3, 9):   # Extract only these elements from the state vector @ 0 < i < 3
                    s_ = state[idx]
                    obs.append(extract_complex(s_))
            elif i == 3:    # for state |110>
                for idx in (2, 4, 6, 10, 12, 18):   # Extract only these elements from the state vector @ i == 3
                    s_ = state[idx]
                    obs.append(extract_complex(s_))

        if not state_only:
            obs.extend(self.get_current_amp_time_norm_tuple(action, include_time=include_time))

        obs = np.concatenate(obs).squeeze()
        obs *= self.OBS_SCALE

        return obs.astype(np.float32)

    def norm_action(self, action):
        """
        Normalize the given action using the scaling factor.
        """
        SCALE = self.action_scaling['z']
        return action / SCALE

    def denorm_action(self, action):
        """
        Denormalize the given action using the scaling factor.
        """
        SCALE = self.action_scaling['z']
        return np.array(action) * SCALE

    @staticmethod
    def transform_action(action):
        """
        Apply a sign-preserving polynomial transformation to the action.

        The transformation adjusts the action amplitude based on the configured
        polynomial order. This is used to enhance agent exploration and test
        smooth pulse generation policies.

        Args:
            action (array-like): The action to be transformed.

        Returns:
            np.ndarray: Transformed action with the same shape as the input.
        """
        # harmonic methods
        # action = np.clip(np.abs(action), 0, 1 - np.exp(-5))
        # action = -0.2 * np.log(np.ones_like(action) - action)
        signs = np.sign(action)
        action = np.power(np.abs(action), ZCQPEE.ACT_POLY_ORDER)
        return signs * action

    def step(self, action, can_early_term=False):
        """
        Executes a single step in the environment by applying the given action.

        Args:
            action (array-like): A set of amplitude deltas or absolute values for the pulse.
            can_early_term (bool): If True, allows early termination when the reward threshold is exceeded.

        Returns:
            tuple: Next observation, reward, termination flag, truncation flag, and auxiliary info.
        """
        # print(action)
        action = np.squeeze(action)
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
            self.step_states = self.forward_dynamics(np.squeeze(amp_abs_denorm))
        except Exception as e:
            print(e)
            crash = True

        terminated, truncated, terminal = False, False, False

        # Calculate reward for current action
        reward, tv_penalty, fid, concur, unitarity, realised_gate = self._reward_function(self.step_states, self.forward_dynamics_amps_cur)

        # Assign penalty if pulse crashed the simulator & send trancation signal to stop the episode
        terminal_penalty = self.TERMINAL_PENALTY
        if crash:
            reward = terminal_penalty
            truncated = True
        # Assign smaller penalty if the pulse os OOB
        elif oob_pulse:
            reward = terminal_penalty * (1 - (self.cur_idx / self.pulse_length))
            truncated = True

        if reward > 0 and can_early_term:
            terminated = True

        # Collect for rendering
        self.final_states_all.append(self.step_states.copy())
        self.rewards.append(reward)
        self.fidelities.append(fid)
        self.concurrences.append(concur)
        self.unitarities.append(unitarity)

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

        info = {'tv_penalty': tv_penalty, 'fidelity': fid, 'oob_pulse': oob_pulse,
                'crash': crash, 'terminal_state': terminal, 'concurrence': concur, 'unitarity': unitarity}
        if self.DUMP_INFO:
            info.update({'step_states': self.step_states, 'realised_gate': realised_gate})

        return self.current_obs, reward, terminated, truncated, info

    def preprocess_action(self, action_denorm):
        """
        Preprocesses the action to produce absolute pulse amplitudes.

        Converts the relative (delta-based) action into absolute amplitudes
        or retains the absolute form, depending on the environment's mode.

        Args:
            action_denorm (array-like): Denormalized action amplitudes for the current step.

        Returns:
            tuple:
                - np.ndarray: Processed absolute pulse amplitudes.
                - bool: Flag indicating if the pulse exceeds bounds.
        """
        oob_pulse = False
        # Delta formalism
        if self.delta_mode:
            if self.cur_idx == 0:
                # First time-step assumes a prev. amplitude of 0 - start from zero-pulse
                prev_amp_abs_denorm = 0.
            else:
                # In delta formalism, we need the previous amplitude
                prev_amp_abs_denorm = self.forward_dynamics_amps_cur[-1]

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

        return np.squeeze(amp_abs_denorm).reshape(1,-1), oob_pulse

    def forward_dynamics(self, amp_abs_denorm):
        """
        Simulates the quantum system's state evolution for the current time step.

        Propagates the system's state vector using the simulator's solver
        over the given amplitudes and updates the forward dynamics.

        Note: Requires self.cur_idx==0 during the first step to calculate the pulse time correctly.

        Args:
            amp_abs_denorm (array-like): Absolute pulse amplitudes for the current step.

        Returns:
            list: Updated state vectors after the simulation step.
        """

        # Step through the simulator with new amplitude until next time-step in simulation
        self.forward_dynamics_amps_cur = amp_abs_denorm
        final_states = []
        for k, _ in enumerate(self.step_states):
            final_state = None
            for i in range(self.N_TIME_STEPS):
                t = self.tlist[self.cur_idx + i + 1]    # The +1 will skip the first time sample @ 0ns
                final_state = self.solvers[k].step(t, args=dict(A=amp_abs_denorm[i]))
            final_states.append(final_state)
        return final_states

    @staticmethod
    def _compute_total_variation(acts):
        """
        Compute the total variation (sum of absolute differences) of actions.
        """
        return np.sum(np.abs(np.diff(acts)))

    def _reward_function(self, step_states, acts=None):
        """
        Computes the reward for the current step based on metrics on realised quantum gate and pulse smoothness.

        The reward includes:
        - Fidelity-based metric for target sqrtiSWAP gate or Concurrence-Unitarity-based metric.
        - Penalties for abrupt action changes (TV penalty).

        Args:
            step_states (list): Final state vectors after the step.
            acts (array-like, optional): Actions applied in the step.

        Returns:
            tuple:
                - float: Computed reward value.
                - float: Total variation penalty.
                - float: Fidelity score.
                - float: Concurrence metric.
                - float: Unitarity metric.
                - Qobj: Realized gate matrix.
        """
        n_states = len(step_states)
        realised_gate = np.zeros(shape=(n_states, n_states), dtype=np.complex128)
        for i, s in enumerate(step_states):
            for j, b in enumerate(self.basis_states):
                realised_gate[i,j] = s.overlap(b)
        realised_gate = Qobj(realised_gate, dims=([[2,2], [2,2]]))

        c = concurrence(*c1c2c3(realised_gate))
        u = (realised_gate.dag() * realised_gate).tr() / 4

        overlaps = [s.overlap(t) for s, t in zip(step_states, self.target_states)]
        f = np.abs(np.sum(overlaps)) ** 2 / len(overlaps) ** 2

        if self.USE_FIDELITY:
            rew = self._fid2rew(f)
        else:
            metric = (
                    (self.CONCURRENCE_WEIGHT * c + self.UNITARITY_WEIGHT * u)
                    / (self.CONCURRENCE_WEIGHT + self.UNITARITY_WEIGHT)
            )
            rew = self._fid2rew(metric)

        # rew += (self.CONCURRENCE_WEIGHT * c + self.UNITARY_WEIGHT * u) / (self.CONCURRENCE_WEIGHT + self.UNITARY_WEIGHT)

        rew *= self.REW_SCALE
        rew -= self.REW_THRESH  # any rew below threshold obtains a -ve value; +ve otherwise

        if acts is None:
            return rew, 0.0, f, c, u

        total_variation = self._compute_total_variation(acts)
        tv_penalty = total_variation * self.TV_PENALTY_SCALE
        rew -= tv_penalty

        return rew, tv_penalty, f, c, u, realised_gate

    @staticmethod
    def _fid2rew(fid):
        """
        Expands the error of a number approaching 1 using a logarithmic scale.

        This function maps a value in the range [0, 1) to a reward-like metric that
        emphasizes values close to 1. The logarithmic transformation inflates the
        difference between values near 1, making small deviations from 1 more
        significant.

        Args:
            value (float): A number in the range [0, 1), where 1 represents an ideal case.

        Returns:
            float: Logarithmic expansion of the error (1 - value).
        """
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
        return None

    def render(self, mode='human', save_path=None, fps=80, **kwargs):
        """
        Visualizes the pulse evolution and system dynamics during the episode.

        Args:
            mode (str): Render mode (default: 'human').
            save_path (str, optional): Path to save the animation as a file.
            fps (int): Frames per second for the rendered animation.
            **kwargs: Additional rendering options.

        Returns:
            Optional: Animation object if `save_path` is None; otherwise, saves the animation.
        """
        mpl.rcParams['font.size'] = 10
        states = self.final_states_all
        fidelities = np.multiply(self.fidelities, 100.)
        infidelities = [100 - item for item in fidelities]

        # Setup figure and axes
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle(f'sqrtiSWAP\naction scale={self.action_scaling[self.channel_label]:.2e}')
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
            action_line.set_data(self.tlist[:idx_], self.pulse_amplitudes_denorm[:idx_])
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
                f"          sqrt_iSWAP gate optimisation.\n"
                f"          Action delta_mode={self.delta_mode}\n"
                f"          Action polynomial order={ZCQPEE.ACT_POLY_ORDER}")

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


def test_obs_ranges(num_episodes=50, random_seed=42):
    import seaborn as sns
    from tqdm import trange
    """
    Perform Monte Carlo simulations with random actions on ZCQPEE and produce violin plots
    of the observable populations.
    """
    np.random.seed(random_seed)

    # Initialize environment
    env_kwargs = {
        "pulse_length": 1000,
        "delta_mode": True,
        "T": 50,
        "tv_penalty_scale": 1e-3,
        "fid_thresh": 0.99,
        "rew_scale": 1.0,
        "n_time_steps": 3,
        "action_scaling": {"z": 20.0},
        "a_norm_max": 1,
        "add_prev_obs": False
    }

    model_params = setup_ZCQubits4MKrauss_params()
    env = ZCQPEE(model_params=model_params, **env_kwargs)
    episode_length = env_kwargs['pulse_length'] // env_kwargs['n_time_steps']
    n_obs = env.n_obs

    # Collect observables from all episodes
    all_observables = []

    for episode in trange(num_episodes):
        obs, _ = env.reset(seed=random_seed)
        episode_observables = []
        for step in range(episode_length):
            random_action = env.action_space.sample()
            obs, r, done, _, _ = env.step(random_action)
            episode_observables.append(obs)
            if done:
                break
        all_observables.extend(episode_observables)

    # Convert observables to NumPy array for visualization
    all_observables = np.array(all_observables)

    # Create violin plots for each observation dimension
    plt.figure(figsize=(12, 8))
    sns.violinplot(data=all_observables, scale="width", inner="quartile")
    plt.title("Monte Carlo Violin Plots of Observation Dimensions")
    plt.xlabel("Observation Dimensions")

    plt.figure(figsize=(12, 8))

    plt.plot(np.arange(n_obs), np.min(all_observables, axis=0), marker='o', label='Min')
    plt.plot(np.arange(n_obs), np.max(all_observables, axis=0), marker='o', label='Max')
    plt.plot(np.arange(n_obs), np.median(all_observables, axis=0), marker='o', label='Med')
    plt.plot(np.arange(n_obs), np.mean(all_observables, axis=0), marker='o', label='Avg')
    plt.axhline(1, color='k', ls='dashed')
    plt.axhline(-1, color='k', ls='dashed')
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    # test_mkrauss_pulse()
    # env = ZCQPEE()
    # env._reward_function(env.basis_states)
    # env._reward_function(env.target_states)
    test_obs_ranges()
