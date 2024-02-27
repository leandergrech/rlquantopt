import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

from qutip_qip.circuit import QubitCircuit
from qutip_qip.device import SCQubits
from qutip import basis, ptrace


class QuPulseEpisodicEnv(gym.Env):
    def __init__(self, pulse_length=300):
        super(QuPulseEpisodicEnv, self).__init__()
        self.cur_step = 0
        self.pulse_length = pulse_length
        self.delta_t = 1    # ns

        # Initialise simulator
        self.n_levels = n_levels = 3
        qc = QubitCircuit(N=2)
        qc.add_gate("CNOT", controls=0, targets=1)

        self.simulator = SCQubits(num_qubits=2, dims=[n_levels, n_levels], wq=[5.15, 5.09], wr=[5.96, 5.96], g=[0.1, 0.1], alpha=[-0.3, -0.3], omega_single=[0.01, 0.01], omega_cr=[0.01, 0.01], t1=50.e3, t2=20.e3)
        # print(processor.get_control('sx0'))
        self.simulator.pulse_mode = 'discrete'
        self.simulator.load_circuit(qc)

        self.channel_labels = self.simulator.get_control_labels()
        self.n_channels = n_channels = len(self.channel_labels)

        # Action space holds amplitude deltas for all channels
        self.action_space = spaces.Box(low=-1, high=1, shape=(n_channels,), dtype=np.float32)
        # Observation space holds 4 basis vectors of a 2 qubit system + previous pulse amplitudes
        self.nb_basis_states = 2**2
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.nb_basis_states + n_channels,), dtype=np.float32)

        self.amplitude_buffer = self.init_amplitude_buffer()
        self.current_state = None

        self.basis00 = basis([self.n_levels, self.n_levels], [0, 0])

    def init_amplitude_buffer(self):
        return np.zeros((self.n_channels, self.pulse_length))  # First row for u01, second row for d1

    def reset(self, *params):
        self.cur_step = 0
        self.amplitude_buffer = np.zeros(shape=(self.n_channels, self.pulse_length))
        # TODO: Should I initialise state to all zeros or to [1, 0, 0, 0]?
        self.current_state = np.concatenate([np.zeros(self.nb_basis_states + self.n_channels)])
        return self.current_state

    def step(self, action):
        prev_action = np.zeros(self.n_channels)
        if self.cur_step > 0:
            prev_action = self.amplitude_buffer[:, self.cur_step - 1]
        self.amplitude_buffer[:, self.cur_step] = prev_action + action
        done = False

        # Update state with forward dynamics
        # TODO: denormalise action
        final_state = self.forward_dynamics(self.amplitude_buffer)

        self.cur_step += 1

        # Check if episode is done
        if self.cur_step >= self.pulse_length:
            done = True
            reward = self.objective_function(final_state)
        else:
            reward = 0  # Reward is calculated at the end of the episode

        # TODO: normalise action amplitude buffer in the current state
        self.current_state = np.concatenate([np.diag(final_state), self.amplitude_buffer])
        return self.current_state, reward, done, {}

    def forward_dynamics(self, amplitudes):
        for i, pulse in enumerate(self.simulator.pulses):
            self.simulator.pulses[i].coeff = amplitudes[i][:self.cur_step+1]
            self.simulator.pulses[i].tlist = np.arange(self.cur_step+1)
        # TODO: Set tlist for each pulse

        # Use qiskit.pulse to simulate the system's forward dynamics based on the amplitudes
        result = self.simulator.run_state(init_state=self.basis00)
        return result.states[-1]

    def objective_function(self, state):
        # Check the fidelity of the circuit being implemented
        # The following is just one starting basis state |00> with CNOT gate. Result should be |00>
        # TODO: expand fidelity computation to cover all basis vectors
        fidelity00 = np.real((self.basis00.dag() * ptrace(state, [0,1]) * self.basis00)[0,0])
        return -np.log(1 - fidelity00)


if __name__ == '__main__':
    env = QuPulseEpisodicEnv()
    print(env.reset())
    print(env.step(np.zeros(8)))
