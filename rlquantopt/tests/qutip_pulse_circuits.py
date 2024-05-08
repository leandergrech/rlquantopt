import matplotlib.pyplot as plt
import numpy as np
from qutip import basis, ptrace
from qutip_qip.circuit import QubitCircuit
from qutip_qip.device import SCQubits

from rlquantopt.utils.utils import animate_matrices

nb_levels = 2

gate = 'CNOT'
qc = QubitCircuit(N=2)
qc.add_gate("CNOT", controls=0, targets=1)
# qc.add_gate(gate, targets=1)

processor = SCQubits(num_qubits=2, dims=[nb_levels, nb_levels], wq=[5.15, 5.09], wr=[5.96, 5.96], g=[0.1, 0.1],
                     alpha=[-0.3, -0.3], omega_single=[0.01, 0.01], omega_cr=[0.01, 0.01], t1=50.e3, t2=20.e3)
print(processor.get_control_labels())
# print(processor.get_control('sx0'))
# processor.pulse_mode = 'discrete'
processor.load_circuit(qc)
processor.plot_pulses(show_axis=True)
# plt.show()

# Get pulse info
for p in processor.pulses:
    print(f'Pulse {p.label}: min {min(p.coeff)}, max {max(p.coeff)}')

# Plot pulses
plt.rcParams['font.size'] = 20

# Run simulation
init_q0 = 0
init_q1 = 0
basis_states = []
for init_q0 in range(2):
    for init_q1 in range(2):
        basis00 = basis([nb_levels, nb_levels], [init_q0, init_q1])
        basis_states.append(basis00)

# target_states = sc.dot(basis_states)

from rlquantopt.rl_envs.qu_pulse_episodic_env import QuPulseEpisodicEnv
env = QuPulseEpisodicEnv()

results = []
for basis in basis_states:
    results.append(processor.run_state(init_state=basis))
    # ani = animate_matrices([np.real(item) for item in result.states], n_levels=nb_levels)
    # ani.save(f'{gate}_2-qubits_{nb_levels}-levels_|{init_q0}{init_q1}>-init.gif')
    # print(f"No Decoherence - Probability of measuring state |{init_q0}{init_q1}>:")
    # print(result)
    # fidelity = np.real((basis00.dag() * ptrace(result.states[-1], [0,1]) * basis00)[0,0])
    # fidelity = np.real((basis00.dag() * ptrace(result.states[-1], [0,1]) * basis00)[0,0])
fidelity = env.reward_function()
print(f'{fidelity*100:.2f}%')
plt.show()
	# for init_q1 in range(2):
	# 	basis00 = basis([nb_levels, nb_levels], [init_q0, init_q1])
	# 	result = processor.run_state(init_state=basis00)
	# 	ani = animate_matrices([np.real(item) for item in result.states], n_levels=nb_levels)
	# 	ani.save(f'{gate}_2-qubits_{nb_levels}-levels_{init_q0}{init_q1}-init.gif')
	# 	# plt.show()
	# 	# print("No Decoherence - Probability of measuring state 00:")
	# 	# print(result)
	# 	fidelity = np.real((basis00.dag() * ptrace(result.states[-1], [0, 1]) * basis00)[0, 0])
	# 	# print(f'{fidelity*100:.2f}%')
