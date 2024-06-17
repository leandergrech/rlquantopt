import matplotlib.pyplot as plt
import numpy as np
from qutip import basis, ptrace
from qutip.metrics import fidelity
from qutip_qip.circuit import QubitCircuit
from qutip_qip.device import SCQubits
from qutip_qip.pulse import Pulse
from qutip import sigmaz


from rlquantopt.utils import animate_matrices
from rlquantopt.rl_envs.zcqubits import ZCQubits

n_qubits = 2
nb_levels = 3

gate = 'CNOT'
qc = QubitCircuit(N=n_qubits)
qc.add_gate("CNOT", controls=0, targets=1)
u = qc.compute_unitary()
# qc.add_gate(gate, targets=1)

# processor = ZCQubits(num_qubits=n_qubits)
processor = SCQubits(num_qubits=2, dims=[nb_levels, nb_levels], wq=[5.15, 5.09], wr=[5.96, 5.96], g=[0.1, 0.1], alpha=[-0.3, -0.3], omega_single=[0.01, 0.01], omega_cr=[0.01, 0.01], t1=50.e3, t2=20.e3)
# for lbl in processor.get_control_labels():
#     coeff = np.random.uniform(-1, 1, 10)
#     tlist = np.linspace(0, 10, 11)
#     processor.pulses.append(Pulse(qobj=sigmaz(), targets=[1], tlist=tlist, coeff=coeff, label=lbl))
# print(processor.get_control('sx0'))
# processor.pulse_mode = 'discrete'
processor.load_circuit(qc)
# processor.plot_pulses(show_axis=True)
# plt.show()
# exit(23)
# Get pulse info
for j, p in enumerate(processor.pulses):
    print(f'Pulse {p.label}: min {min(p.coeff)}, max {max(p.coeff)}')

# exit(23)
# Plot pulses
plt.rcParams['font.size'] = 20

# Run simulation
init_q0 = 0
init_q1 = 0
basis_states = []
basis_str = []
for init_q0 in range(2):
    for init_q1 in range(2):
        basis_str.append(f'{init_q0}{init_q1}')
        basis00 = basis([3, 3], [init_q0, init_q1])
        basis_states.append(basis00)

# target_states = sc.dot(basis_states)

# from rlquantopt.rl_envs.qu_pulse_episodic_env import QuPulseEpisodicEnv
# env = QuPulseEpisodicEnv()

results = []
for i, basis in enumerate(basis_states):
    result = processor.run_state(init_state=basis)
    target_sate = u * basis
    fid = fidelity(result.states[-1], target_sate)
    print(f'Initial state: |{basis_str[i]}>\tFidelity: {fid*100:.2f}%')
    # results.append(result)
    ani = animate_matrices([np.abs(item) for item in result.states], n_levels=nb_levels)
    save_name = f'scqubits_{gate}_2-qubits_{nb_levels}-levels_|{basis_str[i]}>-init.gif'
    print(f'Saving state evolution animation to: {save_name}')
    ani.save(save_name)

    # print(f"No Decoherence - Probability of measuring state |{init_q0}{init_q1}>:")
    # print(result)
    # fidelity = np.real((basis.dag() * ptrace(result.states[-1], [0,1]) * basis)[0,0])
    # fidelity = np.real((basis00.dag() * ptrace(result.states[-1], [0,1]) * basis00)[0,0])
# fidelity = env.reward_function()
# plt.show()
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
