import numpy as np
from qutip_qip.device import OptPulseProcessor, LinearSpinChain, SpinChainModel, SCQubits, SCQubitsModel
from qutip_qip.circuit import QubitCircuit
from qutip import sigmaz, sigmax, identity, tensor, basis, ptrace

import matplotlib.pyplot as plt

# qc = QubitCircuit(N=3)
qc = QubitCircuit(N=1)
qc.add_gate("X", targets=0)
# qc.add_gate("SNOT", targets=0)
# qc.add_gate("SNOT", targets=1)
# qc.add_gate("SNOT", targets=2)
#
# # function f(x)
# qc.add_gate("CNOT", controls=0, targets=2)
# qc.add_gate("CNOT", controls=1, targets=2)
#
# qc.add_gate("SNOT", targets=0)
# qc.add_gate("SNOT", targets=1)

processor = SCQubits(num_qubits=1)
processor.load_circuit(qc)

plt.rcParams['font.size']=20
processor.plot_pulses(title="Microwave pulse example", figsize=(15, 10), dpi=100, show_axis=False)
for ax in plt.gcf().axes:
    ax.set_xlabel('Time')
    # ax.set_ylabel('Amplitude')
plt.show(block=True)

# # Without decoherence
# basis00 = basis([3, 3], [0, 0])
# psi0 = basis([3, 3, 3], [0, 0, 0])
# result = processor.run_state(init_state=psi0)
# print("No Decoherence - Probability of measuring state 00:")
# print(np.real((basis00.dag() * ptrace(result.states[-1], [0,1]) * basis00)[0,0]))
#
# # With decoherence
# processor.t1 = 50.e3
# processor.t2 = 20.e3
# psi0 = basis([3, 3, 3], [0, 0, 0])
# result = processor.run_state(init_state=psi0)
# print("Decoherence - Probability of measuring state 00:")
# print(np.real((basis00.dag() * ptrace(result.states[-1], [0,1]) * basis00)[0,0]))

# Using optimal control module
setting_args = {"SNOT": {"num_tslots": 6, "evo_time": 2},
                "X": {"num_tslots": 1, "evo_time": 0.5},
                "CNOT": {"num_tslots": 12, "evo_time": 5}}
opt_processor = OptPulseProcessor(  # Use the control Hamiltonians of the spin chain model.
    num_qubits=3, model=SCQubitsModel(3))
opt_processor.load_circuit(  # Provide parameters for the algorithm
    qc, verbose=True, setting_args=setting_args, merge_gates=False,
    amp_ubound=5, amp_lbound=0)
opt_processor.plot_pulses(title="Control pulse of SCQubits", figsize=(8, 4), dpi=100)

# plt.show()