import numpy as np

from rlquantopt_mc.one_qubit_poly import OneQubit

if __name__ == '__main__':
	nstates = 3
	tlist = np.linspace(0, 10, 2001)
	T = tlist[-1]
	time_slots = 21

	x0 = np.random.uniform(-0.1, 0.1, time_slots - 2)
	method = 'COBYLA'

	gate = np.array([[0, 1], [1, 0]])

	one_qubit = OneQubit(gate, tlist, nstates=nstates)

	print(one_qubit.gate_fidelity(x0))

	one_qubit.run(x0=x0, method=method)

	print(one_qubit.gate_fidelity(one_qubit.x))
