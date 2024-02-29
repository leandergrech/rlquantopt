import krotov
import numpy as np

from rlquantopt_mc.one_qubit import OneQubit

if __name__ == '__main__':
	nstates = 1
	tlist = np.linspace(0, 10, 2000)
	T = tlist[-1]
	time_slots = 100

	x0 = np.array([krotov.shapes.flattop(t, 0, T, t_rise=0.5, func='sinsq') for t in np.linspace(0, T, time_slots - 2)])
	method = 'BFGS'

	gate = np.array([[0, 1], [1, 0]])

	one_qubit = OneQubit(gate, tlist, nstates=nstates)

	print(one_qubit.gate_fidelity(x0))

	result = one_qubit.run(time_slots=time_slots, x0=x0, method=method)

	print(one_qubit.gate_fidelity(result.x))
