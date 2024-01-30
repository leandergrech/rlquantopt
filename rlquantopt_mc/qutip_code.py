import krotov
import numpy as np
from Python.CNOT import CNOT

if __name__ == '__main__':
	# This implicit factor is because frequencies convert to energies as E = h * nu, but our
	# propagation routines assume a unit h_bar = 1 for energies. Thus, the factor h / h_bar = 2 * pi.
	GHz = 2 * np.pi
	MHz = 1e-3 * GHz
	ns = 1
	us = 1000 * ns

	omega_1 = 4.3796 * GHz  # qubit frequency 1
	omega_2 = 4.6137 * GHz  # qubit frequency 2
	omega_d = 4.4985 * GHz  # drive frequency
	alpha_1 = -239.3 * MHz  # anharmonicity 1
	alpha_2 = -242.8 * MHz  # anharmonicity 2
	J = -2.3 * MHz  # effective qubit-qubit coupling
	q1T1 = 38.0 * us  # decay time for qubit 1
	q2T1 = 32.0 * us  # decay time for qubit 2
	q1T2 = 29.5 * us  # dephasing time for qubit 1
	q2T2 = 16.0 * us  # dephasing time for qubit 2
	T = 400 * ns  # gate duration

	n_qubit = 5  # number of states to truncate to

	tlist = np.linspace(0, T, 2000)

	time_slots = 100

	x0_re = 35 * MHz * np.array([krotov.shapes.flattop(t, 0, T, t_rise=20 * ns, func='sinsq') for t in np.linspace(0, T, time_slots)])
	x0_im = np.zeros(time_slots)
	x0 = np.concatenate((x0_re, x0_im))
	method = 'Nelder-Mead'

	liouvillian_kwargs = dict(omega_1=omega_1, omega_2=omega_2, omega_d=omega_d, alpha_1=alpha_1, alpha_2=alpha_2, J=J,
	                          q1T1=q1T1, q2T1=q2T1, q1T2=q1T2, q2T2=q2T2, n_qubit=n_qubit, tlist=tlist)

	cnot = CNOT(liouvillian_kwargs)

	cnot.run(time_slots=time_slots, x0=x0, method=method)
