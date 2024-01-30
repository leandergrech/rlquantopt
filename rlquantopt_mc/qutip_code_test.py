import numpy as np
import matplotlib.pyplot as plt
import datetime

import qutip
import scipy
from qutip import identity
import qutip.control.pulseoptim as cpo


def transmon_hamiltonian(Ec=0.386, EjEc=45, nstates=8, ng=0.0, T=10.0):
	"""Transmon Hamiltonian

	Args:
		Ec: capacitive energy
		EjEc: ratio `Ej` / `Ec`
		nstates: defines the maximum and minimum states for the basis. The
			truncated basis will have a total of ``2*nstates + 1`` states

		ng: offset charge
		T: gate duration
	"""

	Ej = EjEc * Ec
	n = np.arange(-nstates, nstates + 1)
	up = np.diag(np.ones(2 * nstates), k=-1)
	do = up.T
	H0 = qutip.Qobj(np.diag(4 * Ec * (n - ng) ** 2) - Ej * (up + do) / 2.)
	H1 = qutip.Qobj(-2 * np.diag(n))

	return H0, [H1]


def logical_basis(H):
	H0 = H[0]
	eigenvals, eigenvecs = scipy.linalg.eigh(H0.full())
	print(eigenvals)
	ndx = np.argsort(eigenvals.real)
	V = eigenvecs[:, ndx]
	psi0 = qutip.Qobj(V[:, 0])
	psi1 = qutip.Qobj(V[:, 1])
	psi2 = qutip.Qobj(V[:, 2])
	return psi0, psi1, psi2, V


nstates = 2
H = transmon_hamiltonian(nstates=nstates)
H_d, H_c = H
psi0, psi1, psi2, V = logical_basis(H)
proj0 = qutip.ket2dm(psi0)
proj1 = qutip.ket2dm(psi1)

# start point for the gate evolution
U_0 = identity(2 * nstates + 1)
# Target for the gate evolution Hadamard gate
U_targ = (proj0 + np.outer(psi0, psi1.conj()) + np.outer(psi1, psi0.conj()) - proj1) / np.sqrt(2)

# Number of time slots
n_ts = 100
# Time allowed for the evolution
evo_time = 5

# pulse type alternatives: RND|ZERO|LIN|SINE|SQUARE|SAW|TRIANGLE|
p_type = 'SAW'

#Set to None to suppress output files
f_ext = None

result = cpo.optimize_pulse_unitary(H_d, H_c, U_0, U_targ, n_ts, evo_time, out_file_ext=f_ext,
                                    init_pulse_type=p_type, gen_stats=True)

result.stats.report()
print("Final evolution\n{}\n".format(result.evo_full_final))
print("********* Summary *****************")
print("Final fidelity error {}".format(result.fid_err))
print("Final gradient normal {}".format(result.grad_norm_final))
print("Terminated due to {}".format(result.termination_reason))
print("Number of iterations {}".format(result.num_iter))
print("Completed in {} HH:MM:SS.US".format(datetime.timedelta(seconds=result.wall_time)))

U = result.evo_full_final

psi = np.abs(V.conj().T @ (U.full() @ psi0.full())) ** 2
print(psi)


fig1 = plt.figure()
ax1 = fig1.add_subplot(2, 1, 1)
ax1.set_title("Initial control amps")
ax1.set_xlabel("Time")
ax1.set_ylabel("Control amplitude")
ax1.step(result.time, np.hstack((result.initial_amps[:, 0], result.initial_amps[-1, 0])), where='post')

ax2 = fig1.add_subplot(2, 1, 2)
ax2.set_title("Optimised Control Sequences")
ax2.set_xlabel("Time")
ax2.set_ylabel("Control amplitude")
ax2.step(result.time, np.hstack((result.final_amps[:, 0], result.final_amps[-1, 0])), where='post')
plt.tight_layout()
plt.show()
