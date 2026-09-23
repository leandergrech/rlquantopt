from itertools import product

import numpy as np
from matplotlib import pyplot as plt
from qutip import mesolve, Qobj, basis
from scipy.linalg import eigh
from scipy.optimize import minimize
from krotov.functionals import F_avg


class OneQubit:
	def __init__(self, gate, tlist, nstates=8):
		self.tlist = tlist
		self.nstates = nstates
		self.H0, self.H1 = self.transmon_hamiltonian()
		# self.psi0 = basis(2 * self.nstates + 1, 0)
		# self.psi1 = basis(2 * self.nstates + 1, 1)
		# self.basis_states = [self.psi0, self.psi1]
		self.eigenvectors = self.logical_basis(self.H0)
		self.basis_states = [Qobj(self.eigenvectors[:, 0]), Qobj(self.eigenvectors[:, 1])]
		self.full_liouville_basis = [psi * phi.dag() for psi, phi in product(self.basis_states, self.basis_states)]

		if not isinstance(gate, Qobj):
			self.unitary = Qobj(gate)
		else:
			self.unitary = gate

		self.mapped_basis_states = [sum(complex(self.unitary[i, j]) * self.basis_states[i]
		                                for i in range(self.unitary.shape[0])) for j in range(self.unitary.shape[1])]
		# Lots of gates just rearrange the basis states, and we can avoid some complexity by identifying
		# this and setting the mapped_basis_states to the identical objects as the original basis_states
		for i, state in enumerate(self.mapped_basis_states):
			for j, basis_state in enumerate(self.basis_states):
				if state == basis_state:
					self.mapped_basis_states[i] = basis_state

		self.target_states = self.mapped_basis_states

		self.nfev = 0
		self.iter = 0
		self.H = None
		self.time_slots = None
		self.x0 = None
		self.x = None
		self.F = None
		self.min_kwargs = None

	def transmon_hamiltonian(self, Ec=0.386, EjEc=45, ng=0.):
		Ej = EjEc * Ec
		n = np.arange(-self.nstates, self.nstates + 1)
		up = np.diag(np.ones(2 * self.nstates), k=-1)
		do = up.T
		H0 = Qobj(np.diag(4 * Ec * (n - ng) ** 2) - Ej * (up + do) / 2.0)
		H1 = Qobj(-2 * np.diag(n))

		return H0, H1

	@staticmethod
	def logical_basis(H):
		eigenvals, eigenvecs = eigh(H.full())
		ndx = np.argsort(eigenvals.real)
		return eigenvecs[:, ndx]

	def run(self, time_slots, x0, method=None, **min_kwargs):
		self.time_slots = time_slots
		self.x0 = x0
		self.plot_pulse(self.x0, self.tlist)
		self.plot_population_dynamics(self.x0, self.tlist)
		self.nfev = 0
		self.iter = 0
		# Minimize
		result = minimize(fun=self.cost_fun, x0=self.x0, method=method, options=min_kwargs, callback=self.callback)

		self.plot_pulse(result.x, self.tlist)
		self.plot_population_dynamics(result.x, self.tlist)

		return result

	def gate_fidelity(self, x):
		self.H = [self.H0, [self.H1, self.get_params(x, self.tlist)]]

		args_list = [(self.H, state, self.tlist) for state in self.full_liouville_basis]

		results = [self.wrapped_mesolve(args) for args in args_list]

		return F_avg(results, self.basis_states, self.unitary, self.mapped_basis_states, prec=1e-3)

	@staticmethod
	def wrapped_mesolve(args):
		H, psi, tlist = args
		sol = mesolve(H, psi, tlist)
		return sol.all_states[-1]

	@staticmethod
	def get_params(x, tlist):
		bin_len = int(len(tlist) / (len(x) + 2))
		_x = np.zeros(bin_len)
		_x = np.append(_x, np.repeat(x, bin_len))
		_x = np.append(_x, np.zeros(bin_len))

		return _x

	# noinspection PyUnusedLocal,PyTypeChecker
	def cost_fun(self, x, *args):
		self.nfev += 1
		self.x = x

		self.H = [self.H0, [self.H1, self.get_params(self.x, self.tlist)]]

		args_list = [(self.H, state, self.tlist) for state in self.basis_states]

		results = [self.wrapped_mesolve(args) for args in args_list]

		self.F = 0.5 * (np.abs(results[0].overlap(self.target_states[0])) ** 2 +
		                np.abs(results[1].overlap(self.target_states[1])) ** 2)

		return 1 - self.F

	# noinspection PyUnusedLocal
	def callback(self, x):
		self.iter += 1
		print(self.iter, self.nfev, self.F)

	# noinspection PyTypeChecker
	def plot_population_dynamics(self, x, tlist):
		H = [self.H0, [self.H1, self.get_params(x, tlist)]]

		e_ops = [self.basis_states[0].proj(), self.basis_states[1].proj()]

		sol0 = mesolve(H, self.basis_states[0], tlist, e_ops=e_ops)
		sol1 = mesolve(H, self.basis_states[1], tlist, e_ops=e_ops)

		fig, axs = plt.subplots(ncols=2, figsize=(16, 8))
		axs = np.ndarray.flatten(axs)
		labels = ['0', '1']
		expectations = [sol0.expect, sol1.expect]

		for ax, exp, title in zip(axs, expectations, labels):
			for i, label in enumerate(labels):
				ax.plot(tlist, exp[i], label=label)
			ax.legend()
			ax.set_title(title)
		plt.show()

	def plot_pulse(self, x, tlist):
		fig, ax = plt.subplots(figsize=(16, 8))
		ax.plot(tlist, self.get_params(x, tlist))
		plt.show()
