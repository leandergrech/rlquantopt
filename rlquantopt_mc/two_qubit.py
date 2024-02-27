from itertools import product
from multiprocessing import Pool

import numpy as np
from matplotlib import pyplot as plt
from qutip import tensor, identity, destroy, mesolve, ket, Qobj
from scipy.linalg import eigh
from scipy.optimize import minimize
from krotov.functionals import F_avg


class TwoQubit:
	def __init__(self, liouvillian_kwargs, gate):
		self.liouvillian_kwargs = liouvillian_kwargs
		self.omega_1 = self.liouvillian_kwargs.get('omega_1')
		self.omega_2 = self.liouvillian_kwargs.get('omega_2')
		self.omega_d = self.liouvillian_kwargs.get('omega_d')
		self.alpha_1 = self.liouvillian_kwargs.get('alpha_1')
		self.alpha_2 = self.liouvillian_kwargs.get('alpha_2')
		self.J = self.liouvillian_kwargs.get('J')
		self.q1T1 = self.liouvillian_kwargs.get('q1T1')
		self.q2T1 = self.liouvillian_kwargs.get('q2T1')
		self.q1T2 = self.liouvillian_kwargs.get('q1T2')
		self.q2T2 = self.liouvillian_kwargs.get('q2T2')
		self.n_qubit = self.liouvillian_kwargs.get('n_qubit')
		self.tlist = self.liouvillian_kwargs.get('tlist')
		self.H0, self.H1_re, self.H1_im, self.c_ops = self.two_qubit_transmon_liouvillian_operators(
			self.omega_1, self.omega_2, self.omega_d, self.alpha_1, self.alpha_2, self.J, self.q1T1, self.q2T1,
			self.q1T2, self.q2T2, self.n_qubit
		)

		self.psi00 = ket((0, 0), dim=(self.n_qubit, self.n_qubit))
		self.psi01 = ket((0, 1), dim=(self.n_qubit, self.n_qubit))
		self.psi10 = ket((1, 0), dim=(self.n_qubit, self.n_qubit))
		self.psi11 = ket((1, 1), dim=(self.n_qubit, self.n_qubit))

		# self.eigenvectors = self.logical_basis(self.H0)
		# self.psi00 = Qobj(self.eigenvectors[:, 0], dims=[[self.n_qubit, self.n_qubit], [1, 1]])
		# self.psi01 = Qobj(self.eigenvectors[:, 1], dims=[[self.n_qubit, self.n_qubit], [1, 1]])
		# self.psi10 = Qobj(self.eigenvectors[:, self.n_qubit], dims=[[self.n_qubit, self.n_qubit], [1, 1]])
		# self.psi11 = Qobj(self.eigenvectors[:, self.n_qubit + 1], dims=[[self.n_qubit, self.n_qubit], [1, 1]])

		self.basis_states = [self.psi00, self.psi01, self.psi10, self.psi11]
		self.full_liouville_basis = [psi * phi.dag() for psi, phi in product(self.basis_states, self.basis_states)]

		if not isinstance(gate, Qobj):
			self.unitary = Qobj(gate, dims=[[2, 2], [2, 2]])
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

		self.initial_states = [self._rho1(self.basis_states), self._rho2(self.basis_states), self._rho3(self.basis_states)]
		self.initial_state_purities = [s.purity() for s in self.initial_states]

		self.target_states = [self._rho1(self.mapped_basis_states), self._rho2(self.mapped_basis_states), self._rho3(self.mapped_basis_states)]

		self.nfev = 0
		self.iter = 0
		self.H = None
		self.time_slots = None
		self.x0 = None
		self.weights = None
		self.x = None
		self.F = None

	@staticmethod
	def _rho1(basis_states):
		return sum(((4 - i) / 10) * psi * psi.dag() for i, psi in enumerate(basis_states))

	@staticmethod
	def _rho2(basis_states):
		return 0.25 * sum(psi_i * psi_j.dag() for psi_i, psi_j in product(basis_states, repeat=2))

	@staticmethod
	def _rho3(basis_states):
		return 0.25 * sum(psi * psi.dag() for psi in basis_states)

	@staticmethod
	def two_qubit_transmon_liouvillian_operators(omega_1, omega_2, omega_d, alpha_1, alpha_2, J, q1T1, q2T1, q1T2, q2T2,
	                                             n_qubit):
		b1 = tensor(identity(n_qubit), destroy(n_qubit))
		b2 = tensor(destroy(n_qubit), identity(n_qubit))

		H0 = (
				(omega_1 - omega_d - alpha_1 / 2) * b1.dag() * b1
				+ (alpha_1 / 2) * b1.dag() * b1 * b1.dag() * b1
				+ (omega_2 - omega_d - alpha_2 / 2) * b2.dag() * b2
				+ (alpha_2 / 2) * b2.dag() * b2 * b2.dag() * b2
				+ J * (b1.dag() * b2 + b1 * b2.dag())
		)

		H1_re = 0.5 * (b1 + b1.dag() + b2 + b2.dag())  # 0.5 is due to RWA
		H1_im = 0.5j * (b1.dag() - b1 + b2.dag() - b2)

		A1 = np.sqrt(1 / q1T1) * b1  # decay of qubit 1
		A2 = np.sqrt(1 / q2T1) * b2  # decay of qubit 2
		A3 = np.sqrt(1 / q1T2) * b1.dag() * b1  # dephasing of qubit 1
		A4 = np.sqrt(1 / q2T2) * b2.dag() * b2  # dephasing of qubit 2

		return H0, H1_re, H1_im, [A1, A2, A3, A4]

	@staticmethod
	def logical_basis(H):
		eigenvals, eigenvecs = eigh(H.full())
		ndx = np.argsort(eigenvals.real)
		return eigenvecs[:, ndx]

	def run(self, time_slots, x0, weights=None, method=None):
		if weights is None:
			self.weights = np.array([1. / (3 * p) for p in self.initial_state_purities])
		else:
			self.weights = np.array([w / p for w, p in zip(weights, self.initial_state_purities)]) / sum(weights)
		self.time_slots = time_slots
		self.x0 = x0
		self.plot_pulse(self.x0, self.tlist)
		self.plot_population_dynamics(self.x0, self.tlist)
		self.nfev = 0
		self.iter = 0
		# Minimize
		result = minimize(fun=self.cost_fun, x0=self.x0, method=method, callback=self.callback)

		self.plot_pulse(result.x, self.tlist)
		self.plot_population_dynamics(result.x, self.tlist)

		return result

	def gate_fidelity(self, x):
		x_re = x[:len(x) // 2]
		x_im = x[len(x) // 2:]
		bin_len = int(2 * len(self.tlist) / len(x))

		self.H = [self.H0, [self.H1_re, np.repeat(x_re, bin_len)], [self.H1_im, np.repeat(x_im, bin_len)]]

		args_list = [(self.H, state, self.tlist, self.c_ops) for state in self.full_liouville_basis]

		results = [self.wrapped_mesolve(args) for args in args_list]

		return F_avg(results, self.basis_states, self.unitary)

	@staticmethod
	def wrapped_mesolve(args):
		H, psi, tlist, c_ops = args
		sol = mesolve(H, psi, tlist, c_ops=c_ops)
		return sol.states[-1]

	# noinspection PyUnusedLocal,PyTypeChecker
	def cost_fun(self, x, *args):
		self.nfev += 1
		self.x = x
		x_re = self.x[:len(self.x) // 2]
		x_im = self.x[len(self.x) // 2:]
		bin_len = int(2 * len(self.tlist) / len(self.x))

		self.H = [self.H0, [self.H1_re, np.repeat(x_re, bin_len)], [self.H1_im, np.repeat(x_im, bin_len)]]

		args_list = [(self.H, state, self.tlist, self.c_ops) for state in self.initial_states]

		# with Pool() as pool:
		# 	results = pool.map(self.wrapped_mesolve, args_list)

		results = [self.wrapped_mesolve(args) for args in args_list]

		self.F = sum(w * np.real(np.trace(s1 * s2)) for w, s1, s2 in zip(self.weights, self.target_states, results))

		# sol00 = mesolve(self.H, self.psi00, self.tlist, c_ops=self.c_ops, e_ops=self.psi11.proj())
		#
		# self.F = np.abs(sol00.expect[0][-1]) ** 2

		# result = mesolve(self.H, self.psi00, self.tlist, c_ops=self.c_ops)
		#
		# print(np.trace(result.states[0].full() @ result.states[0].full()))
		# print(np.trace(result.states[-1].full() @ result.states[-1].full()))
		# quit()

		return 1 - self.F

	# noinspection PyUnusedLocal
	def callback(self, x):
		self.iter += 1
		print(self.iter, self.nfev, self.F)

	# noinspection PyTypeChecker
	def plot_population_dynamics(self, x, tlist):
		x_re = x[:len(x) // 2]
		x_im = x[len(x) // 2:]
		bin_len = int(2 * len(tlist) / len(x))

		H = [self.H0, [self.H1_re, np.repeat(x_re, bin_len)], [self.H1_im, np.repeat(x_im, bin_len)]]

		e_ops = [self.psi00.proj(), self.psi01.proj(), self.psi10.proj(), self.psi11.proj()]

		sol00 = mesolve(H, self.psi00, tlist, c_ops=self.c_ops, e_ops=e_ops)
		sol01 = mesolve(H, self.psi01, tlist, c_ops=self.c_ops, e_ops=e_ops)
		sol10 = mesolve(H, self.psi10, tlist, c_ops=self.c_ops, e_ops=e_ops)
		sol11 = mesolve(H, self.psi11, tlist, c_ops=self.c_ops, e_ops=e_ops)

		fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(16, 8))
		axs = np.ndarray.flatten(axs)
		labels = ['00', '01', '10', '11']
		expectations = [sol00.expect, sol01.expect, sol10.expect, sol11.expect]

		for ax, exp, title in zip(axs, expectations, labels):
			for i, label in enumerate(labels):
				ax.plot(tlist, exp[i], label=label)
			ax.legend()
			ax.set_title(title)
		plt.show()

	@staticmethod
	def plot_pulse(x, tlist):
		x_re = x[:len(x) // 2]
		x_im = x[len(x) // 2:]
		bin_len = int(2 * len(tlist) / len(x))
		y_re = np.repeat(x_re, bin_len)
		y_im = np.repeat(x_im, bin_len)

		fig, ax = plt.subplots(figsize=(16, 8))
		ax.plot(tlist, y_re)
		ax.plot(tlist, y_im)
		plt.show()
