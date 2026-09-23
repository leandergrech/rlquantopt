from itertools import product

import numpy as np
from krotov.functionals import F_avg
from qutip import mesolve, Qobj

from scipy.linalg import eigh
from scipy.optimize import minimize

from rlquantopt_mc.plotter import plot_pulse, plot_population_dynamics
from utils import save_result, cubic_spline_with_zeroes


class OneQubit:
	def __init__(self, gate, tlist, nstates=8):
		self.tlist = tlist
		self.nstates = nstates
		self.H0, self.H1 = self.transmon_hamiltonian()
		self.eigenvectors = self.logical_basis(self.H0)
		self.basis_states = [Qobj(self.eigenvectors[:, 0]), Qobj(self.eigenvectors[:, 1])]
		self.full_liouville_basis = [psi * phi.dag() for psi, phi in product(self.basis_states, self.basis_states)]

		if not isinstance(gate, Qobj):
			self.unitary = Qobj(gate)
		else:
			self.unitary = gate

		self.min_kwargs = None
		self.nfev = 0
		self.iter = 0
		self.H = None
		self.time_slots = None
		self.x0 = None
		self.x = None
		self.F = None

	def transmon_hamiltonian(self, Ec=0.386, EjEc=45, ng=0.):
		Ej = EjEc * Ec
		n = np.arange(-self.nstates, self.nstates + 1)
		up = np.diag(np.ones(2 * self.nstates), k=-1)
		do = up.T
		H0 = Qobj(np.diag(4 * Ec * (n - ng) ** 2) - Ej * (up + do) / 2.)
		H1 = Qobj(-2 * np.diag(n))

		return H0, H1

	@staticmethod
	def logical_basis(H):
		eigenvals, eigenvecs = eigh(H.full())
		ndx = np.argsort(eigenvals.real)
		return eigenvecs[:, ndx]

	def run(self, x0, method=None, min_kwargs=None, save_result=True):
		if min_kwargs is None:
			self.min_kwargs = dict()
		else:
			self.min_kwargs = min_kwargs
		self.time_slots = np.linspace(0, self.tlist[-1], len(x0) + 2)
		self.x0 = x0
		plot_pulse(self.tlist, self.x0, self.time_slots)
		plot_population_dynamics(self.H0, self.H1, self.basis_states, self.tlist, self.x0, self.time_slots)
		self.nfev = 0
		self.iter = 0
		# Minimize
		result = minimize(fun=self.cost_fun, x0=self.x0, method=method, options=min_kwargs, callback=self.callback)

		if save_result:
			self._save_result(result)

		plot_pulse(self.tlist, result.x, self.time_slots)
		plot_population_dynamics(self.H0, self.H1, self.basis_states, self.tlist, self.x, self.time_slots)

		return result

	def gate_fidelity(self, x):
		self.time_slots = np.linspace(0, self.tlist[-1], (len(x) + 2))
		self.H = [self.H0, [self.H1, cubic_spline_with_zeroes(x, self.time_slots)(self.tlist)]]

		args_list = [(self.H, state, self.tlist) for state in self.full_liouville_basis]

		results = [self.wrapped_mesolve(args) for args in args_list]

		return F_avg(results, self.basis_states, self.unitary, prec=1e-4)

	@staticmethod
	def wrapped_mesolve(args):
		H, psi, tlist = args
		sol = mesolve(H, psi, tlist)
		return sol.all_states[-1]

	# noinspection PyUnusedLocal,PyTypeChecker
	def cost_fun(self, x, *args):
		self.nfev += 1
		self.x = x

		self.H = [self.H0, [self.H1, cubic_spline_with_zeroes(self.x, self.time_slots)(self.tlist)]]

		args_list = [(self.H, state, self.tlist) for state in self.basis_states]

		results = [self.wrapped_mesolve(args) for args in args_list]

		self.F = 0.5 * (np.abs(results[0].overlap(self.basis_states[1])) ** 2 + np.abs(
			results[1].overlap(self.basis_states[0])) ** 2)

		return 1 - self.F

	# noinspection PyUnusedLocal
	def callback(self, x):
		self.iter += 1
		print(self.iter, self.nfev, self.F)

	def _save_result(self, result):
		folder = f'data'
		data = dict(
			data=dict(
				H0=self.H0.get_data().todense(),
				H1=self.H1.get_data().todense(),
				basis_states=[s.get_data().todense() for s in self.basis_states],
				nstates=self.nstates,
				tlist=self.tlist,
				time_slots=self.time_slots,
				unitary=self.unitary.get_data().todense(),
			),
			result=dict(
				params=result.x,
				fun=result.fun,
				nfev=result.nfev,
				nit=result.nit
			)
		)

		save_result(data, folder)
