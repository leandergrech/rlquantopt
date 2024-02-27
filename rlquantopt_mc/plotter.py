import json

import numpy as np
from matplotlib import pyplot as plt
from qutip import mesolve, Qobj

from rlquantopt_mc.utils import Decoder, cubic_spline_with_zeroes


def plot_population_dynamics(H0, H1, basis_states, tlist, x, time_slots):
	basis_states = [Qobj(s) for s in basis_states]
	H = [Qobj(H0), [Qobj(H1), cubic_spline_with_zeroes(x, time_slots)(tlist)]]

	e_ops = [basis_states[0].proj(), basis_states[1].proj()]

	sol0 = mesolve(H, basis_states[0], tlist, e_ops=e_ops)
	sol1 = mesolve(H, basis_states[1], tlist, e_ops=e_ops)

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


def plot_pulse(tlist, x, time_slots):
	fig, ax = plt.subplots(figsize=(16, 8))
	ax.plot(tlist, cubic_spline_with_zeroes(x, time_slots)(tlist))
	plt.show()


if __name__ == "__main__":
	with open('data/26_02_2024_15_32_05.json', 'r') as f:
		data = json.load(f, cls=Decoder)

	H0 = data['data']['H0']
	H1 = data['data']['H1']
	basis_states = data['data']['basis_states']
	tlist = data['data']['tlist']
	params = data['result']['params']
	time_slots = data['data']['time_slots']

	plot_population_dynamics(H0, H1, basis_states, tlist, params, time_slots)
