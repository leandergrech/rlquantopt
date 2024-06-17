from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt
from qutip import mesolve
from qutip.core import destroy, identity, tensor, Qobj, ket
from scipy.optimize import minimize
import pandas as pd


class ZCQubits:
    """
    test
    """

    def __init__(self, num_qubits, qubit_dims=None, coupler_dims=None, **params):
        self.num_qubits = num_qubits
        self.qubit_dims = qubit_dims if qubit_dims is not None else [3] * num_qubits
        self.coupler_dims = coupler_dims if coupler_dims is not None else 3
        self.params = {
            "omega_s": [6] * self.num_qubits,
            "alpha_s": [-300e-3] * self.num_qubits,
            "g": [70e-3] * self.num_qubits,
            "alpha_c": -230e-3,
            "omega_r": 0.0,
            "omega_c_0": 7.445
        }
        self.params.update(deepcopy(params))
        self._drift = []
        self._set_up_drift()
        self._control = None
        self._set_up_control()

        self.tlist = None
        self.x = None
        self.F = None
        self.iter = None
        self.nfev = None
        self.initial_states = None
        self.target_states = None
        self.num_params = None

    def _set_up_drift(self):
        destroy_op_tb = destroy(self.coupler_dims)
        # Coupler drift self interaction
        l = [identity(self.qubit_dims[m]) for m in range(self.num_qubits)]
        l.append(-2 * np.pi * self.params['omega_r'] * destroy_op_tb.dag() * destroy_op_tb +
                 np.pi * self.params["alpha_c"] * destroy_op_tb.dag() ** 2 * destroy_op_tb ** 2)
        self._drift.append(tensor(*l))
        # Qubit drift
        for m in range(self.num_qubits):
            destroy_op = destroy(self.qubit_dims[m])
            # qubit self interaction
            l = [identity(self.qubit_dims[m]) for m in range(self.num_qubits)]
            l.append(identity(self.coupler_dims))
            l[m] = (-2 * np.pi * self.params["omega_r"] * destroy_op.dag() * destroy_op +
                    np.pi * self.params["alpha_s"][m] * destroy_op.dag() ** 2 * destroy_op ** 2)
            self._drift.append(tensor(*l))
            # coupler - qubit interaction
            coeff = 2 * np.pi * self.params["g"][m]
            l = [identity(self.qubit_dims[m]) for m in range(self.num_qubits)]
            l.append(identity(self.coupler_dims))
            l[m] = destroy_op
            l[-1] = destroy_op_tb.dag()
            self._drift.append(coeff * tensor(*l))
            l[m] = destroy_op.dag()
            l[-1] = destroy_op_tb
            self._drift.append(coeff * tensor(*l))

        self._drift = sum(self._drift)

    def _set_up_control(self):
        destroy_op_tb = destroy(self.coupler_dims)
        l = [identity(self.qubit_dims[m]) for m in range(self.num_qubits)]
        l.append(destroy_op_tb.dag() * destroy_op_tb)
        self._control = 2 * np.pi * self.params['omega_c_0'] * tensor(*l)

    def mesolve(self, x, tlist, initial_states):
        bin_len = int(len(tlist) / len(x))
        H = [self._drift, [self._control, np.repeat(x, bin_len)]]

        results = [mesolve(H, state, tlist) for state in initial_states]

        return results

    @staticmethod
    def fidelity(states, target_states):
        return np.mean([np.abs(s.overlap(t)) ** 2 for s, t in zip(states, target_states)])

    def run(self, tlist, initial_states, target_states, num_params, x0=None):
        self.tlist = tlist
        self.initial_states = initial_states
        self.target_states = target_states
        self.num_params = num_params
        self.x = x0 if x0 is not None else np.random.uniform(-1, 1, self.num_params)

        self.iter = 0
        self.nfev = 0

        result = minimize(fun=self.cost_fun, method='Nelder-Mead', x0=self.x, callback=self.callback)

        data = pd.DataFrame([self.tlist, np.repeat(self.x, int(len(self.tlist) / len(self.x)))]).transpose()
        data.to_csv('data.csv')

        self.plot_pulse(result.x, self.tlist)
        self.plot_population_dynamics(result.x, self.tlist)

        return result

    def cost_fun(self, x, *args):
        self.nfev += 1
        self.x = x

        results = self.mesolve(self.x, self.tlist, self.initial_states)
        states = [res.states[-1] for res in results]

        self.F = self.fidelity(states, self.target_states)

        # return -np.log10(1 - self.F)
        return 1 - self.F

    def callback(self, x):
        self.iter += 1
        print(self.iter, self.nfev, self.F)

    def plot_population_dynamics(self, x, tlist):
        bin_len = int(len(tlist) / len(x))
        H = [self._drift, [self._control, np.repeat(x, bin_len)]]

        e_ops = [s.proj() for s in self.initial_states]

        expectations = [mesolve(H, s, tlist, e_ops=e_ops).expect for s in self.initial_states]

        fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(16, 8))
        axs = np.ndarray.flatten(axs)
        labels = ['00', '01', '10', '11']

        for ax, exp, title in zip(axs, expectations, labels):
            for i, label in enumerate(labels):
                ax.plot(tlist, exp[i], label=label)
            ax.legend()
            ax.set_title(title)
        fig.savefig('population.pdf')

    # plt.show()

    @staticmethod
    def plot_pulse(x, tlist):
        bin_len = int(len(tlist) / len(x))
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(tlist, np.repeat(x, bin_len))
        fig.savefig('pulse.pdf')
    # plt.show()


if __name__ == "__main__":
    num_qubits = 2  # KEEP 2
    qubit_dims = [3, 3]
    coupler_dims = 3

    full_dims = qubit_dims.copy()
    full_dims.append(coupler_dims)

    psi00 = ket((0, 0, 0), dim=full_dims)
    psi01 = ket((0, 1, 0), dim=full_dims)
    psi10 = ket((1, 0, 0), dim=full_dims)
    psi11 = ket((1, 1, 0), dim=full_dims)

    basis_states = [psi00, psi01, psi10, psi11]

    unitary = Qobj(np.array([
        [1, 0, 0, 0],
        [0, 0, 1j, 0],
        [0, 1j, 0, 0],
        [0, 0, 0, 1]
    ]), dims=[[2, 2], [2, 2]])

    mapped_basis_states = [sum(unitary[i, j] * basis_states[i]
                               for i in range(unitary.shape[0])) for j in range(unitary.shape[1])]
    # Lots of gates just rearrange the basis states, and we can avoid some complexity by identifying
    # this and setting the mapped_basis_states to the identical objects as the original basis_states
    for i, state in enumerate(mapped_basis_states):
        for j, basis_state in enumerate(basis_states):
            if state == basis_state:
                mapped_basis_states[i] = basis_state

    params = {
        "omega_s": [5.8899, 5.0311],
        "alpha_s": [-324e-3, -235e-3],
        "g": [100e-3, 71.4e-3]
    }

    model = ZCQubits(num_qubits, qubit_dims=qubit_dims, coupler_dims=coupler_dims, **params)

    data = pd.read_csv('configs/data.csv')

    results = model.mesolve(data['1'].to_numpy(), data['0'].to_numpy(), basis_states)
    states = [res.states[-1] for res in results]

    print(model.fidelity(states, mapped_basis_states))

# tlist = np.linspace(0, 200, 400)
# num_params = 5

# model.run(tlist, basis_states, mapped_basis_states, num_params)