from copy import deepcopy
from typing import Callable
import numpy as np
from scipy.interpolate import interp1d, CubicSpline
from scipy.optimize import minimize
import pandas as pd
from scipy.linalg import sqrtm
from qutip import Qobj, ket, QobjEvo, SESolver, expect
from qutip.core import destroy, identity, tensor, Qobj, ket


# noinspection PyUnresolvedReferences
def fidelity(A, B) -> float:
    if not isinstance(A, Qobj):
        A = Qobj(A)
    if not isinstance(B, Qobj):
        B = Qobj(B)
    _A = A.full()
    _B = B.full()
    if A.isket:
        if B.isket:
            f = np.abs(_B.conj().T @ _A) ** 2
        elif B.isbra:
            f = np.abs(_B.conj() @ _A) ** 2
        else:
            f = (_A.conj().T @ _B @ _A).real
    elif A.isbra:
        if B.isket:
            f = np.abs(_A.conj() @ _B) ** 2
        elif B.isbra:
            f = np.abs(_B @ _A.conj().T) ** 2
        else:
            f = (_A.conj() @ _B @ _A.T).real
    else:
        if B.isket:
            f = (_B.conj().T @ _A @ _B).real
        elif B.isbra:
            f = (_B.conj() @ _A @ _B.T).real
        else:
            sqrtA = sqrtm(_A)
            f = np.trace(sqrtm(sqrtA @ _B @ sqrtA)).real ** 2

    return f[0, 0]

'''
The following is a MWE of the RL environment's reward function. It will be maximimised during training.
'''


class RLEnv():
    FID_THRESH = 0.99

    TV_PENALTY_SCALE = 0.1

    def _reward_function(self, step_states, acts=None):
        f = np.mean([fidelity(s, t) for s, t in zip(step_states, self.target_states)])
        rew = self._fid2rew(f)
        if f >= self.FID_THRESH:    # remove tv penalty upon successful pulse
            return rew, 0.0, f

        if acts is None:
            return rew, 0.0, f

        total_variation = np.sum(np.abs(np.diff(acts)))
        tv_penalty = total_variation * self.TV_PENALTY_SCALE
        rew -= tv_penalty

        return rew, tv_penalty, f

    @staticmethod
    def _fid2rew(fid):
        return -np.log10(1 - fid)

    def __init__(self):
        model_params = dict()
        self.model_params = setup_ZCQubits4MKrauss_params(**model_params)

        full_dims = self.model_params.get("qubit_dims").copy()
        full_dims.append(self.model_params.get("coupler_dims"))
        psi00 = ket((0, 0, 0), dim=full_dims)
        psi01 = ket((0, 1, 0), dim=full_dims)
        psi10 = ket((1, 0, 0), dim=full_dims)
        psi11 = ket((1, 1, 0), dim=full_dims)
        basis_states_ket = [psi00, psi01, psi10, psi11]
        self.basis_states = basis_states_ket.copy()
        self.initial_states = self.basis_states.copy()

        # Set up SE solvers

        self.simulator = ZCQubits(**self.model_params)
        self.solvers = [SESolver(self.simulator.H) for _ in range(len(self.basis_states))]

        self.basis_states_str = ('$|000\\rangle$', '$|010\\rangle$', '$|100\\rangle$', '$|110\\rangle$',)

        # Set up gate
        self.unitary_iswap = U = self.get_iswap_u()

        # Set up target states
        mapped_basis_states = [sum(U[i, j] * self.basis_states[i]
                                   for i in range(U.shape[0])) for j in range(U.shape[1])]
        # Lots of gates just rearrange the basis states, and we can avoid some complexity by identifying this and setting the mapped_basis_states to the identical objects as the original basis_states
        for i, state in enumerate(mapped_basis_states):
            for j, basis_state in enumerate(self.basis_states):
                if state == basis_state:
                    mapped_basis_states[i] = basis_state
        self.target_states = mapped_basis_states.copy()

    def get_iswap_u(self):
        return Qobj(np.array([
            [1, 0, 0, 0],
            [0, 0, 1j, 0],
            [0, 1j, 0, 0],
            [0, 0, 0, 1]
        ]), dims=[[2, 2], [2, 2]])  # iSWAP


def setup_ZCQubits4MKrauss_params(**model_params):
    # model_params = model_params if model_params is not None else {}
    ret_model_params = dict()
    ret_model_params["omega_s"] = model_params.get("omega_s", [6.0, 5.9])
    ret_model_params["alpha_s"] = model_params.get("alpha_s", [-290e-3, -310e-3])
    ret_model_params["g"] = model_params.get("g", [70e-3, 70e-3])
    ret_model_params["alpha_c"] = model_params.get("alpha_c", -200e-3)
    ret_model_params["omega_r"] = model_params.get("omega_r", 6.2)
    ret_model_params["omega_c_0"] = model_params.get("omega_c_0", 6.7)
    ret_model_params["n_levels"] = n_levels = model_params.get("n_levels", 3)
    ret_model_params["num_qubits"] = num_qubits = model_params.get("num_qubits", 2)
    ret_model_params["coupler_dims"] = model_params.get("coupler_dims", 3)
    ret_model_params["qubit_dims"] = [n_levels] * num_qubits

    return ret_model_params


class ZCQubits:
    def __init__(self, **params):
        self.params = {
            "omega_s": [5.8899, 5.0311],    # [6] * self.num_qubits,
            "alpha_s": [-324e-3, -235e-3],  # [-300e-3] * self.num_qubits,
            "g": [100e-3, 71.4e-3],         # [70e-3] * self.num_qubits,
            "alpha_c": -230e-3,
            "omega_r": 0.0,
            "omega_c_0": 7.445,
            "n_levels": 3,
            "coupler_dims": 3,
            "num_qubits": 2
        }
        self.params.update(deepcopy(params))

        self.n_levels = self.params["n_levels"]
        self.num_qubits = self.params["num_qubits"]
        self.qubit_dims = [self.n_levels] * self.num_qubits
        self.coupler_dims = self.params["coupler_dims"]

        self.drift = self._set_up_drift()
        self.control = self._set_up_control()

        self.tlist = None
        self.x = None
        self.num_params = None
        self.iter = None
        self.nfev = None
        self.F = None
        self.initial_states = None
        self.target_states = None
        self.options = None
        self.order = None
        self.interpolation = None
        self.H = None
        self.solver = None
        self._control_func = None
        self.threads = None

        # self.H = QobjEvo([self.drift, [self.control, lambda t, A: A]], args={'A': 0.}, order=0)
        self.H = QobjEvo([self.drift, [self.control, self.bubu]], args={'A': 0.}, order=0)

    @staticmethod
    def bubu(t, A):
        return A

    def _set_up_drift(self):
        drift = []
        b = destroy(self.coupler_dims)
        # Coupler drift self interaction
        l = [identity(self.qubit_dims[m]) for m in range(self.num_qubits)]
        l.append(2 * np.pi * (self.params['omega_c_0'] - self.params['omega_r']) * b.dag() * b +
                 np.pi * self.params["alpha_c"] * b.dag() ** 2 * b ** 2)
        drift.append(tensor(*l))
        # Qubit drift
        for m in range(self.num_qubits):
            a = destroy(self.qubit_dims[m])
            # qubit self interaction
            l = [identity(self.qubit_dims[m_]) for m_ in range(self.num_qubits)]
            # Meeting MKrauss 05/08/2024 changes
            l.append(identity(self.coupler_dims))
            l[m] = (2 * np.pi * (self.params["omega_s"][m] - self.params["omega_r"]) * a.dag() * a +
                    np.pi * self.params["alpha_s"][m] * a.dag() ** 2 * a ** 2)

            drift.append(tensor(*l))
            # coupler - qubit interaction
            coeff = 2 * np.pi * self.params["g"][m]
            l = [identity(self.qubit_dims[m_]) for m_ in range(self.num_qubits)]
            l.append(identity(self.coupler_dims))
            l[m] = a
            l[-1] = b.dag()
            drift.append(coeff * tensor(*l))
            l[m] = a.dag()
            l[-1] = b
            drift.append(coeff * tensor(*l))

        return sum(drift)

    def _set_up_control(self):
        b = destroy(self.coupler_dims)
        l = [identity(self.qubit_dims[m]) for m in range(self.num_qubits)]
        l.append(b.dag() * b)
        return tensor(*l)

