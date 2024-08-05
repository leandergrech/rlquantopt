import os.path
from copy import deepcopy
# from functools import partial
# from multiprocessing import Pool
from typing import Callable
import numpy as np
from matplotlib import pyplot as plt
from qutip import QobjEvo, SESolver, expect
from qutip.core import destroy, identity, tensor, Qobj, ket
from scipy.interpolate import interp1d, CubicSpline
from scipy.optimize import minimize
import pandas as pd
from scipy.linalg import sqrtm
from stable_baselines3 import PPO

from rlquantopt.utils.rl_utils import get_pulse_data


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


def setup_ZCQubits4MKrauss_params(model_params):
    if model_params is None:
        model_params = {}
    ret_model_params = {}
    ret_model_params["omega_s"] = model_params.get("omega_s", [6.0, 5.9])
    ret_model_params["alpha_s"] = model_params.get("alpha_s", [-290e-3, -310e-3])
    ret_model_params["g"] = model_params.get("g", [70e-3, 70e-3])
    ret_model_params["alpha_c"] = model_params.get("alpha_c", -200e-3)
    ret_model_params["omega_r"] = model_params.get("omega_r", 6.2)
    ret_model_params["omega_c_0"] = model_params.get("omega_c_0", 6.7)
    ret_model_params["n_levels"] = model_params.get("n_levels", 3)
    ret_model_params["num_qubits"] = model_params.get("num_qubits", 2)
    ret_model_params["coupler_dims"] = model_params.get("coupler_dims", 3)
    ret_model_params["qubit_dims"] = [ret_model_params["n_levels"]] * ret_model_params["num_qubits"]

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

        self.H = QobjEvo([self.drift, [self.control, lambda t, A: A]], args={'A': 0.}, order=0)

    def _set_up_drift(self):
        drift = []
        destroy_op_tb = destroy(self.coupler_dims)
        # Coupler drift self interaction
        l = [identity(self.qubit_dims[m]) for m in range(self.num_qubits)]
        l.append(2 * np.pi * (self.params['omega_c_0'] - self.params['omega_r']) * destroy_op_tb.dag() * destroy_op_tb +
                 np.pi * self.params["alpha_c"] * destroy_op_tb.dag() ** 2 * destroy_op_tb ** 2)
        drift.append(tensor(*l))
        # Qubit drift
        for m in range(self.num_qubits):
            destroy_op = destroy(self.qubit_dims[m])
            # qubit self interaction
            l = [identity(self.qubit_dims[m]) for m in range(self.num_qubits)]
            l.append(identity(self.coupler_dims))
            l[m] = (2 * np.pi * (self.params["omega_s"][m] - self.params["omega_r"]) * destroy_op.dag() * destroy_op +
                    np.pi * self.params["alpha_s"][m] * destroy_op.dag() ** 2 * destroy_op ** 2)
            drift.append(tensor(*l))
            # coupler - qubit interaction
            coeff = 2 * np.pi * self.params["g"][m]
            l = [identity(self.qubit_dims[m]) for m in range(self.num_qubits)]
            l.append(identity(self.coupler_dims))
            l[m] = destroy_op
            l[-1] = destroy_op_tb.dag()
            drift.append(coeff * tensor(*l))
            l[m] = destroy_op.dag()
            l[-1] = destroy_op_tb
            drift.append(coeff * tensor(*l))

        return sum(drift)

    def _set_up_control(self):
        destroy_op_tb = destroy(self.coupler_dims)
        l = [identity(self.qubit_dims[m]) for m in range(self.num_qubits)]
        l.append(destroy_op_tb.dag() * destroy_op_tb)
        # return 2 * np.pi * self.params['omega_c_0'] * tensor(*l)
        return tensor(*l)

    def control_func(self) -> Callable:
        _x = np.copy(self.x)
        _x = np.insert(_x, 0, 0)
        _x = np.append(_x, 0)
        time_slots = np.linspace(0, self.tlist[-1], len(_x))
        if self.interpolation == 'cubic':
            return CubicSpline(time_slots, _x, bc_type='clamped')
        return interp1d(time_slots, _x, kind=self.interpolation, bounds_error=False, fill_value=0.)

    def float_control_func(self, t: float) -> float:
        # Each time you call float_control_func for evaluating a pulse we have to set up _control_function
        return float(self._control_func(t))

    def run(self, tlist, initial_states, target_states, num_params, method='Nelder-Mead', x0=None, options=None,
            interpolation='cubic', order=0, csv_fn='data.csv'):
        self.tlist = tlist
        self.initial_states = initial_states
        self.target_states = target_states
        self.num_params = num_params
        self.x = x0 if x0 is not None else np.random.uniform(-1, 1, self.num_params)
        self.order = order
        self.interpolation = interpolation
        self.options = dict(store_final_state=True)
        options = dict() if options is None else options
        self.options.update(deepcopy(options))
        self.threads = len(self.initial_states)

        self._control_func = self.control_func()
        self.H = QobjEvo([self.drift, [self.control, self.float_control_func]], args=dict(), order=self.order)
        self.solver = SESolver(self.H, options=self.options)

        self.iter = 0
        self.nfev = 0

        self.plot_pulse('initial_pulse')
        self.plot_population_dynamics('initial_dynamics')

        # result = minimize(fun=self.cost_fun, method=method, x0=self.x, callback=self.callback, options=dict(maxiter=5))
        result = minimize(fun=self.cost_fun, method=method, x0=self.x, callback=self.callback)

        pd.DataFrame([self.tlist, self.control_func()(self.tlist)]).transpose().to_csv(csv_fn)

        self.plot_pulse('final_pulse')
        self.plot_population_dynamics('final_dynamics')

        return result

    # noinspection PyUnusedLocal
    def cost_fun(self, x, *args):
        self.nfev += 1
        self.x = x

        results = self.solve()

        states = [res.final_state for res in results]

        self.F = np.mean([fidelity(state, target) for state, target in zip(states, self.target_states)])

        # return -np.log10(1 - self.F)
        return 1 - self.F

    def run_solver(self, state, e_ops):
        return self.solver.run(state, self.tlist, e_ops=e_ops)

    # def solve(self, e_ops=None):
    #     self._control_func = self.control_func()
    #     # results = []
    #     # for s in self.initial_states:
    #     #     res = self.run_solver(s, e_ops)
    #     #     results.append(res)
    #     _run_solver = partial(self.run_solver, e_ops=e_ops)
    #     with Pool(self.threads) as pool:
    #         results = pool.map(_run_solver, self.initial_states)
    #     return results

    # noinspection PyUnusedLocal
    def callback(self, x):
        self.iter += 1
        print(self.iter, self.nfev, self.F)

    def plot_population_dynamics(self, filename=None):
        e_ops = [state.proj() for state in self.initial_states]
        results = self.solve(e_ops)
        expectations = [res.expect for res in results]

        fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(16, 8))
        axes = np.ndarray.flatten(axes)
        labels = ['00', '01', '10', '11']

        for ax, expec, title in zip(axes, expectations, labels):
            for i, label in enumerate(labels):
                ax.plot(self.tlist, expec[i], label=label)
            ax.legend()
            ax.set_title(title)

        if filename:
            fig.savefig(f'{filename}.pdf')

        plt.show()

    def plot_pulse(self, filename=None):
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(self.tlist, self.control_func()(self.tlist))

        _x = np.copy(self.x)
        _x = np.insert(_x, 0, 0)
        _x = np.append(_x, 0)
        ax.scatter(np.linspace(0, self.tlist[-1], len(_x)), _x)

        if filename:
            fig.savefig(f'{filename}.pdf')

        plt.show()


# def test():
#     num_qubits = 2  # KEEP 2
#     qubit_dims = [3, 3]
#     coupler_dims = 3
#
#     full_dims = qubit_dims.copy()
#     full_dims.append(coupler_dims)
#
#     psi00 = ket((0, 0, 0), dim=full_dims)
#     psi01 = ket((0, 1, 0), dim=full_dims)
#     psi10 = ket((1, 0, 0), dim=full_dims)
#     psi11 = ket((1, 1, 0), dim=full_dims)
#
#     basis_states = [psi00, psi01, psi10, psi11]
#
#     unitary = Qobj(np.array([
#         [1, 0, 0, 0],
#         [0, 0, 1j, 0],
#         [0, 1j, 0, 0],
#         [0, 0, 0, 1]
#     ]), dims=[[2, 2], [2, 2]])
#
#     target_states = [sum(unitary[i, j] * basis_states[i]
#                          for i in range(unitary.shape[0])) for j in range(unitary.shape[1])]
#     '''
#     # Lots of gates just rearrange the basis states, and we can avoid some complexity by identifying
#     # this and setting the mapped_basis_states to the identical objects as the original basis_states
#     for i, target_state in enumerate(target_states):
#         for j, basis_state in enumerate(basis_states):
#             if target_state == basis_state:
#                 target_states[i] = basis_state
#     '''
#     params = {
#         "omega_s": [5.8899, 5.0311],
#         "alpha_s": [-324e-3, -235e-3],
#         "g": [100e-3, 71.4e-3]
#     }
#
#     model = ZCQubits(num_qubits, qubit_dims=qubit_dims, coupler_dims=coupler_dims, **params)
#
#     pulse_length = 120
#     # rl_dir = f'../rl_agents/ZCQPEE{pulse_length}pl-PPO/09-06-24_172549_ZCQPEE{pulse_length}pl'
#     # rl_dir = f'../rl_agents/ZCQPEE{pulse_length}pl-PPO/10-06-24_143544_ZCQPEE{pulse_length}pl'
#     # rl_dir = f'../rl_agents/ZCQPEE{pulse_length}pl-PPO/13-06-24_115241_ZCQPEE{pulse_length}pl/best_model'
#     # rl_fn = f'rl_model_65000_steps.zip'
#     # rl_path = os.path.join(rl_dir, rl_fn)
#     # rl_name = os.path.splitext(rl_fn)[0]
#     # from rlquantopt.rl_envs.zc_qpee import ZCQPEE
#
#     # env = ZCQPEE(pulse_length)
#     # dt = env.dt
#
#     # rl_agent = PPO.load(rl_path)
#
#     # optimised_pulse_file = 'optimised_pulse_60params_200ns_201samples.csv'
#     # optimised_pulse_file = 'configs/mc_optimised_pulse.csv'
#     # optimised_pulse_file = '/home/leander/code/rlquantopt/rlquantopt/rl_agents/ZCQPEE120pl-PPO/09-06-24_172549_ZCQPEE120pl/pulses/rl_model_55000_steps.csv'
#     # par_dir = '/home/leander/code/rlquantopt/rlquantopt/rl_agents/ZCQPEE120pl-PPO/10-06-24_143544_ZCQPEE120pl/pulses/'
#     par_dir = '/home/leander/code/rlquantopt/rlquantopt/rl_agents/ZCQPEE120pl-PPO/13-06-24_115241_ZCQPEE120pl/best_model/pulses'
#     # optimised_pulse_file = os.path.join(par_dir, 'rl_model_35000_steps.csv')
#     optimised_pulse_file = os.path.join(par_dir, 'best_model.csv')
#     model_name = os.path.splitext(os.path.basename(optimised_pulse_file))[0]
#     model_dir = os.path.dirname(optimised_pulse_file)
#
#     data = pd.read_csv(optimised_pulse_file)
#     pulse = data['amplist'].to_numpy()
#     # pulse_length = len(pulse)
#     tlist = data['tlist'].to_numpy()
#     dt = tlist[1] - tlist[0]
#
#     fig, ax = plt.subplots()
#     ax.plot(tlist, pulse)
#     ax.set_title(f'RL pulse ({model_name}); {len(pulse)} pulse length; T=200ns')
#     fig.savefig(os.path.join(model_dir, f'{model_name}_pulse.pdf'))
#
#     """
#     RUNNING FULL PULSE AT ONCE, starting from each basis_state, respectively
#     """
#     H = QobjEvo([model.drift, [model.control, pulse]], args={}, tlist=tlist, order=0)
#     solver = SESolver(H, options=dict(store_final_state=True))
#
#     # noinspection PyUnresolvedReferences
#     e_ops = [state.proj() for state in target_states]
#     results = [solver.run(basis_state, tlist, e_ops=e_ops) for basis_state in basis_states]
#
#     states = [res.final_state for res in results]
#
#     f = [fidelity(s, t) for s, t in zip(states, target_states)]
#     expectations = [res.expect for res in results]
#
#     print('Full pulse')
#     print(f)
#
#
#     fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(16, 8))
#     axs = np.ndarray.flatten(axs)
#     labels = ['00', '01', '10', '11']
#     for ax, exp, title in zip(axs, expectations, labels):
#         for i, label in enumerate(labels):
#             ax.plot(tlist, exp[i], label=label)
#         ax.legend()
#         ax.set_title(title)
#     fig.suptitle('Full pulse')
#     fig.savefig(os.path.join(model_dir, f'{model_name}_full-pulse_populations.pdf'))
#
#     """
#     RUNNING PULSE STEP-WISE, starting from each basis_state, respectively. Each intermittent state is obtained from the previous step call
#     """
#     H = QobjEvo([model.drift, [model.control, lambda t, A: A]], args={'A': 0.})
#     solver = [SESolver(H) for _ in range(4)]
#
#     f = []
#     all_f = []
#     expectations = []
#     # s = None
#     s = basis_states.copy()
#     cur_amp = 0.
#     states = [[s[i].full()] for i in range(len(basis_states))]
#     for k, (basis_state, target_state) in enumerate(zip(basis_states, target_states)):
#         l = [[expect(e, basis_state)] for e in e_ops]
#         all_f.append([])
#         fid = 0.
#         for i, t in enumerate(tlist[1:]):
#             solver[k].start(s[k], 0)
#
#             # obs = np.concatenate([env.extract_current_state([states[k][-1] for k in range(len(basis_states))], state_only=True), [cur_amp]])
#             # action = rl_agent.predict(obs)[0]
#             # action = env.denorm_action(action)
#             # if i == 0:
#             #     act_delta = 0.
#             # else:
#             #     act_delta = pulse[i] - pulse[i - 1]
#             #
#             # cur_amp += act_delta
#             cur_amp = pulse[i]
#             s[k] = solver[k].step(dt, args={'A': cur_amp})
#             states[k].append(s[k].full().copy())
#             for j, e in enumerate(e_ops):
#                 l[j].append(expect(e, s[k]))
#             fid = fidelity(s[k], target_state)
#             all_f[-1].append(fid)
#         f.append(fid)
#         expectations.append(l)
#
#     print('Step pulse')
#     print(f)
#
#     fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(16, 8))
#     axs = np.ndarray.flatten(axs)
#     labels = ['|000⟩', '|010⟩', '|100⟩', '|110⟩']
#
#     for ax, exp, title in zip(axs, expectations, labels):
#         for i, label in enumerate(labels):
#             ax.plot(tlist, exp[i], label=label)
#         ax.legend()
#         ax.set_title(title)
#     fig.suptitle('Step pulse')
#     fig.savefig(os.path.join(model_dir, f'{model_name}_step-pulse_population.pdf'))
#
#     from matplotlib.animation import FuncAnimation
#     from matplotlib import cm
#     from matplotlib.colors import LogNorm
#     import matplotlib as mpl
#
#     fig, axs = plt.subplots(4, figsize=(15, 10))
#     # gs = mpl.gridspec.GridSpec(ncols=6, nrows=4)
#     # axs = [fig.add_subplot(gs[i, :]) for i in range(4)]
#
#     EPS = 1e-3
#     cmap = cm.CMRmap
#     norm = LogNorm(vmin=EPS, vmax=1)
#     ims = []
#     for k, ax in enumerate(axs):
#         ax.set_title(f'{labels[k]}')
#         s = np.clip(np.abs(states[k][0]).T, EPS, 1)
#         im = ax.imshow(s, animated=True, cmap=cmap, norm=norm, origin='upper')
#         ims.append(im)
#
#     fig.subplots_adjust(right=0.8)
#     ax_colorbar = fig.add_axes([0.85, 0.15, 0.05, 0.7])
#     fig.colorbar(ims[0], ax=ax_colorbar)
#
#     def init():
#         for k, ax in enumerate(axs):
#             s = np.clip(np.abs(states[k][0]).T, EPS, 1)
#             im = ax.imshow(s, animated=True, cmap=cmap, norm=norm, origin='upper')
#             ims[k] = im
#         return ims,
#
#     def animate(i):
#         fig.suptitle(f'Step pulse @ step {i}')
#         for k, ax in enumerate(axs):
#             s = np.clip(np.abs(states[k][i]).T, EPS, 1)
#             im = ax.imshow(s, animated=True, cmap=cmap, norm=norm, origin='upper')
#             ims[k] = im
#         return ims,
#
#     # plt.show()
#     frames = np.arange(len(states[0]))
#     ani = FuncAnimation(fig, animate, frames=frames, init_func=init, blit=False)
#     ffwriter = mpl.animation.FFMpegWriter(fps=15)
#     save_path = os.path.join(model_dir, f'zcqubits_{model_name}_render.mp4')
#     print(save_path)
#     ani.save(save_path, dpi=100, writer=ffwriter)

# def main():
#     num_qubits = 2  # KEEP 2
#     qubit_dims = [3, 3]
#     coupler_dims = 3
#
#     full_dims = qubit_dims.copy()
#     full_dims.append(coupler_dims)
#
#     psi00 = ket((0, 0, 0), dim=full_dims)
#     psi01 = ket((0, 1, 0), dim=full_dims)
#     psi10 = ket((1, 0, 0), dim=full_dims)
#     psi11 = ket((1, 1, 0), dim=full_dims)
#
#     basis_states = [psi00, psi01, psi10, psi11]
#
#     unitary = Qobj(np.array([
#         [1, 0, 0, 0],
#         [0, 0, 1j, 0],
#         [0, 1j, 0, 0],
#         [0, 0, 0, 1]
#     ]), dims=[[2, 2], [2, 2]])
#
#     mapped_basis_states = [sum(unitary[i, j] * basis_states[i]
#                                for i in range(unitary.shape[0])) for j in range(unitary.shape[1])]
#     # Lots of gates just rearrange the basis states, and we can avoid some complexity by identifying
#     # this and setting the mapped_basis_states to the identical objects as the original basis_states
#     for i, state in enumerate(mapped_basis_states):
#         for j, basis_state in enumerate(basis_states):
#             if state == basis_state:
#                 mapped_basis_states[i] = basis_state
#
#     params = {
#         "omega_s": [5.8899, 5.0311],
#         "alpha_s": [-324e-3, -235e-3],
#         "g": [100e-3, 71.4e-3]
#     }
#
#     model = ZCQubits(num_qubits, qubit_dims=qubit_dims, coupler_dims=coupler_dims, **params)
#
#     T = 400
#     nb_samples = 401
#     tlist = np.linspace(0, T, nb_samples)
#     num_params = 80
#     method = 'Nelder-Mead'
#     interpolation = 'nearest'
#     csv_fn = f'optimised_pulse_{num_params}params_{T}ns_{nb_samples}samples.csv'
#
#     result = model.run(tlist=tlist, initial_states=basis_states, target_states=mapped_basis_states,
#                        num_params=num_params, method=method, interpolation=interpolation, csv_fn=csv_fn)
#
#     print(result)


# def test_step(par_dir='/home/leander/code/rlquantopt/rlquantopt/rl_envs/configs', csv_file='data_lilmc.csv'):
#     num_qubits = 2  # KEEP 2
#     qubit_dims = [3, 3]
#     coupler_dims = 3
#
#     full_dims = qubit_dims.copy()
#     full_dims.append(coupler_dims)
#
#     psi00 = ket((0, 0, 0), dim=full_dims)
#     psi01 = ket((0, 1, 0), dim=full_dims)
#     psi10 = ket((1, 0, 0), dim=full_dims)
#     psi11 = ket((1, 1, 0), dim=full_dims)
#
#     basis_states = [psi00, psi01, psi10, psi11]
#
#     unitary = Qobj(np.array([
#         [1, 0, 0, 0],
#         [0, 0, 1j, 0],
#         [0, 1j, 0, 0],
#         [0, 0, 0, 1]
#     ]), dims=[[2, 2], [2, 2]])
#
#     target_states = [sum(unitary[i, j] * basis_states[i]
#                          for i in range(unitary.shape[0])) for j in range(unitary.shape[1])]
#     params = {
#         "omega_s": [5.8899, 5.0311],
#         "alpha_s": [-324e-3, -235e-3],
#         "g": [100e-3, 71.4e-3]
#     }
#
#     model = ZCQubits(num_qubits, qubit_dims=qubit_dims, coupler_dims=coupler_dims, **params)
#
#     # par_dir = '/home/leander/code/rlquantopt/rlquantopt/rl_agents/ZCQPEE120pl-PPO/13-06-24_115241_ZCQPEE120pl/best_model/pulses'
#     optimised_pulse_file = os.path.join(par_dir, csv_file)
#     model_name = os.path.splitext(os.path.basename(optimised_pulse_file))[0]
#     model_dir = os.path.dirname(optimised_pulse_file)
#
#
#     data = pd.read_csv(optimised_pulse_file)
#     pulse = data['amplist'].to_numpy()
#     # pulse_length = len(pulse)
#     tlist = data['tlist'].to_numpy()
#     dt = tlist[1] - tlist[0]
#
#     fig, ax = plt.subplots()
#     ax.plot(tlist, pulse)
#     ax.set_title(f'RL pulse ({model_name}); {len(pulse)} pulse length; T=200ns')
#     fig.savefig(os.path.join(model_dir, f'{model_name}_pulse_step.pdf'))
#
#     """
#         RUNNING PULSE STEP-WISE, starting from each basis_state, respectively. Each intermittent state is obtained from the previous step call
#         """
#     H = QobjEvo([model.drift, [model.control, lambda t, A: A]], args={'A': 0.})
#     solver = [SESolver(H) for _ in range(4)]
#     e_ops = [state.proj() for state in basis_states]
#
#     f = []
#     all_f = []
#     expectations = []
#     # s = None
#     s = basis_states.copy()
#     # cur_amp = 0.
#     states = [[s[i].full()] for i in range(len(basis_states))]
#     for k, (basis_state, target_state) in enumerate(zip(basis_states, target_states)):
#         l = [[expect(e, basis_state)] for e in e_ops]
#         all_f.append([])
#         fid = 0.
#         solver[k].start(s[k], 0)
#         for i, t in enumerate(tlist[1:]):
#             cur_amp = pulse[i]
#             t = tlist[i]
#             s_ = solver[k].step(t, args={'A': cur_amp})
#             s[k] = s_
#             states[k].append(s[k].full().copy())
#             for j, e in enumerate(e_ops):
#                 l[j].append(expect(e, s[k]))
#             fid = fidelity(s[k], target_state)
#             all_f[-1].append(fid)
#         f.append(fid)
#         expectations.append(l)
#
#     print('Step pulse')
#     print(f, np.mean(f))
#
#     fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(16, 8))
#     axs = np.ndarray.flatten(axs)
#     labels = ['|000⟩', '|010⟩', '|100⟩', '|110⟩']
#
#     for ax, exp, title in zip(axs, expectations, labels):
#         for i, label in enumerate(labels):
#             ax.plot(tlist, exp[i], label=label)
#         ax.legend()
#         ax.set_title(title)
#     fig.suptitle('Step pulse')
#     fig.savefig(os.path.join(model_dir, f'{model_name}_step-pulse_population.pdf'))

LABELS = ['|000⟩', '|010⟩', '|100⟩', '|110⟩']
def test_step(model_params, pulse_file='/home/leander/code/rlquantopt/rlquantopt/rl_envs/configs/data_lilmc.csv', save_dir=None, plot=True):
    # par_dir = '/home/leander/code/rlquantopt/rlquantopt/rl_agents/ZCQPEE120pl-PPO/13-06-24_115241_ZCQPEE120pl/best_model/pulses'
    pulse_name = os.path.splitext(os.path.basename(pulse_file))[0]
    pulse_dir = os.path.dirname(pulse_file)
    if save_dir is None:
        save_dir = pulse_dir
    tlist, pulse = get_pulse_data(pulse_file)
    print(f'T={tlist[-1]}')

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

    target_states = [sum(unitary[i, j] * basis_states[i]
                         for i in range(unitary.shape[0])) for j in range(unitary.shape[1])]

    model = ZCQubits(**model_params)

    """
        RUNNING PULSE STEP-WISE, starting from each basis_state, respectively. Each intermittent state is obtained from the previous step call
        """
    H = QobjEvo([model.drift, [model.control, lambda t, A: A]], args={'A': 0.})
    solver = [SESolver(H) for _ in range(4)]
    e_ops = [state.proj() for state in basis_states]

    f = []
    all_f = []
    expectations = []
    s = basis_states.copy()
    states = [[s[i]] for i in range(len(basis_states))]
    for k, (basis_state, target_state) in enumerate(zip(basis_states, target_states)):
        l = [[expect(e, basis_state)] for e in e_ops]
        all_f.append([])
        fid = 0.
        solver[k].start(s[k], 0)
        for i, t in enumerate(tlist[1:]):
            cur_amp = pulse[i]
            t = tlist[i]
            s_ = solver[k].step(t, args={'A': cur_amp})
            s[k] = s_
            states[k].append(s_)
            for j, e in enumerate(e_ops):
                l[j].append(expect(e, s_))
            fid = fidelity(s_, target_state)
            all_f[-1].append(fid)
        f.append(fid)
        expectations.append(l)

    print('Step pulse')
    res_str = f"Final fidelities={f}, Mean={np.mean(f)*100.:.2f}%"
    print(res_str)

    if plot:
        title = f'RL step pulse ({pulse_name})\npulse length = {len(pulse)} ; T={tlist[-1]}ns\n{res_str}'
        fig, ax = plt.subplots()
        ax.plot(tlist, pulse)
        ax.set_title(title)
        fig.savefig(os.path.join(save_dir, f'{pulse_name}_pulse-step.pdf'))

        fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(16, 8))
        fig.suptitle(res_str)
        axs = np.ndarray.flatten(axs)

        for ax, exp, ttl in zip(axs, expectations, LABELS):
            for i, label in enumerate(LABELS):
                ax.plot(tlist, exp[i], label=label)
            ax.legend()
            ax.set_title(ttl)
        fig.suptitle(title)
        fig.savefig(os.path.join(save_dir, f'{pulse_name}_population-step.pdf'))

    states = np.squeeze(np.array(states))
    return states, pulse, all_f


def test_full(model_params, pulse_file='/home/leander/code/rlquantopt/rlquantopt/rl_envs/configs/data_lilmc.csv', save_dir=None, plot=True):
    pulse_dir = os.path.dirname(pulse_file)
    pulse_name = os.path.splitext(os.path.basename(pulse_file))[0]
    if save_dir is None:
        save_dir = pulse_dir

    tlist, pulse = get_pulse_data(pulse_file)
    print(f'T={tlist[-1]}')

    full_dims = model_params["qubit_dims"].copy()
    full_dims.append(model_params["coupler_dims"])

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

    target_states = [sum(unitary[i, j] * basis_states[i]
                         for i in range(unitary.shape[0])) for j in range(unitary.shape[1])]

    model = ZCQubits(**model_params)

    """
    RUNNING FULL PULSE AT ONCE, starting from each basis_state, respectively
    """
    H = QobjEvo([model.drift, [model.control, pulse]], args={}, tlist=tlist, order=0)
    solver = SESolver(H, options=dict(store_final_state=True))

    # noinspection PyUnresolvedReferences
    e_ops = [state.proj() for state in target_states]
    results = [solver.run(basis_state, tlist, e_ops=e_ops) for basis_state in basis_states]

    states = [res.final_state for res in results]

    f = [fidelity(s, t) for s, t in zip(states, target_states)]
    expectations = [res.expect for res in results]

    print('Full pulse')
    res_str = f"Final fidelities={f}, Mean={np.mean(f)*100.:.2f}%"
    print(res_str)

    if plot:
        title = f'RL full pulse ({pulse_name})\npulse length = {len(pulse)} ; T={tlist[-1]}ns\n{res_str}'
        fig, ax = plt.subplots()
        ax.set_title(title)
        ax.plot(tlist, pulse)
        ax.set_title(f'RL pulse ({pulse_name}); {len(pulse)} pulse length; T=200ns')
        fig.savefig(os.path.join(save_dir, f'{pulse_name}_pulse-full.pdf'))

        fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(16, 8))
        fig.suptitle(res_str)
        axs = np.ndarray.flatten(axs)
        for ax, exp, ttl in zip(axs, expectations, LABELS):
            for i, label in enumerate(LABELS):
                ax.plot(tlist, exp[i], label=label)
            ax.legend()
            ax.set_title(ttl)
        fig.suptitle(title)
        fig.savefig(os.path.join(save_dir, f'{pulse_name}_population-full.pdf'))


if __name__ == "__main__":
    # pulse_file = '/home/leander/code/rlquantopt/rlquantopt/rl_agents/ZCQPEE120pl-PPO/20-06-24_120311_ZCQPEE120pl/best_model/pulses/best_model_T200ns_delta_mode_default_model_to_term.csv'
    pulse_file = '/rlquantopt/rl_envs/configs/data_lilmc/data_lilmc.csv'
    save_dir = os.path.splitext(pulse_file)[0]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    test_full(pulse_file=pulse_file, save_dir=save_dir)
    test_step(pulse_file=pulse_file, save_dir=save_dir)
    # main()
