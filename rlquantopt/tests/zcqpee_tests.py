import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from qutip import QobjEvo, SESolver, ket, Qobj

from build.lib.rlquantopt.utils.utils import generate_random_alphanumeric
from rlquantopt.rl_envs.zcqubits import ZCQubits, setup_ZCQubits4MKrauss_params, fidelity


model_params = setup_ZCQubits4MKrauss_params({})
n_levels = model_params.get("n_levels")
channel_label = 'z'
simulator = ZCQubits(**model_params) 

full_dims = model_params.get("qubit_dims").copy()
full_dims.append(model_params.get("coupler_dims"))
psi00 = ket((0, 0, 0), dim=full_dims)
psi01 = ket((0, 1, 0), dim=full_dims)
psi10 = ket((1, 0, 0), dim=full_dims)
psi11 = ket((1, 1, 0), dim=full_dims)
basis_states_ket = [psi00, psi01, psi10, psi11]
basis_states = basis_states_ket.copy()
initial_states = basis_states.copy()
step_states = basis_states.copy()

basis_states_str = ('$|000\\rangle$', '$|010\\rangle$','$|100\\rangle$','$|110\\rangle$',)

# Set up target states
def get_iswap_u():
    return Qobj(np.array([
        [1, 0, 0, 0],
        [0, 0, 1j, 0],
        [0, 1j, 0, 0],
        [0, 0, 0, 1]
    ]), dims=[[2, 2], [2, 2]])  # iSWAP

U = get_iswap_u()
mapped_basis_states = [sum(U[i, j] * basis_states[i]
                           for i in range(U.shape[0])) for j in range(U.shape[1])]
# Lots of gates just rearrange the basis states, and we can avoid some complexity by identifying this and setting the mapped_basis_states to the identical objects as the original basis_states
for i, state in enumerate(mapped_basis_states):
    for j, basis_state in enumerate(basis_states):
        if state == basis_state:
            mapped_basis_states[i] = basis_state
target_states = mapped_basis_states.copy()
e_ops = [s_.proj() for s_ in target_states]

pulse_path = '/home/leander/code/rlquantopt/rlquantopt/rl_envs/configs/data_mkrauss/good_guess_analytical_correct.csv'
data = pd.read_csv(pulse_path)
pulse = data['pulse'].to_numpy()
tlist = data['time'].to_numpy()

pulse_diff = np.insert(np.diff(pulse), 0, 0.)


def test_N(N, render=True):
    global step_states
    # Set up SE solvers
    solvers = [SESolver(simulator.H) for _ in range(len(basis_states))]
    for k, initial_state in enumerate(initial_states):
        solvers[k].start(initial_state, 0.)

    cur_idx = 0
    def forward_dynamics(amp_abs_denorm, step_states):
        """
        Requires cur_idx==0 during the first step.
        This is to calculate the pulse time correctly.
        """

        # Step through the simulator with new amplitude until next time-step in simulation
        # amps_cur = np.insert(amp_abs_denorm, 0, 0.)
        nonlocal cur_idx
        final_states = []
        tlist_ = tlist[:N]
        for k, s_ in enumerate(step_states):
            for i in range(N):
                t = tlist[cur_idx + i]    # Think that the first action needs the first time-step since the first is always amp=0 @t=0
                final_state = solvers[k].step(t, args={'A': amp_abs_denorm[i]})
            # H = QobjEvo([simulator.drift, [simulator.control, amp_abs_denorm]], tlist=tlist_, order=0)
            # solver = SESolver(H, options=dict(store_final_state=True))
            # result = solver.run(s_, tlist_, e_ops=e_ops)
            final_states.append(final_state)
            # final_states.append(result.final_state)
        cur_idx += N

        return final_states

    idx = 0
    fids = []
    while idx < len(tlist):
        action = pulse[idx:idx+N]
        step_states = forward_dynamics(action, step_states)
        f = np.mean([fidelity(s, t) for s, t in zip(step_states, target_states)])
        fids.append(f)
        idx += N

    tlist2 = np.linspace(tlist[0], tlist[-1], len(fids))
    if render:
        fig, axs = plt.subplots(2)
        fig.suptitle(f'N = {N}')
        ax = axs[0]
        t_max = tlist2[np.argmax(fids)]
        ax.set_title(f'Max fidelity = {max(fids)*100:.2f}%  @ {t_max:.2f}ns')
        ax.axvline(t_max, color='k', ls='--')
        ax.plot(tlist2, fids, c='k', label='Fidelity', marker='.')
        ax = axs[1]
        ax.plot(tlist, pulse, c='r', label='Pulse amplitude', marker='.')
        for ax in axs:
            ax.legend(loc='best')
        # plt.show()
    step_states = basis_states.copy()

from datetime import datetime as dt
from tqdm import trange
for N in (5, 50, 500):
    start = dt.now()
    its = 1
    for i in trange(its):
        test_N(N, render=i==0)
    duration = (dt.now() - start).total_seconds()/its
    print(f'N = {N}, duration={duration:.2f}s')

plt.show()