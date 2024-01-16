import os
import numpy as np
import qiskit.pulse as pulse
import qiskit_dynamics.pulse as qdp
from qiskit_dynamics import Solver
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info.states import Statevector

from datetime import datetime
now = datetime.now
import matplotlib.pyplot as plt



def pulsate_gaussian(r=0.1, w=5., dt=0.222, amp=1., beta=2.):
    # # Strength of Rabi=rate in GHz
    # r = 0.1
    #
    # # Freq of the qubit transition in GHz
    # w = 5.
    #
    # # Sample rate of the backend in ns
    # dt = 0.222
    #
    # # Define gaussian envelope to have a pi rotation
    # amp = 1.
    area = 1
    sig = area*0.399128/r/amp
    T = 4*sig
    duration = int(T/dt)

    # 1.75 factor is used to approximately get sx gate.
    # Further "calibration" could be done to refine the pulse amplitude
    with pulse.build(name="sx-sy schedule") as xp:
        pulse.play(pulse.Drag(duration, amp/1.75, sig/dt, beta),
                  pulse.DriveChannel(0))
        pulse.shift_phase(np.pi/2, pulse.DriveChannel(0))
        pulse.play(pulse.Drag(duration, amp/1.75, sig/dt, beta),
                  pulse.DriveChannel(0))

    # xp.draw()

    plt.rcParams["font.size"] = 16

    # converter =qdp.InstructionToSignals(dt, carriers={"d0": w})

    # signals = converter.get_signals(xp)
    # fig, axs = plt.subplots(1, 2, figsize=(14, 4.5))
    # for ax, title in zip(axs, ["envelope", "signal"]):
    #     signals[0].draw(0, 2*T, 2000, title, axis=ax)
    #     ax.set_xlabel("Time (ns)")
    #     ax.set_ylabel("Amplitude")
    #     ax.set_title(title)
    #     ax.vlines(T, ax.get_ylim()[0], ax.get_ylim()[1], "k", ls="dashed")
    #
    # start = dt * 60
    # end = dt*80
    # axs[-1].set_xlim((start, end))
    # axs[-1].set_ylim((-0.1, 0.1))

    # construct operators
    X = Operator.from_label('X')
    Z = Operator.from_label('Z')

    drift = 2 * np.pi * w * Z/2
    operators = [2 * np.pi * r * X/2]

    # construct solver
    hamiltonian_solver = Solver(
        static_hamiltonian=drift,
        hamiltonian_operators=operators,
        rotating_frame=drift,
        rwa_cutoff_freq=2 * 5.0,
        hamiltonian_channels=['d0'],
        channel_carrier_freqs={'d0': w},
        dt=dt)

    # Start the qubit in its ground state
    y0 = Statevector([1., 0.])

    start = now()
    sol = hamiltonian_solver.solve(t_span=[0., 2*T], y0=y0, signals=xp,
                                   atol=1e-8, rtol=1e-8)
    end = now()
    print(f'Simulated in {end - start}')

    return dict(sol=sol, T=T)


def plot_populations(results, ax=None):
    sol = results['sol']
    T = results['T']

    pop0 = [psi.probabilities()[0] for psi in sol.y]
    pop1 = [psi.probabilities()[1] for psi in sol.y]

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sol.t, pop0, lw=3, label="Population in |0>")
    ax.plot(sol.t, pop1, lw=3, label="Population in |1>")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Population (probabilies)")
    ax.legend(frameon=False)
    ax.set_ylim([0, 1.05])
    ax.set_xlim([0, 2 * T])
    ax.vlines(T, 0, 1.05, "k", ls='dashed')

    return ax




plt.rcParams['font.size'] = 18

beta = 1.0
save_dir = f'gridsearch_beta[{beta:.1f}]_r[0.05,0.25, 0.05]_w[1,6,1]'
if not os.path.exists:
    os.makedirs(save_dir)

for r in np.arange(0.05, 0.26, 0.05):
    fig, axs = plt.subplots(3, 2, figsize=(20, 15))
    for w, ax in zip(np.arange(1, 7), np.ravel(axs)):
        print(f'r={r}, w={w}')
        # results = pulsate_gaussian(r=0.1, w=5., amp=1.0)
        start = now()
        results = pulsate_gaussian(r=r, w=w, amp=1.0, beta=beta)
        end = now()
        print(f'Done in {end - start}')
        ax = plot_populations(results, ax)
        ax.set_title(f'r={r}, w={w}')

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'r[{r}]_w[1,6].png'))
    # plt.show()
