"""Tunable-coupler transmon model in JAX, restricted to fixed excitation-number sectors.

The model is the one of rlquantopt.rl_envs.zcqubits.ZCQubits (paper eq. 1):
two fixed-frequency transmons (q0, q1) coupled to a tunable coupler (c), each
truncated to 3 levels, in the rotating frame at ``omega_r``. Frequencies are in
GHz, energies in rad/ns, time in ns. The control u(t) multiplies b†b and is in
rad/ns.

The Hamiltonian conserves N = a0†a0 + a1†a1 + b†b, so the computational basis
states only explore the N=1 sector (|001>, |010>, |100>) and the N=2 sector
(|002>, |011>, |020>, |101>, |110>, |200>), with kets ordered |q0 q1 c>.
Propagating these 3x3 and 6x6 blocks exactly is equivalent to evolving the
full 27-dimensional system.
"""
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp

from rlquantopt.jx.config import cdtype, fdtype

N_LEVELS = 3
FULL_DIM = N_LEVELS ** 3

# Full-space indices (9*q0 + 3*q1 + c) of the sector basis states.
SECTOR1_IDX = np.array([1, 3, 9])                 # |001>, |010>, |100>
SECTOR2_IDX = np.array([2, 4, 6, 10, 12, 18])     # |002>, |011>, |020>, |101>, |110>, |200>
# Positions of the computational states inside their sector.
S1_POS_010, S1_POS_100 = 1, 2
S2_POS_110 = 4


class ModelParams(NamedTuple):
    """Physical parameters, GHz. Defaults are setup_ZCQubits4MKrauss_params (the paper run).

    Plain floats/tuples so the params can sit in a static (hashable) config; any field
    may be replaced by a traced array, e.g. ``params._replace(omega_s=omega)``.
    """
    omega_s: tuple = (5.0311, 5.8899)
    alpha_s: tuple = (-324e-3, -235e-3)
    g: tuple = (100e-3, 71.4e-3)
    alpha_c: float = -230e-3
    omega_r: float = 6.0
    omega_c_0: float = 7.445


class SectorHamiltonian(NamedTuple):
    """H_N(u) = drift + u * diag(control) for the N=1 and N=2 sectors."""
    drift1: jnp.ndarray     # (3, 3) complex
    control1: jnp.ndarray   # (3,) real, diagonal of b†b
    drift2: jnp.ndarray     # (6, 6) complex
    control2: jnp.ndarray   # (6,) real


def _ops():
    a = np.diag(np.sqrt(np.arange(1, N_LEVELS)), 1)
    eye = np.eye(N_LEVELS)
    kron3 = lambda x, y, z: np.kron(np.kron(x, y), z)
    a0 = kron3(a, eye, eye)
    a1 = kron3(eye, a, eye)
    b = kron3(eye, eye, a)
    return a0, a1, b


_A0, _A1, _B = _ops()


def full_hamiltonian(params: ModelParams):
    """Drift and control operators on the full 27-dim space (for cross-checks and sectors)."""
    two_pi = 2 * jnp.pi
    n = lambda op: op.conj().T @ op
    anh = lambda op: op.conj().T @ op.conj().T @ op @ op
    drift = (two_pi * (params.omega_c_0 - params.omega_r) * n(_B)
             + jnp.pi * params.alpha_c * anh(_B))
    for m, a in enumerate((_A0, _A1)):
        drift = drift + (two_pi * (params.omega_s[m] - params.omega_r) * n(a)
                         + jnp.pi * params.alpha_s[m] * anh(a)
                         + two_pi * params.g[m] * (a @ _B.conj().T + a.conj().T @ _B))
    control = jnp.asarray(n(_B))
    return jnp.asarray(drift, cdtype()), jnp.asarray(control, fdtype())


# --8<-- [start:sector_hamiltonian]
def sector_hamiltonian(params: ModelParams) -> SectorHamiltonian:
    drift, control = full_hamiltonian(params)
    blk = lambda M, idx: M[np.ix_(idx, idx)]
    return SectorHamiltonian(
        drift1=blk(drift, SECTOR1_IDX), control1=jnp.diag(control)[SECTOR1_IDX],
        drift2=blk(drift, SECTOR2_IDX), control2=jnp.diag(control)[SECTOR2_IDX],
    )
# --8<-- [end:sector_hamiltonian]


# --8<-- [start:propagate]
def _expm_herm(H, dt):
    """exp(-i H dt) for Hermitian H via eigendecomposition (exact, GPU friendly)."""
    w, v = jnp.linalg.eigh(H)
    return (v * jnp.exp(-1j * w * dt)[..., None, :]) @ v.conj().swapaxes(-1, -2)


class SectorState(NamedTuple):
    """Evolved computational states. U1 columns are the images of the N=1 basis; psi2 is the image of |110>."""
    U1: jnp.ndarray     # (3, 3) complex
    psi2: jnp.ndarray   # (6,) complex


def initial_state() -> SectorState:
    return SectorState(U1=jnp.eye(3, dtype=cdtype()),
                       psi2=jnp.zeros(6, cdtype()).at[S2_POS_110].set(1.0))


def propagate(h: SectorHamiltonian, state: SectorState, amps, dt) -> SectorState:
    """Apply piecewise-constant amplitudes ``amps`` (rad/ns), each held for ``dt`` ns."""
    def body(s, u):
        P1 = _expm_herm(h.drift1 + jnp.diag(u * h.control1), dt)
        P2 = _expm_herm(h.drift2 + jnp.diag(u * h.control2), dt)
        return SectorState(P1 @ s.U1, P2 @ s.psi2), None
    state, _ = jax.lax.scan(body, state, jnp.asarray(amps))
    return state
# --8<-- [end:propagate]


# --8<-- [start:realised_gate]
def realised_gate(state: SectorState):
    """4x4 matrix G[i, j] = <psi_i(t)|b_j> in the basis |00>, |01>, |10>, |11> (v1 convention).

    |000> is an eigenstate with eigenvalue 0, so G[0, 0] = 1.
    """
    U1, psi2 = state
    i1, i2 = S1_POS_010, S1_POS_100
    V = jnp.array([[U1[i1, i1], U1[i2, i1]],
                   [U1[i1, i2], U1[i2, i2]]]).conj()
    e = psi2[S2_POS_110].conj()
    G = jnp.zeros((4, 4), cdtype())
    G = G.at[0, 0].set(1.0).at[1:3, 1:3].set(V).at[3, 3].set(e)
    return G
# --8<-- [end:realised_gate]


def sector_amplitudes(state: SectorState):
    """The 12 amplitudes v1 puts in the observation: N=1 images of |010>, |100>, then the N=2 image of |110>."""
    return jnp.concatenate([state.U1[:, S1_POS_010], state.U1[:, S1_POS_100], state.psi2])


# --8<-- [start:measured]
# Experimentally measurable observables (idea i06): what a lab reads out after preparing an input state
# and playing the pulse so far. The coupler is not read out (traced out); each transmon is read out in
# three levels, so leakage to |2> is visible.
_A = np.zeros((3, 3)); _A[0, 0] = _A[1, 1] = 1
_PX = np.zeros((3, 3)); _PX[0, 1] = _PX[1, 0] = 1
_PY = np.zeros((3, 3), complex); _PY[0, 1], _PY[1, 0] = -1j, 1j
_PZ = np.zeros((3, 3)); _PZ[0, 0], _PZ[1, 1] = 1, -1
PAULIS_2Q = np.stack([np.kron(a, b) for a in (_A, _PX, _PY, _PZ) for b in (_A, _PX, _PY, _PZ)])   # (16, 9, 9)
COMP_IDX = np.array([0, 1, 3, 4])            # |00>, |01>, |10>, |11> among the 9 qubit states |q0 q1>
N_MEASURED = 3 * 5 + 3 * 16


def _full(idx, v):
    return jnp.zeros(FULL_DIM, v.dtype).at[idx].set(v)


def _qubit_rho(psi):
    """Reduced density matrix of the two transmons (9x9), coupler traced out."""
    p = psi.reshape(3, 3, 3)
    return jnp.einsum("abc,dec->abde", p, p.conj()).reshape(9, 9)


def measured_observables(state: SectorState):
    """63 numbers: for inputs |01>, |10>, |11> the readout probabilities of 00, 01, 10, 11 and of any
    transmon in |2>; for inputs |0+>, |+0>, |+1> the 16 two-qubit Pauli expectations (I on the
    computational levels only), which carry the relative phases that populations cannot see."""
    U1, psi2 = state
    psi01 = _full(SECTOR1_IDX, U1[:, S1_POS_010])
    psi10 = _full(SECTOR1_IDX, U1[:, S1_POS_100])
    psi11 = _full(SECTOR2_IDX, psi2)
    e000 = jnp.zeros(FULL_DIM, U1.dtype).at[0].set(1.0)
    pops = []
    for psi in (psi01, psi10, psi11):
        p = jnp.real(jnp.diag(_qubit_rho(psi)))[COMP_IDX]
        pops.append(jnp.concatenate([p, jnp.atleast_1d(1 - p.sum())]))
    paulis = []
    for psi in ((e000 + psi01) / np.sqrt(2), (e000 + psi10) / np.sqrt(2), (psi01 + psi11) / np.sqrt(2)):
        rho = _qubit_rho(psi)
        paulis.append(jnp.real(jnp.einsum("kij,ji->k", PAULIS_2Q, rho)))
    return jnp.concatenate(pops), jnp.concatenate(paulis)
# --8<-- [end:measured]
