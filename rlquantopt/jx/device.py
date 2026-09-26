"""A realistic tunable-coupler device (idea i10): no rotating-wave approximation, direct qubit-qubit
coupling, a coupler tuned through an asymmetric-SQUID flux curve, couplings that scale with frequency.

Parameters are those of Sung et al., PRX 11, 021058 (2021), Table 1 (idle point): qubits 4.16 and
4.00 GHz, anharmonicities -220 / -210 MHz; coupler idle 5.45 GHz, tunable 3.7-6.7 GHz, anharmonicity
-90 MHz; g_1c = 72.5 MHz, g_2c = 71.5 MHz, g_12 = 5.0 MHz. The flux curve is the asymmetric-SQUID form
omega_c(Phi) = omega_max [cos^2(pi Phi) + d^2 sin^2(pi Phi)]^(1/4) (Koch et al. 2007) with omega_max
= 6.7 GHz and d = (3.7 / 6.7)^2 from the quoted tuning range (our assumption: the range is the full
SQUID swing), Phi in units of the flux quantum. Qubit-coupler couplings scale as sqrt(omega_c / omega_c,idle).

H = sum_k omega_k n_k + eta_k/2 n_k (n_k - 1) + sum g_kl (a_k + a_k^dag)(a_l + a_l^dag), 3 levels per mode,
kets |q1 q2 c>. Without the RWA excitation number is not conserved, but parity is (the coupling terms
change it by 0 or +-2), so H is block-diagonal in parity. States with more than 3 excitations are dropped
(they are ~2 omega away and reached only through counter-rotating terms); against the full 27-level
space this shifts the |11> phase by ~1e-3 rad over 30 ns and the fidelity by < 1e-5 (tests/test_jx_device.py). Blocks: even (N = 0, 2: 7 states), odd (N = 1, 3: 10 states).

The computational states are the dressed eigenstates of the idle Hamiltonian with the largest overlap on
|000>, |010>, |100>, |110>, and the gate is taken in the frame of the idle Hamiltonian (its dressed energies
removed), the usual convention; local phases are then absorbed by the free virtual-Z corrections.
"""
import itertools
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp

from rlquantopt.jx.config import cdtype, fdtype

LEVELS = 3
TWO_PI = 2 * np.pi


class DeviceParams(NamedTuple):
    """GHz; flux in flux quanta. Defaults: Sung et al. 2021, idle point."""
    omega_1: float = 4.16
    omega_2: float = 4.00
    eta_1: float = -0.220
    eta_2: float = -0.210
    omega_c_max: float = 6.7
    squid_d: float = (3.7 / 6.7) ** 2
    eta_c: float = -0.090
    g_1c: float = 0.0725
    g_2c: float = 0.0715
    g_12: float = 0.0050
    omega_c_idle: float = 5.45


def coupler_frequency(phi, p: DeviceParams):
    c, s = jnp.cos(jnp.pi * phi), jnp.sin(jnp.pi * phi)
    return p.omega_c_max * (c ** 2 + p.squid_d ** 2 * s ** 2) ** 0.25


def idle_flux(p: DeviceParams):
    """Flux (0..0.5) at which the coupler sits at omega_c_idle: closed form of the asymmetric-SQUID curve
    (the frequency falls monotonically on this branch)."""
    r4 = (p.omega_c_idle / p.omega_c_max) ** 4                      # cos^2 + d^2 sin^2 = r4
    s2 = (1 - r4) / (1 - p.squid_d ** 2)
    return float(np.arcsin(np.sqrt(s2)) / np.pi)


# --- basis and operators -------------------------------------------------------------------------------
STATES = [s for s in itertools.product(range(LEVELS), repeat=3) if sum(s) <= 3]       # (q1, q2, c)
EVEN = [s for s in STATES if sum(s) % 2 == 0]
ODD = [s for s in STATES if sum(s) % 2 == 1]


def _ops(block, rwa=False):
    """Number operators and the coupling operators (a_k + a_k^dag)(a_l + a_l^dag) restricted to a block;
    with ``rwa`` only the excitation-conserving part a_k a_l^dag + a_k^dag a_l."""
    idx = {s: i for i, s in enumerate(block)}
    n = len(block)
    num = np.zeros((3, n))
    for s, i in idx.items():
        num[:, i] = s
    cpl = {}
    for (k, l) in ((0, 2), (1, 2), (0, 1)):
        M = np.zeros((n, n))
        for s, i in idx.items():
            for dk in (-1, 1):
                for dl in (-1, 1):
                    if rwa and dk == dl:
                        continue
                    t = list(s)
                    t[k] += dk
                    t[l] += dl
                    t = tuple(t)
                    if t in idx and min(t) >= 0:
                        ak = np.sqrt(max(s[k], t[k]))       # <t|a or a^dag|s> for mode k
                        al = np.sqrt(max(s[l], t[l]))
                        M[idx[t], i] += ak * al
        cpl[(k, l)] = M
    return num, cpl


_EVEN_OPS, _ODD_OPS = _ops(EVEN), _ops(ODD)
_EVEN_OPS_RWA, _ODD_OPS_RWA = _ops(EVEN, rwa=True), _ops(ODD, rwa=True)


class BlockHamiltonian(NamedTuple):
    """Everything but the coupler frequency is fixed per device; H(phi) is assembled per sample."""
    diag_fixed_e: jnp.ndarray   # qubit energies (+ anharmonicities of all modes), even block
    diag_fixed_o: jnp.ndarray
    nc_e: jnp.ndarray           # coupler occupation, even / odd
    nc_o: jnp.ndarray
    cq_e: jnp.ndarray           # (a_q + a_q^dag)(a_c + a_c^dag) for q1 and q2 (stacked), even / odd
    cq_o: jnp.ndarray
    c12_e: jnp.ndarray
    c12_o: jnp.ndarray
    g: jnp.ndarray              # (g_1c, g_2c, g_12) at idle, rad/ns
    wc_idle: float
    params: DeviceParams
    lin_slope: float = 0.0      # simplified model: omega_c = omega_c,idle + lin_slope (phi - lin_phi0); 0 = SQUID curve
    lin_phi0: float = 0.0


def linearisation(p: DeviceParams):
    """(slope GHz per flux quantum, idle flux) of the SQUID curve at the nominal idle point."""
    phi0 = idle_flux(p)
    return float((coupler_frequency(phi0 + 1e-6, p) - coupler_frequency(phi0 - 1e-6, p)) / 2e-6), phi0


def block_hamiltonian(p: DeviceParams, simplified=False, lin=None):
    """``simplified``: the model one would write down first, for the sim-to-sim transfer test: rotating-wave
    approximation, no direct qubit-qubit coupling, the coupler frequency linear in flux around idle (the slope of
    the SQUID curve there), couplings independent of frequency. ``lin`` = linearisation(nominal device), so that
    a drifted device is simplified around the lab's nominal model."""
    def fixed(num):
        n1, n2, nc = num
        return TWO_PI * (p.omega_1 * n1 + p.omega_2 * n2 + p.eta_1 / 2 * n1 * (n1 - 1)
                         + p.eta_2 / 2 * n2 * (n2 - 1) + p.eta_c / 2 * nc * (nc - 1))
    (num_e, cpl_e), (num_o, cpl_o) = (_EVEN_OPS_RWA, _ODD_OPS_RWA) if simplified else (_EVEN_OPS, _ODD_OPS)
    slope, phi0 = (lin or linearisation(p)) if simplified else (0.0, 0.0)
    return BlockHamiltonian(
        diag_fixed_e=jnp.asarray(fixed(num_e)), diag_fixed_o=jnp.asarray(fixed(num_o)),
        nc_e=jnp.asarray(num_e[2]), nc_o=jnp.asarray(num_o[2]),
        cq_e=jnp.asarray(np.stack([cpl_e[(0, 2)], cpl_e[(1, 2)]])), cq_o=jnp.asarray(np.stack([cpl_o[(0, 2)], cpl_o[(1, 2)]])),
        c12_e=jnp.asarray(cpl_e[(0, 1)]), c12_o=jnp.asarray(cpl_o[(0, 1)]),
        g=TWO_PI * jnp.asarray([p.g_1c, p.g_2c, 0.0 if simplified else p.g_12]), wc_idle=p.omega_c_idle, params=p,
        lin_slope=slope, lin_phi0=phi0)


def h_blocks(h: BlockHamiltonian, phi):
    """(H_even, H_odd) in rad/ns at coupler flux phi."""
    lin = h.lin_slope != 0          # traced inside jit: evaluate both, select
    wc = jnp.where(lin, h.wc_idle + h.lin_slope * (phi - h.lin_phi0), coupler_frequency(phi, h.params))
    scale = jnp.where(lin, 1.0, jnp.sqrt(wc / h.wc_idle))
    out = []
    for diag, nc, cq, c12 in ((h.diag_fixed_e, h.nc_e, h.cq_e, h.c12_e), (h.diag_fixed_o, h.nc_o, h.cq_o, h.c12_o)):
        H = jnp.diag(diag + TWO_PI * wc * nc) + scale * (h.g[0] * cq[0] + h.g[1] * cq[1]) + h.g[2] * c12
        out.append(H)
    return out


def _expm_herm(H, dt):
    w, v = jnp.linalg.eigh(H)
    return (v * jnp.exp(-1j * w * dt)[None, :]) @ v.conj().T


class DeviceState(NamedTuple):
    """Images of the dressed computational states: even block columns (|00>, |11>), odd (|01>, |10>)."""
    even: jnp.ndarray   # (7, 2)
    odd: jnp.ndarray    # (10, 2)


def dressed_basis(h: BlockHamiltonian, phi_idle):
    """Dressed idle eigenstates closest to |000>, |110> (even) and |010>, |100> (odd), and their energies.
    Pure JAX, so it can run per episode inside jit (each drifted device has its own dressed states)."""
    He, Ho = h_blocks(h, phi_idle)
    out = []
    for H, block, targets in ((He, EVEN, [(0, 0, 0), (1, 1, 0)]), (Ho, ODD, [(0, 1, 0), (1, 0, 0)])):
        w, v = jnp.linalg.eigh(H)
        cols, ens = [], []
        for t in targets:
            i = block.index(t)
            j = jnp.argmax(jnp.abs(v[i, :]))
            vec = v[:, j] * jnp.exp(-1j * jnp.angle(v[i, j]))      # real positive on the bare state
            cols.append(vec)
            ens.append(w[j])
        out.append((jnp.stack(cols, 1).astype(cdtype()), jnp.stack(ens)))
    return out      # [(even vectors (7,2), energies (2,)), (odd vectors (10,2), energies (2,))]


def initial_state(basis):
    (ve, _), (vo, _) = basis
    return DeviceState(even=ve, odd=vo)


def propagate(h: BlockHamiltonian, state: DeviceState, phis, dt):
    """Piecewise-constant flux samples ``phis`` (flux quanta), each held for dt ns."""
    def body(s, phi):
        He, Ho = h_blocks(h, phi)
        return DeviceState(_expm_herm(He, dt) @ s.even, _expm_herm(Ho, dt) @ s.odd), None
    state, _ = jax.lax.scan(body, state, jnp.asarray(phis))
    return state


def gate(state: DeviceState, basis, t_total):
    """4x4 gate <d_i| e^{i H_idle t} U |d_j> on the dressed computational states |00>, |01>, |10>, |11>,
    in the physical convention (as metrics.physical_gate returns). Transitions between blocks are zero
    by parity, and within the odd block |01> <-> |10> is the iSWAP-type exchange."""
    (ve, ee), (vo, eo) = basis
    Ge = (ve.conj().T @ state.even) * jnp.exp(1j * ee * t_total)[:, None]     # rows |00>, |11>
    Go = (vo.conj().T @ state.odd) * jnp.exp(1j * eo * t_total)[:, None]      # rows |01>, |10>
    G = jnp.zeros((4, 4), cdtype())
    G = G.at[0, 0].set(Ge[0, 0]).at[0, 3].set(Ge[0, 1]).at[3, 0].set(Ge[1, 0]).at[3, 3].set(Ge[1, 1])
    return G.at[1:3, 1:3].set(Go)
