"""Gate metrics for excitation-preserving two-qubit gates, in closed form.

The realised gate is G = diag(g00, V, e) in the basis |00>, |01>, |10>, |11>,
with V = [[a, b], [c, d]]. For such a gate U·Ũ (Ũ = (σy⊗σy) Uᵀ (σy⊗σy)) is
diag(g00·e, V·[[d, b], [c, a]], g00·e), whose eigenvalues are g00·e (twice)
and (ad + bc) ± 2·sqrt(abcd). This lets us follow weylchamber.c1c2c3
(Childs et al., PRA 68, 052311) exactly without a general 4x4 eig, which JAX
does not provide on GPU.
"""
import jax
import jax.numpy as jnp

WEYL_DIGITS = 8     # weylchamber.prec.DEFAULT_WEYL_PRECISSION
_M = jnp.array([[1., 1., 0.], [1., 0., 1.], [0., 1., 1.]])


# --8<-- [start:weyl]
def weyl_eigenvalues(G):
    g00, a, b, c, d, e = G[0, 0], G[1, 1], G[1, 2], G[2, 1], G[2, 2], G[3, 3]
    p = a * d + b * c
    r = 2 * jnp.sqrt(a * b * c * d)
    ev = jnp.stack([g00 * e, p + r, p - r, g00 * e])
    det = g00 * e * (a * d - b * c)
    return ev / jnp.sqrt(det)


def c1c2c3(G, digits=WEYL_DIGITS):
    """Weyl chamber coordinates (units of π) of a block-diagonal 4x4 gate, as weylchamber.c1c2c3.

    ``digits=None`` skips the final rounding, which otherwise has zero gradient (use it for GRAPE).
    """
    two_S = jnp.angle(weyl_eigenvalues(G)) / jnp.pi
    two_S = jnp.where(two_S <= -0.5, two_S + 2.0, two_S)
    S = jnp.sort(two_S / 2.0)[::-1]
    n = jnp.round(jnp.sum(S)).astype(int)
    S = S - (jnp.arange(4) < n)
    S = jnp.roll(S, -n)
    c1, c2, c3 = _M @ S[:3]
    c1 = jnp.where(c3 < 0, 1 - c1, c1)
    c3 = jnp.abs(c3)
    c = jnp.stack([c1, c2, c3])
    return c if digits is None else jnp.round(c, digits) + 0.0
# --8<-- [end:weyl]


# --8<-- [start:cost]
def concurrence(c):
    """Gate concurrence from Weyl coordinates, as weylchamber.concurrence."""
    c1, c2, c3 = c
    in_pe = (c1 + c2 >= 0.5) & (c1 - c2 <= 0.5) & (c2 + c3 <= 0.5)
    rolled = jnp.roll(c, 1)
    m = jnp.concatenate([c - rolled, c + rolled])
    return jnp.where(in_pe, 1.0, jnp.max(jnp.abs(jnp.sin(jnp.pi * m))))


def unitarity(G):
    """U = Tr(G†G)/4: 1 when no population leaks out of the computational subspace."""
    return jnp.sum(jnp.abs(G) ** 2) / 4


def cost_JT(G, concurrence_weight=1.0, unitarity_weight=3.0, digits=WEYL_DIGITS):
    """J_T = 1 - (w_c C + w_u U)/(w_c + w_u); paper eq. 2 with the default 1:3 weights."""
    C = concurrence(c1c2c3(G, digits))
    U = unitarity(G)
    K = concurrence_weight + unitarity_weight
    return 1 - (concurrence_weight * C + unitarity_weight * U) / K, C, U
# --8<-- [end:cost]


# --8<-- [start:fidelity]
# Named-gate fidelity (idea i06). Target in the basis |00>, |01>, |10>, |11> (|q0 q1>).
SQRT_ISWAP = jnp.array([[1, 0, 0, 0],
                        [0, 1 / jnp.sqrt(2), 1j / jnp.sqrt(2), 0],
                        [0, 1j / jnp.sqrt(2), 1 / jnp.sqrt(2), 0],
                        [0, 0, 0, 1]], dtype=jnp.complex128)
_GRID = jnp.stack(jnp.meshgrid(*[jnp.linspace(-jnp.pi, jnp.pi, 10, endpoint=False)] * 3, indexing="ij"), -1).reshape(-1, 3)


def physical_gate(G):
    """<b_i|U|b_j> from ``physics.realised_gate``, which follows v1 and returns its conjugate transpose."""
    return G.conj().T


def _z_trace(P, V, th):
    """Tr(V† D_after P D_before) for block-diagonal P and V, as a function of the three phase combinations
    (x, y, z) = (a1 + b1, a1 + b0, a0 + b1) that the Z corrections D = diag(1, e^{i a1}, e^{i a0}, e^{i(a0 + a1)})
    before (b) and after (a) the gate can set; the global phase does not enter |Tr|."""
    t = jnp.conj(V) * P
    x, y, z = th[..., 0], th[..., 1], th[..., 2]
    e = lambda p: jnp.exp(1j * p)
    return (t[0, 0] + t[1, 1] * e(x) + t[1, 2] * e(y) + t[2, 1] * e(z) + t[2, 2] * e(y + z - x)
            + t[3, 3] * e(y + z))


def best_z_phases(P, V):
    """Z-correction phases maximising |Tr|: best point of a 10^3 grid, then 8 modified-Newton steps."""
    f = lambda th: jnp.abs(_z_trace(P, V, th)) ** 2
    th = _GRID[jnp.argmax(jax.vmap(f)(_GRID))]

    def newton(th, _):
        g, H = jax.grad(f)(th), jax.hessian(f)(th)
        w, Q = jnp.linalg.eigh(H)
        w = jnp.minimum(w, -1e-6 * (1 + jnp.abs(w).max()))     # ascent direction even off the concave region
        return th - Q @ ((Q.T @ g) / w), None

    th, _ = jax.lax.scan(newton, th, None, 8)
    return th


def fidelity_free_z(G, V=SQRT_ISWAP):
    """Average gate fidelity to V after the best virtual-Z corrections, leakage included
    (Pedersen et al., Phys. Lett. A 367, 47 (2007)): F = (Tr(M M†) + |Tr M|^2) / 20, M = V† D_a P D_b.
    The phases are optimised on a stopped-gradient copy: at the optimum dF/dphase = 0, so the gradient
    with respect to the pulse is exact. Returns (F, U) with U the unitarity Tr(P†P)/4."""
    P = physical_gate(G)
    th = jax.lax.stop_gradient(best_z_phases(jax.lax.stop_gradient(P), V))
    U = jnp.sum(jnp.abs(P) ** 2) / 4
    F = (4 * U + jnp.abs(_z_trace(P, V, th)) ** 2) / 20
    return F, U


def cost(G, objective="pe", concurrence_weight=1.0, unitarity_weight=3.0, digits=WEYL_DIGITS):
    """(cost, C, U) for the objective: ``pe`` = the paper's J_T; ``sqrt_iswap`` = 1 - F to sqrt(iSWAP), free Z."""
    if objective == "pe":
        return cost_JT(G, concurrence_weight, unitarity_weight, digits)
    if objective == "sqrt_iswap":
        F, U = fidelity_free_z(G, SQRT_ISWAP)
        return 1 - F, concurrence(c1c2c3(G, digits)), U
    raise ValueError(f"unknown objective {objective!r}")
# --8<-- [end:fidelity]
