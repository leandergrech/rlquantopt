"""Gate metrics for excitation-preserving two-qubit gates, in closed form.

The realised gate is G = diag(g00, V, e) in the basis |00>, |01>, |10>, |11>,
with V = [[a, b], [c, d]]. For such a gate U·Ũ (Ũ = (σy⊗σy) Uᵀ (σy⊗σy)) is
diag(g00·e, V·[[d, b], [c, a]], g00·e), whose eigenvalues are g00·e (twice)
and (ad + bc) ± 2·sqrt(abcd). This lets us follow weylchamber.c1c2c3
(Childs et al., PRA 68, 052311) exactly without a general 4x4 eig, which JAX
does not provide on GPU.
"""
import jax.numpy as jnp

WEYL_DIGITS = 8     # weylchamber.prec.DEFAULT_WEYL_PRECISSION
_M = jnp.array([[1., 1., 0.], [1., 0., 1.], [0., 1., 1.]])


def weyl_eigenvalues(G):
    g00, a, b, c, d, e = G[0, 0], G[1, 1], G[1, 2], G[2, 1], G[2, 2], G[3, 3]
    p = a * d + b * c
    r = 2 * jnp.sqrt(a * b * c * d)
    ev = jnp.stack([g00 * e, p + r, p - r, g00 * e])
    det = g00 * e * (a * d - b * c)
    return ev / jnp.sqrt(det)


def c1c2c3(G, digits=WEYL_DIGITS):
    """Weyl chamber coordinates (units of π) of a block-diagonal 4x4 gate, as weylchamber.c1c2c3."""
    two_S = jnp.angle(weyl_eigenvalues(G)) / jnp.pi
    two_S = jnp.where(two_S <= -0.5, two_S + 2.0, two_S)
    S = jnp.sort(two_S / 2.0)[::-1]
    n = jnp.round(jnp.sum(S)).astype(int)
    S = S - (jnp.arange(4) < n)
    S = jnp.roll(S, -n)
    c1, c2, c3 = _M @ S[:3]
    c1 = jnp.where(c3 < 0, 1 - c1, c1)
    c3 = jnp.abs(c3)
    return jnp.round(jnp.stack([c1, c2, c3]), digits) + 0.0


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


def cost_JT(G, concurrence_weight=1.0, unitarity_weight=3.0):
    """J_T = 1 - (w_c C + w_u U)/(w_c + w_u); paper eq. 2 with the default 1:3 weights."""
    C = concurrence(c1c2c3(G))
    U = unitarity(G)
    K = concurrence_weight + unitarity_weight
    return 1 - (concurrence_weight * C + unitarity_weight * U) / K, C, U
