"""Policy-level generalisation (paper Figs. 13-17): re-run the policy on detuned systems.

Unlike rlquantopt.jx.robustness, which replays one fixed pulse, the policy here
generates a new pulse for every (Δω0, Δω1), closing the loop through its
observations. The per-point metric follows the paper's GeneralisationTests
notebook: the best per-step reward of the deterministic episode, with C, U and
the time at that step. Like the notebook, episodes continue after an out-of-bounds
truncation (``stop_on_truncation=False``).
"""
import numpy as np
import jax
import jax.numpy as jnp

from rlquantopt.jx import env as jenv
from rlquantopt.jx.agents.common import evaluate


def sweep(model, params, cfg: jenv.EnvConfig, d_omega0_mhz, d_omega1_mhz, chunk=512, stop_on_truncation=False):
    """Return dict of arrays (len(d_omega0), len(d_omega1)): best_reward, JT, C, U, t_best."""
    w0 = np.asarray(cfg.model.omega_s)
    g0, g1 = np.meshgrid(np.asarray(d_omega0_mhz), np.asarray(d_omega1_mhz), indexing="ij")
    omegas = np.stack([w0[0] + g0.ravel() * 1e-3, w0[1] + g1.ravel() * 1e-3], axis=-1)

    def one(om):
        ep = evaluate(model, params, cfg, om, stop_on_truncation)
        r = jnp.where(ep["alive"], ep["reward"], -jnp.inf)
        k = jnp.argmax(r)
        return dict(best_reward=r[k], JT=ep["JT"][k], C=ep["concurrence"][k], U=ep["unitarity"][k], t_best=ep["t"][k])

    f = jax.jit(jax.vmap(one))
    parts = [jax.device_get(f(jnp.asarray(omegas[i:i + chunk]))) for i in range(0, len(omegas), chunk)]
    return {k: np.concatenate([p[k] for p in parts]).reshape(g0.shape) for k in parts[0]}


def island_extent(best_reward, d0, d1, threshold=3.8):
    """Bounding box (MHz) of the connected region with reward >= threshold that contains the nominal point,
    or the region nearest to it (paper Fig. 16 reports Δω ranges of this island)."""
    from scipy import ndimage
    mask = best_reward >= threshold
    labels, n = ndimage.label(mask)
    if n == 0:
        return None
    i0, i1 = np.argmin(np.abs(d0)), np.argmin(np.abs(d1))
    lab = labels[i0, i1]
    if lab == 0:
        pts = np.argwhere(mask)
        lab = labels[tuple(pts[np.argmin(((pts - [i0, i1]) ** 2).sum(1))])]
    idx = np.argwhere(labels == lab)
    return dict(d_omega0=(float(d0[idx[:, 0].min()]), float(d0[idx[:, 0].max()])),
                d_omega1=(float(d1[idx[:, 1].min()]), float(d1[idx[:, 1].max()])), n_points=len(idx))
