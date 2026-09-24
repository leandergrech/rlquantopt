"""How the coverage of one pulse shrinks as more Hamiltonian parameters drift.

For D = 1 ... 8 drifting parameters (added in the order omega_0, omega_1, omega_c, alpha_0,
alpha_1, alpha_c, g_0, g_1), each drawn independently and uniformly within ±delta of its
nominal value (relative), we sample M drifted systems and report the fraction on which a
fixed pulse still gives J_T <= threshold, and the implied number of distinct calibrations
N_D = 1 / fraction needed to cover the box with pulses of the same robustness.

delta = 1.1e-3 matches the recool drift of the qubit frequencies (±5.7 MHz, rlquantopt.jx.drift);
for the other parameters no drift measurements were found, so the same relative spread is a
stated assumption. Writes docs/figures/drift_dimension.{png,json}.

    JAX_PLATFORMS=cpu python scripts/drift_dimension.py
"""
import argparse
import json
import os

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rlquantopt.jx import physics, metrics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAMES = ["ω₀", "ω₁", "ω_c", "α₀", "α₁", "α_c", "g₀", "g₁"]
DT = 0.05


def nominal_vector(m=physics.ModelParams()):
    return np.array([m.omega_s[0], m.omega_s[1], m.omega_c_0, m.alpha_s[0], m.alpha_s[1], m.alpha_c, m.g[0], m.g[1]])


def JT_of(u, p):
    params = physics.ModelParams(omega_s=p[0:2], omega_c_0=p[2], alpha_s=p[3:5], alpha_c=p[5], g=p[6:8])
    sector = physics.propagate(physics.sector_hamiltonian(params), physics.initial_state(), u, DT)
    return metrics.cost_JT(physics.realised_gate(sector))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta", type=float, default=1.1e-3)
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--threshold", type=float, default=1e-3)
    ap.add_argument("--absolute-mhz", type=float, default=None,
                    help="stylised mode: every parameter drifts by ±this many MHz instead of ±delta relative")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "figures"))
    args = ap.parse_args()

    d = np.load(os.path.join(ROOT, "docs", "figures", "rl_grape_robust.npz"))
    pulses = {"RL (paper)": d["pulse_rl"], "GRAPE": d["pulse_grape"], "robust GRAPE (recool, ω₀ ω₁)": d["pulse_robust"]}
    p0 = nominal_vector()
    rng = np.random.default_rng(0)
    f = jax.jit(jax.vmap(JT_of, in_axes=(None, 0)))
    res = {}
    for name, u in pulses.items():
        u = jnp.asarray(u)
        rows = []
        for D in range(1, len(p0) + 1):
            eps = np.zeros((args.samples, len(p0)))
            eps[:, :D] = rng.uniform(-1, 1, (args.samples, D))
            drifted = p0 + eps * args.absolute_mhz * 1e-3 if args.absolute_mhz else p0 * (1 + eps * args.delta)
            J = np.asarray(f(u, jnp.asarray(drifted)))
            frac = float(np.mean(J <= args.threshold))
            rows.append(dict(D=D, fraction_covered=frac, median_JT=float(np.median(J)),
                             calibrations_needed=float(1 / frac) if frac > 0 else float("inf")))
            print(f"{name:32s} D={D}  covered {frac:6.3f}  median J_T {np.median(J):.1e}", flush=True)
        res[name] = rows

    os.makedirs(args.out, exist_ok=True)
    json.dump(dict(settings=dict(delta=args.delta, absolute_mhz=args.absolute_mhz, samples=args.samples, threshold=args.threshold, order=NAMES),
                   results=res), open(os.path.join(args.out, f"drift_dimension{args.tag}.json"), "w"), indent=2)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True)
    for name, rows in res.items():
        D = [r["D"] for r in rows]
        axs[0].semilogy(D, [max(r["fraction_covered"], 0.5 / args.samples) for r in rows], "o-", label=name)
        axs[1].semilogy(D, [r["median_JT"] for r in rows], "o-", label=name)
    axs[0].set_ylabel(f"fraction with J_T ≤ {args.threshold:g}")
    axs[1].set_ylabel("median $J_T$ over drifted systems")
    axs[1].axhline(args.threshold, c="k", ls="--", lw=0.8)
    for ax in axs:
        ax.set_xticks(range(1, len(NAMES) + 1), [f"{i}\n+{n}" for i, n in enumerate(NAMES, 1)])
        ax.set_xlabel("number of drifting parameters D (parameter added)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    spread = f"±{args.absolute_mhz} MHz" if args.absolute_mhz else f"±{args.delta * 100:.2f} %"
    axs[0].set_title(f"Coverage of one pulse, each drifting parameter within {spread}", fontsize=10)
    axs[1].set_title("Typical gate error as more parameters drift", fontsize=10)
    fig.savefig(os.path.join(args.out, f"drift_dimension{args.tag}.png"), dpi=120)
    print("wrote", os.path.join(args.out, f"drift_dimension{args.tag}.png"))


if __name__ == "__main__":
    main()
