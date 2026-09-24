"""How far can the coupler frequency drift before the existing pulses fail?

In the paper's model the coupler term is 2π(ω_c − ω_r) b†b + u(t) b†b, so a coupler drift δ is
exactly a static offset 2πδ on the control. This scans δ for the stored pulses (RL, GRAPE, and the
robust-GRAPE pulses for the recool and fabrication ranges, which never saw coupler drift), with the
qubit frequencies nominal, and marks the literature-based coupler drift ranges.

Coupler drift ranges come from flux-offset drift measured by Dai et al., PRX Quantum 2, 040313
(2021): 1.3 mΦ0 RMS after two days, 2.0 mΦ0 RMS after 17 days, 20 mΦ0 for the worst loop.
Converting flux to frequency assumes a symmetric-SQUID coupler with f_max = 8 GHz parked at
7.445 GHz (our model has no flux map). Writes docs/figures/coupler_drift_scan.{png,json}.
"""
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
FIG = os.path.join(ROOT, "docs", "figures")
DT = 0.05
F_MAX, F_PARK = 8.0, 7.445          # GHz, assumed symmetric SQUID coupler


def coupler_shift_mhz(dphi_m, f_max=F_MAX, f_park=F_PARK):
    """Frequency shift (MHz) of a symmetric-SQUID transmon parked at f_park for a flux offset dphi_m (mΦ0)."""
    x0 = np.arccos((f_park / f_max) ** 2)
    x = x0 + np.pi * dphi_m * 1e-3
    return (f_max * np.sqrt(np.abs(np.cos(x))) - f_park) * 1e3


def ranges():
    rows = {}
    for label, dphi in (("2 days, RMS", 1.3), ("17 days, RMS", 2.0), ("17 days, worst loop", 20.0)):
        up, down = coupler_shift_mhz(-dphi), coupler_shift_mhz(dphi)
        rows[label] = dict(flux_mphi0=dphi, shift_mhz=(float(down), float(up)), half_width_mhz=float(max(abs(up), abs(down))))
    return rows


def JT(u, d_mhz, model=physics.ModelParams()):
    m = model._replace(omega_c_0=model.omega_c_0 + d_mhz * 1e-3)
    s = physics.propagate(physics.sector_hamiltonian(m), physics.initial_state(), u, DT)
    return metrics.cost_JT(physics.realised_gate(s))[0]


def main():
    rg = ranges()
    for k, v in rg.items():
        print(f"{k:22s} {v['flux_mphi0']:5.1f} mΦ0 -> {v['shift_mhz'][0]:+7.1f} / {v['shift_mhz'][1]:+7.1f} MHz", flush=True)
    rec, fab = np.load(os.path.join(FIG, "rl_grape_robust.npz")), np.load(os.path.join(FIG, "rl_grape_robust_fab.npz"))
    pulses = {"RL (paper)": rec["pulse_rl"], "GRAPE": rec["pulse_grape"],
              "robust GRAPE (recool)": rec["pulse_robust"], "robust GRAPE (fabrication)": fab["pulse_robust"]}
    d = np.linspace(-150, 150, 301)
    f = jax.jit(jax.vmap(JT, in_axes=(None, 0)))
    scan = {k: np.asarray(f(jnp.asarray(u), jnp.asarray(d))) for k, u in pulses.items()}
    out = dict(assumption=dict(f_max_ghz=F_MAX, f_park_ghz=F_PARK), ranges=rg, d_mhz=d.tolist(),
               width_JT_below_1e3_mhz={k: float(np.ptp(d[v <= 1e-3])) if (v <= 1e-3).any() else 0.0 for k, v in scan.items()})
    for k, v in out["width_JT_below_1e3_mhz"].items():
        print(f"{k:28s} J_T <= 1e-3 over a {v:.1f} MHz wide coupler window", flush=True)
    json.dump(out | dict(scan={k: v.tolist() for k, v in scan.items()}), open(os.path.join(FIG, "coupler_drift_scan.json"), "w"))

    fig, ax = plt.subplots(figsize=(9, 4.4), constrained_layout=True)
    for k, v in scan.items():
        ax.semilogy(d, v, label=k)
    for (lab, r), c in zip(rg.items(), ("#999999", "#666666", "#333333")):
        w = r["half_width_mhz"]
        ax.axvspan(-w, w, color=c, alpha=0.08)
        ax.text(w, 3e-1, f" {lab}\n ±{w:.0f} MHz", fontsize=7, va="top", color=c)
    ax.axhline(1e-3, c="k", ls="--", lw=0.8)
    ax.set_xlabel("coupler frequency drift δ [MHz] (= static control offset 2πδ)")
    ax.set_ylabel("$J_T$ (qubits nominal)")
    ax.set_title("Existing pulses under coupler drift; shaded: flux-drift ranges [Dai 2021], f_max = 8 GHz assumed", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(FIG, "coupler_drift_scan.png"), dpi=120)
    print("wrote coupler_drift_scan.png")


if __name__ == "__main__":
    main()
