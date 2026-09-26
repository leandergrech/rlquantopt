"""The carrier MDP of ideas i09/i10 (docs/figures/carrier_mdp_a.png, carrier_mdp_b.png).

(a) A schematic in the style of the paper's Fig. 2: at each decision point o_k the policy outputs
    a_k = (dA, dphi, db), which nudges the knobs; for the next K samples the pulse is
    u(t) = b_k + A_k cos(2 pi f_d t + phi_k). The carrier is drawn slower than it is, for legibility.
(b) What a trained policy (ibis, carrier, calibration only, K = 15) plays on one drifted device: its knobs
    and the resulting pulse.

    python scripts/plot_carrier_mdp.py [--run RUN_DIR]
"""
import argparse
import glob
import importlib.util
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "docs", "figures")
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def schematic(ax):
    K, n_show = 12, 3                       # samples per step drawn, steps drawn before and after the gap
    A = [0.3, 0.55, 0.9, 1.0, 0.8, 0.45]
    b = [0.0, 0.05, 0.15, 0.2, 0.1, 0.0]
    ph = [0.0, 0.0, 0.3, 0.3, 0.6, 0.6]
    f = 1 / 9.0                             # carrier period of 9 samples in the drawing
    ax.axhline(0, color="k", lw=0.8)
    x0 = 0
    for seg, ks in enumerate(((0, 1, 2), (3, 4, 5))):
        for k in ks:
            s = np.arange(K) + k * K
            xs = x0 + np.arange(K) + (k - ks[0]) * K
            u = b[k] + A[k] * np.cos(2 * np.pi * f * s + ph[k])
            ax.vlines(xs, 0, u, color="0.55", lw=0.8, linestyles="dashed")
            ax.plot(xs, u, "o", color="k", ms=3)
            ax.hlines([b[k] + A[k], b[k] - A[k]], xs[0] - 0.5, xs[-1] + 0.5, color="#b5651d", lw=1.2)
            ax.hlines(b[k], xs[0] - 0.5, xs[-1] + 0.5, color="#2b4c7e", lw=1.2, linestyles="dotted")
            label = f"$a_{{{k}}}$" if seg == 0 else ("$a_{T-1}$" if k == ks[-1] else f"$a_{{T-{ks[-1] - k + 1}}}$")
            obs = f"$o_{{{k}}}$" if seg == 0 else ("$o_{T-1}$" if k == ks[-1] else f"$o_{{T-{ks[-1] - k + 1}}}$")
            ax.annotate("", xy=(xs[-1] + 0.5, 1.55), xytext=(xs[0] - 0.5, 1.55),
                        arrowprops=dict(arrowstyle="-", color="k", lw=0.8, connectionstyle="bar,fraction=0.12"))
            ax.text((xs[0] + xs[-1]) / 2, 1.78, label, ha="center", fontsize=11)
            ax.axvline(xs[0] - 0.5, color="k", lw=0.6, ls=":")
            ax.text(xs[0] - 0.5, -1.55, obs, ha="center", fontsize=11, bbox=dict(fc="white", ec="none", pad=1.5))
        x0 = xs[-1] + 1 + 6 if seg == 0 else x0
        if seg == 0:
            ax.text(xs[-1] + 3.5, 0.05, r"$\cdots$", fontsize=14, ha="center")
    ax.axvline(xs[-1] + 0.5, color="k", lw=0.6, ls=":")
    ax.text(xs[-1] + 0.5, -1.55, "$o_T$", ha="center", fontsize=11, bbox=dict(fc="white", ec="none", pad=1.5))
    ax.annotate("", xy=(K - 0.5, -1.25), xytext=(-0.5, -1.25), arrowprops=dict(arrowstyle="<->", lw=0.8))
    ax.text(K / 2 - 0.5, -1.2, r"$K\,\Delta t$", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(-1.8, 2.05)
    ax.set_xlim(-2, xs[-1] + 3)
    ax.set_yticks([])
    ax.set_xticks([])
    for sp in ("top", "right", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.set_ylabel("$u(t)$", fontsize=12)
    ax.plot([], [], color="#b5651d", label=r"envelope $b_k \pm A_k$")
    ax.plot([], [], color="#2b4c7e", ls=":", label=r"offset $b_k$")
    ax.plot([], [], "o", color="k", ms=3, label=r"samples $u = b_k + A_k \cos(2\pi f_d t + \varphi_k)$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=9, frameon=False)
    ax.set_title(r"(a) Each action $a_k = (\Delta A, \Delta\varphi, \Delta b) \in [-1,1]^3$ nudges the knobs; "
                 r"the next $K$ samples follow the carrier (drawn slower than it is)" "\n"
                 r"observation $o_k = (\mathbf{c},\ A_k,\ b_k,\ \cos\varphi_k,\ \sin\varphi_k,\ t_k)$, with "
                 r"$\mathbf{c}$ the calibration measured once before the pulse (qubit and coupler frequency offsets)",
                 fontsize=10)


def rollout_panel(axs, run):
    import jax
    import jax.numpy as jnp
    ie, c = _load("ibis_eval"), _load("coupler_drift_experiment")
    from rlquantopt.jx import env as jenv
    cfg, model, params, step, _ = ie.load_run(run)
    rng = np.random.default_rng(11)
    dev = [c.model_at(*rng.uniform(-1, 1, 2) * c.W_Q, rng.uniform(-1, 1) * c.W_C) for _ in range(24)][3]
    obs, s = jenv.reset_params(dev, cfg, jax.random.PRNGKey(0))
    step = jax.jit(lambda s, o: jenv.step(s, model.dist(params["actor"], o)[0], cfg))
    knobs, amps, JT = [], [], []
    for _ in range(cfg.n_steps):
        obs, s, r, term, trunc, info = step(s, obs)
        knobs.append(np.asarray(s.knobs))
        amps.append(np.asarray(s.amps_cur))
        JT.append(float(info["JT"]))
    knobs, u = np.asarray(knobs), np.concatenate(amps)
    t = (np.arange(len(u)) + 0.5) * cfg.dt
    tk = (np.arange(cfg.n_steps) + 1) * cfg.n_time_steps * cfg.dt
    ax = axs[0]
    ax.plot(t, u, color="k", lw=0.5)
    ax.step(tk, knobs[:, 2] + knobs[:, 0], where="pre", color="#b5651d", lw=1.2, label="$b_k \\pm A_k$")
    ax.step(tk, knobs[:, 2] - knobs[:, 0], where="pre", color="#b5651d", lw=1.2)
    ax.step(tk, knobs[:, 2], where="pre", color="#2b4c7e", lw=1.2, ls=":", label="$b_k$")
    ax.set_ylabel("$u(t)$ [rad/ns]")
    ax.set_title(f"(b) A trained policy on one drifted device (calibration only, K = {cfg.n_time_steps}, "
                 f"{cfg.n_steps} decisions): 1 − F = {JT[-1]:.1e} at the end", fontsize=10)
    ax.legend(fontsize=8, loc="upper right", frameon=False)
    ax2 = axs[1]
    ax2.step(tk, np.unwrap(knobs[:, 1]), where="pre", color="k", lw=1.2)
    ax2.set_ylabel(r"$\varphi_k$ [rad]")
    ax2.set_xlabel("t [ns]")
    for a in axs:
        a.grid(alpha=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="default: runs/i09_ibis/*car_ctx_k15 (seed 123)")
    args = ap.parse_args()
    run = args.run or sorted(glob.glob(os.path.join(ROOT, "runs", "i09_ibis", "ppo_*_s123_car_ctx_k15")))[0]
    fa, ax_a = plt.subplots(figsize=(12, 3.9), constrained_layout=True)
    schematic(ax_a)
    fa.savefig(os.path.join(FIG, "carrier_mdp_a.png"), dpi=150)
    fb = plt.figure(figsize=(12, 4.8), constrained_layout=True)
    gs = fb.add_gridspec(2, 1, height_ratios=[1, 0.45])
    ax_b = fb.add_subplot(gs[0])
    ax_c = fb.add_subplot(gs[1], sharex=ax_b)
    rollout_panel((ax_b, ax_c), run)
    fb.savefig(os.path.join(FIG, "carrier_mdp_b.png"), dpi=150)
    path = os.path.join(FIG, "carrier_mdp_a.png")
    print("wrote", path)


if __name__ == "__main__":
    main()
