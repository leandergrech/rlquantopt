"""Idea i08 hippogriff: identify the device's drift with shrinking uncertainty, then act on the belief.

Stage 1, simulation (pretraining):
- PPO with domain randomisation over the drift range, the policy conditioned on the drift:
  pi(a | s, z), z = the device's normalised (gain, offset) in [-1, 1]^2 (``DriftTracker(show_z=True)``),
  and a blind control that never sees z (``show_z=False``, the classic robust policy).
- Fox's improvement-equivalent model rides along (no exploitation, so training is plain PPO) and its
  calibration, predicted against realised return change per update, is logged.

Stage 2, deployment on a held-out device, with the drift unknown. The belief over z is Gaussian,
N(m, P), starting from the prior of the whole range (mean 0, variance 1/3 per dimension). After every
step an extended Kalman update with the measured y = (change of the commanded amplitude, tracking
score) and the measurement Jacobian J = dy/dz at m:

    P <- (I - K J) P (I - K J)^T + K R K^T,   K = P J^T (J P J^T + R)^-1,

so, for the linearised model, P never grows: the uncertainty is largest at the first action and shrinks
with every interaction (a belief that is recursive over interactions, not an imagined rollout).
The measurement model is either the environment's own ``measure`` (A, grey box: the physics is known and
only its parameters drift) or a network trained on simulated transitions (B, black box).

Hippogriff's actions, from ``n_candidates`` proposals around the policy's action pi(s, m):
- **trust region**: every action is clipped to c = c_min + (1 - c_min)(1 - rho), with
  rho = (det P / det P0)^(1 / 2d) the belief's geometric standard deviation relative to the prior:
  small first actions, widening as the uncertainty shrinks;
- **safety**: a candidate is vetoed if at any sigma point of the belief the model predicts an overshoot;
- **probing** while rho > rho_stop: the candidate with the largest expected information gain about z,
  1/2 log det(I + R^-1 J P J^T) (ties broken towards pi's action); afterwards pi(s, m) itself.

Modes compared per device: blind, robust (z fixed at the prior mean), oracle (true z), filter_A / filter_B
(the belief updates z, actions from pi: passive identification) and hippogriff_A / hippogriff_B (the full method).

    python -m rlquantopt.jx.hippogriff --seed 0 --devices 12 --episodes 5
"""
import argparse
import dataclasses
import json
import os
import pickle
import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import jax
import jax.numpy as jnp
import optax

from rlquantopt.jx.agents import ppo_plus
from rlquantopt.jx.agents.common import MLP
from rlquantopt.jx.agents.fox import FoxAdapter, FoxConfig
from rlquantopt.jx.ideas import run_root
from rlquantopt.jx.toy_envs import DriftTracker
from rlquantopt.jx.toybench import ENV_PPO

MODES = ("blind", "robust", "oracle", "filter_A", "hippogriff_A", "filter_B", "hippogriff_B")


@dataclass(frozen=True)
class GriffinConfig:
    r_du: float = 1e-3              # model-error floor (std) of the amplitude change (model A)
    r_score: float = 0.05           # model-error floor (std) of the tracking score (model A)
    noise_du: float = 0.0           # actual measurement noise (std) on the amplitude change (finite shots)
    noise_score: float = 0.0        # actual measurement noise (std) on the tracking score; both known to the filter
    n_candidates: int = 16
    probe_sigma: float = 0.5
    tie_beta: float = 0.05
    c_min: float = 0.2
    rho_stop: float = 0.1
    safety_margin: float = 0.02
    z_clip: float = 1.2
    # The belief: "ekf" (linearised Kalman), "ekf_nis" (+ inflation when the innovations are inconsistent),
    # "grid" (exact Bayes on a grid_n x grid_n grid over z; probing by the predictive spread over the belief)
    filter: str = "ekf"
    nis_threshold: float = 9.21     # chi^2, 2 dof, 99 %
    grid_n: int = 41


# --8<-- [start:belief]
def prior(d=2):
    return jnp.zeros(d, jnp.float32), jnp.eye(d, dtype=jnp.float32) / 3.0      # U(-1, 1): variance 1/3


def rho_of(P, P0):
    """Geometric-mean standard deviation of the belief relative to the prior's."""
    d = P.shape[0]
    return jnp.exp((jnp.linalg.slogdet(P)[1] - jnp.linalg.slogdet(P0)[1]) / (2 * d))


def ekf_update(m, P, y, yhat, J, R):
    S = J @ P @ J.T + R
    K = P @ J.T @ jnp.linalg.inv(S)
    A = jnp.eye(P.shape[0], dtype=P.dtype) - K @ J
    P = A @ P @ A.T + K @ R @ K.T                                   # Joseph form: stays symmetric PSD
    return m + K @ (y - yhat), 0.5 * (P + P.T)


def info_gain(J, P, R):
    return 0.5 * jnp.linalg.slogdet(jnp.eye(R.shape[0], dtype=R.dtype) + jnp.linalg.solve(R, J @ P @ J.T))[1]


def ekf_nis_update(m, P, y, yhat, J, R, P0, threshold):
    """EKF update after a consistency check: if the normalised innovation squared exceeds ``threshold``, the
    belief was too sure; P is scaled up by NIS / dim(y) (at most back to the prior's volume) first."""
    nu = y - yhat
    nis = nu @ jnp.linalg.solve(J @ P @ J.T + R, nu)
    d = P.shape[0]
    cap = jnp.exp((jnp.linalg.slogdet(P0)[1] - jnp.linalg.slogdet(P)[1]) / d)       # to the prior's volume
    alpha = jnp.where(nis > threshold, jnp.clip(nis / y.shape[0], 1.0, jnp.maximum(cap, 1.0)), 1.0)
    m, P = ekf_update(m, alpha * P, y, yhat, J, R)
    return m, P, nis


def make_grid(n):
    lin = jnp.linspace(-1, 1, n, dtype=jnp.float32)
    g1, g2 = jnp.meshgrid(lin, lin, indexing="ij")
    return jnp.stack([g1.ravel(), g2.ravel()], -1)


def grid_moments(logw, Z):
    w = jax.nn.softmax(logw)
    m = w @ Z
    d = Z - m
    return m, (d.T * w) @ d + 1e-8 * jnp.eye(Z.shape[1], dtype=Z.dtype)


def grid_update(logw, y, H, R):
    """Exact Bayes on the grid: add the Gaussian log-likelihood of y under each grid point's prediction H."""
    logw = logw - 0.5 * jnp.sum((y - H) ** 2 / jnp.diag(R), -1)
    return logw - jax.scipy.special.logsumexp(logw)


def predictive_info_gain(H, w, R):
    """1/2 log det(I + R^-1 C), C the spread of the predictions H over the belief w: exact for a linear
    model, and it sees the nonlinearity a Jacobian at the mean misses."""
    d = H - w @ H
    C = (d.T * w) @ d
    return 0.5 * jnp.linalg.slogdet(jnp.eye(R.shape[0], dtype=R.dtype) + jnp.linalg.solve(R, C))[1]


def sigma_points(m, P):
    d = m.shape[0]
    L = jnp.linalg.cholesky(d * P + 1e-9 * jnp.eye(d, dtype=P.dtype))
    return jnp.concatenate([m[None], m + L.T, m - L.T])
# --8<-- [end:belief]


def noise_var(gcfg):
    return jnp.array([gcfg.noise_du, gcfg.noise_score], jnp.float32) ** 2


class SimModel:
    """Model A: the environment's own measurement function (the physics is known, its parameters drift)."""

    def __init__(self, env: DriftTracker, gcfg: GriffinConfig):
        self.env = env
        self.R = jnp.diag(jnp.array([gcfg.r_du, gcfg.r_score], jnp.float32) ** 2 + noise_var(gcfg))

    def measure(self, s, a, z):
        return self.env.measure(s, a, z)

    def max_out(self, s, a, z):
        return self.env.max_out(s, a, z)


class LearnedModel:
    """Model B: an MLP (obs, action, z) -> (amplitude change, score, max |output|), trained on simulation."""

    def __init__(self, env: DriftTracker, net: MLP, params, y_mean, y_std, R):
        self.env, self.net, self.params = env, net, params
        self.y_mean, self.y_std, self.R = y_mean, y_std, R

    def _out(self, s, a, z):
        x = jnp.concatenate([self.env.obs_base(s), jnp.clip(a, -1, 1).astype(jnp.float32), z.astype(jnp.float32)])
        return self.net.apply(self.params, x) * self.y_std + self.y_mean

    def measure(self, s, a, z):
        return self._out(s, a, z)[:2]

    def max_out(self, s, a, z):
        return self._out(s, a, z)[2]


def pretrain(seed, steps, show_z):
    """PPO with domain randomisation; fox's model monitors it (exploit off: the updates are plain PPO)."""
    env = DriftTracker(show_z=show_z)
    cfg = ppo_plus.PlusConfig(**dict(ENV_PPO["tracker"], total_steps=steps))
    model, runner, opt = ppo_plus.init(jax.random.PRNGKey(seed), env, cfg)
    update = jax.jit(ppo_plus.make_update(model, cfg))
    fox = FoxAdapter(update, runner, opt, FoxConfig(exploit=False), env.max_steps)
    for _ in range(cfg.n_updates):
        fox.step()
    J = np.array([r["J"] for r in fox.log])
    pred = np.array([r["pred_mean"] for r in fox.log])[:-1]
    dJ = np.diff(J)
    ok = np.isfinite(pred)
    half = ok & (np.arange(len(pred)) >= len(pred) // 2)
    corr = lambda m: float(np.corrcoef(pred[m], dJ[m])[0, 1]) if m.sum() > 2 else float("nan")
    calib = dict(corr_all=corr(ok), corr_second_half=corr(half),
                 sign_agreement=float(np.mean(np.sign(pred[ok]) == np.sign(dJ[ok]))),
                 final_J=float(J[-20:].mean()))
    return model, fox.runner.params, calib


def fit_learned_model(env: DriftTracker, model, params, key, n_envs=256, episodes=4, train_steps=5000):
    """Simulated transitions from the z-conditioned policy with extra action noise (so that the probes hippogriff
    proposes are covered), on random devices; returns model B with R from held-out residuals."""
    A = env.act_dim

    def episode(k):
        k_reset, k_steps = jax.random.split(k)
        _, s = env.reset(k_reset)

        def body(s, kk):
            mu = model.ac.mode(params["actor"], env.obs(s))
            a = jnp.clip(mu + 0.5 * jax.random.normal(kk, (A,), jnp.float32), -1, 1)
            z = s.zhat
            _, out, score, u_next = env.outcome(s, a, s.gain, s.offset)
            y = jnp.stack([u_next - s.u, score, jnp.max(jnp.abs(out))])
            x = jnp.concatenate([env.obs_base(s), a, z])
            _, s2, _, term, trunc, _ = env.step(s, a)
            _, s_new = env.reset(kk)
            s2 = jax.tree_util.tree_map(lambda n, o: jnp.where(term | trunc, n, o), s_new, s2)
            return s2, (x, y)

        _, (x, y) = jax.lax.scan(body, s, jax.random.split(k_steps, env.max_steps))
        return x, y

    keys = jax.random.split(key, n_envs * episodes)
    x, y = jax.jit(jax.vmap(episode))(keys)
    x, y = np.asarray(x).reshape(-1, x.shape[-1]), np.asarray(y).reshape(-1, 3)
    n_val = len(x) // 10
    y_mean, y_std = y[n_val:].mean(0), y[n_val:].std(0) + 1e-6
    net = MLP((128, 128), 3, 1.0, "tanh")
    p = net.init(jax.random.PRNGKey(1), jnp.zeros((1, x.shape[1])))
    tx = optax.adam(1e-3)
    o = tx.init(p)
    xt, yt = jnp.asarray(x[n_val:]), jnp.asarray((y[n_val:] - y_mean) / y_std)

    @jax.jit
    def step(p, o, k):
        idx = jax.random.randint(k, (1024,), 0, xt.shape[0])
        loss, g = jax.value_and_grad(lambda p: jnp.mean((net.apply(p, xt[idx]) - yt[idx]) ** 2))(p)
        u, o = tx.update(g, o, p)
        return optax.apply_updates(p, u), o, loss

    for k in jax.random.split(jax.random.PRNGKey(2), train_steps):
        p, o, loss = step(p, o, k)
    pred = np.asarray(net.apply(p, jnp.asarray(x[:n_val]))) * y_std + y_mean
    resid = y[:n_val] - pred
    r_std = resid.std(0)
    R = jnp.diag(jnp.asarray(r_std[:2] ** 2 + 1e-8, jnp.float32))
    fit = dict(val_rmse=r_std.tolist(), val_r2=(1 - resid.var(0) / y[:n_val].var(0)).tolist())
    return LearnedModel(env, net, p, jnp.asarray(y_mean, jnp.float32), jnp.asarray(y_std, jnp.float32), R), fit


# --8<-- [start:episode]
def make_episode(env: DriftTracker, model, actor_params, meas, gcfg: GriffinConfig, mode: str):
    """One deployment episode on a device, carrying the belief (m, P) in and out. vmapped over devices."""
    A = env.act_dim
    uses_filter = mode.startswith(("filter", "hippogriff"))
    R = meas.R if meas is not None else None
    grid = gcfg.filter == "grid"
    Z = make_grid(gcfg.grid_n) if grid else None
    on_grid = lambda s, a: jax.vmap(lambda z: meas.measure(s, a, z))(Z)                # (G, 2) predictions

    def episode(key, gain, offset, m, P, logw, P0):
        z_true = env.z_of(gain, offset)
        k0, key = jax.random.split(key)
        _, s = env.reset_with(k0, gain, offset, zhat=m)

        def body(carry, k):
            s, m, P, logw, alive, ret = carry
            zhat = {"oracle": z_true, "robust": jnp.zeros(2), "blind": jnp.zeros(2)}.get(mode, m)
            s = s._replace(zhat=zhat.astype(jnp.float32))
            mu = model.ac.mode(actor_params, env.obs(s))
            rho = rho_of(P, P0)
            probing = jnp.array(False)
            if mode.startswith("hippogriff"):
                c = gcfg.c_min + (1 - gcfg.c_min) * (1 - rho)
                noise = gcfg.probe_sigma * jax.random.normal(k, (gcfg.n_candidates, A), jnp.float32)
                cands = jnp.clip(jnp.concatenate([mu[None], jnp.zeros((1, A), jnp.float32), mu + noise]), -c, c)
                sp = sigma_points(m, P)
                worst = jax.vmap(lambda a: jax.vmap(lambda z: meas.max_out(s, a, z))(sp).max())(cands)
                safe = worst <= 1 - gcfg.safety_margin
                if grid:
                    w = jax.nn.softmax(logw)
                    ig = jax.vmap(lambda a: predictive_info_gain(on_grid(s, a), w, R))(cands)
                else:
                    Js = jax.vmap(lambda a: jax.jacfwd(lambda z: meas.measure(s, a, z))(m))(cands)
                    ig = jax.vmap(lambda J: info_gain(J, P, R))(Js)
                close = -jnp.sum((cands - jnp.clip(mu, -c, c)) ** 2, -1)
                probing = rho > gcfg.rho_stop
                score = jnp.where(probing, ig + gcfg.tie_beta * close, close)
                score = jnp.where(safe, score, -jnp.inf)
                a = cands[jnp.where(safe.any(), jnp.argmax(score), jnp.argmin(worst))]
            else:
                a = mu
            _, s2, r, term, trunc, info = env.step(s, a)
            nis = jnp.zeros((), jnp.float32)
            if uses_filter:
                noise = jnp.array([gcfg.noise_du, gcfg.noise_score], jnp.float32)
                y = jnp.stack([s2.u - s.u, info["score"]]) + noise * jax.random.normal(jax.random.fold_in(k, 1), (2,), jnp.float32)
                if grid:
                    logw2 = grid_update(logw, y, on_grid(s, a), R)
                    m2, P2 = grid_moments(logw2, Z)
                    logw = jnp.where(alive, logw2, logw)
                else:
                    J = jax.jacfwd(lambda z: meas.measure(s, a, z))(m)
                    yhat = meas.measure(s, a, m)
                    if gcfg.filter == "ekf_nis":
                        m2, P2, nis = ekf_nis_update(m, P, y, yhat, J, R, P0, gcfg.nis_threshold)
                    else:
                        m2, P2 = ekf_update(m, P, y, yhat, J, R)
                        nu = y - yhat
                        nis = nu @ jnp.linalg.solve(J @ P @ J.T + R, nu)
                m = jnp.where(alive, jnp.clip(m2, -gcfg.z_clip, gcfg.z_clip), m)
                P = jnp.where(alive, P2, P)
            out = dict(zerr=jnp.linalg.norm(m - z_true), rho=rho_of(P, P0), probing=probing & alive,
                       a_max=jnp.max(jnp.abs(a)), alive=alive, nis=nis)
            ret = ret + r * alive
            alive = alive & ~(term | trunc)
            return (s2, m, P, logw, alive, ret), out

        (_, m, P, logw, _, ret), out = jax.lax.scan(
            body, (s, m, P, logw, jnp.array(True), jnp.zeros((), jnp.float32)), jax.random.split(key, env.max_steps))
        return ret, m, P, logw, out

    return jax.jit(jax.vmap(episode, in_axes=(0, 0, 0, 0, 0, 0, None)))


def initial_belief(gcfg: GriffinConfig, n_devices):
    """(m, P, logw, P0) for n_devices at the prior; for the grid filter P0 is the uniform grid's own spread."""
    if gcfg.filter == "grid":
        Z = make_grid(gcfg.grid_n)
        logw = jnp.full((n_devices, Z.shape[0]), -jnp.log(Z.shape[0]), jnp.float32)
        m0, P0 = grid_moments(logw[0], Z)
    else:
        (m0, P0), logw = prior(), jnp.zeros((n_devices, 1), jnp.float32)
    return jnp.tile(m0, (n_devices, 1)), jnp.tile(P0, (n_devices, 1, 1)), logw, P0
# --8<-- [end:episode]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--devices", type=int, default=12)
    p.add_argument("--episodes", type=int, default=5, help="deployment episodes per device (belief carried over)")
    p.add_argument("--pretrain-steps", type=int, default=2_000_000)
    p.add_argument("--set", nargs="*", default=[], help="GriffinConfig overrides, key=value (with --tag: one variant)")
    p.add_argument("--tag", default="", help="appended to the run directory name")
    p.add_argument("--variants", nargs="*", default=None,
                   help="several deployments on the same pretrained policies, each 'key=value,key=value:tag' "
                        "(e.g. 'filter=\"grid\",noise_score=0.5:grid_noisy'); overrides --set/--tag")
    p.add_argument("--cache", default="runs/i08_hippogriff/pretrained",
                   help="pretrained policies and model B per seed are saved here and reused ('' disables)")
    p.add_argument("--out", default="runs")
    return p.parse_args(argv)


def _parse_overrides(text):
    return {k: eval(v) for k, v in (kv.split("=", 1) for kv in text.split(",") if kv)}


def pretrained(args, env, log):
    """The z-conditioned and blind policies and model B for this seed: from the cache, or trained and cached."""
    path = os.path.join(args.cache, f"seed{args.seed}_{args.pretrain_steps}.pkl") if args.cache else None
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            c = pickle.load(f)
        log(f"pretrained policies and model B loaded from {path}")
    else:
        _, params, calib = pretrain(args.seed, args.pretrain_steps, True)
        log(f"z-conditioned policy pretrained; fox monitor calibration {calib}")
        _, params_b, calib_b = pretrain(args.seed, args.pretrain_steps, False)
        log(f"blind policy pretrained; fox monitor calibration {calib_b}")
        model, _, _ = ppo_plus.init(jax.random.PRNGKey(0), env,
                                    ppo_plus.PlusConfig(**dict(ENV_PPO["tracker"], total_steps=args.pretrain_steps)))
        learned, fit = fit_learned_model(env, model, params, jax.random.PRNGKey(100 + args.seed))
        log(f"model B fitted: {fit}")
        c = dict(params=jax.device_get(params), params_b=jax.device_get(params_b), calib=calib, calib_b=calib_b,
                 fit=fit, net_params=jax.device_get(learned.params), y_mean=np.asarray(learned.y_mean),
                 y_std=np.asarray(learned.y_std), R=np.asarray(learned.R))
        if path:
            os.makedirs(args.cache, exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(c, f)
    return c


def deploy(args, gcfg, c, env, env_blind, out, log):
    cfg = ppo_plus.PlusConfig(**dict(ENV_PPO["tracker"], total_steps=args.pretrain_steps))
    model = ppo_plus.init(jax.random.PRNGKey(0), env, cfg)[0]
    model_b = ppo_plus.init(jax.random.PRNGKey(0), env_blind, cfg)[0]
    learned = LearnedModel(env, MLP((128, 128), 3, 1.0, "tanh"), c["net_params"], jnp.asarray(c["y_mean"]),
                           jnp.asarray(c["y_std"]), jnp.asarray(c["R"]) + jnp.diag(noise_var(gcfg)))
    rng = np.random.default_rng(0)                  # the same device list as foxbench
    devs = [(float(rng.uniform(*env.gain_range)), float(rng.uniform(*env.offset_range))) for _ in range(args.devices)]
    gains, offsets = jnp.asarray([d[0] for d in devs], jnp.float32), jnp.asarray([d[1] for d in devs], jnp.float32)
    D = args.devices
    results = dict(args=vars(args), method=dataclasses.asdict(gcfg), calib=c["calib"], calib_blind=c["calib_b"],
                   model_b_fit=c["fit"], devices=devs, modes={})
    os.makedirs(out)
    for mode in MODES:
        meas = SimModel(env, gcfg) if mode.endswith("_A") else learned if mode.endswith("_B") else None
        e, prm, mdl = (env_blind, c["params_b"], model_b) if mode == "blind" else (env, c["params"], model)
        run = make_episode(e, mdl, prm["actor"], meas, gcfg, mode)
        m, P, logw, P0 = initial_belief(gcfg, D)
        rets, zerr, rho, probing, nis = [], [], [], [], []
        for ep in range(args.episodes):
            keys = jax.random.split(jax.random.PRNGKey(10_000 * args.seed + ep), D)
            ret, m, P, logw, o = run(keys, gains, offsets, m, P, logw, P0)
            rets.append(np.asarray(ret).tolist())
            zerr.append(np.asarray(o["zerr"]).tolist())
            rho.append(np.asarray(o["rho"]).tolist())
            probing.append(np.asarray(o["probing"]).sum(1).tolist())
            alive = np.asarray(o["alive"])
            nis.append(float(np.asarray(o["nis"])[alive].mean()))
        results["modes"][mode] = dict(returns=rets, zerr=zerr, rho=rho, probing_steps=probing, mean_nis=nis)
        r = np.array(rets)
        log(f"{mode:13s} return per episode: {np.round(r.mean(1), 1).tolist()}   z error end ep 1/5: "
            f"{np.mean(np.array(zerr)[0, :, -1]):.3f}/{np.mean(np.array(zerr)[-1, :, -1]):.3f}   "
            f"probing ep 1: {np.mean(probing[0]):.1f}   mean NIS ep 1: {nis[0]:.1f}")
        with open(os.path.join(out, "results.json"), "w") as f:
            json.dump(results, f)


def main(argv=None):
    args = parse_args(argv)
    t0 = time.perf_counter()
    log = lambda msg: print(f"[{time.perf_counter() - t0:6.0f}s] {msg}", flush=True)
    env, env_blind = DriftTracker(show_z=True), DriftTracker(show_z=False)
    c = pretrained(args, env, log)
    variants = ([(_parse_overrides(v.rsplit(":", 1)[0]), v.rsplit(":", 1)[1]) for v in args.variants] if args.variants
                else [({k: eval(v) for k, v in (s.split("=", 1) for s in args.set)}, args.tag)])
    stamp = f"{datetime.now():%Y%m%d-%H%M%S}"
    for overrides, tag in variants:
        gcfg = GriffinConfig(**overrides)
        out = os.path.join(run_root(args.out, "hippogriff"), f"seed{args.seed}_{stamp}{'_' + tag if tag else ''}")
        log(f"--- variant {tag or 'default'}: {overrides}")
        deploy(args, gcfg, c, env, env_blind, out, log)


if __name__ == "__main__":
    main()
