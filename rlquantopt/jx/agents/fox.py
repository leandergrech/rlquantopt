"""Idea i06 "fox": sample-cost-aware adaptation with an improvement-equivalent model on top of PPO.

The improvement equivalence principle: a model does not need to predict the world, or even the return;
it only needs to predict which way the return improves. After every PPO update we observe the step that
was taken, dtheta_k, and (one batch later) the change in return it caused, dJ_k. The model is a Bayesian
linear regression, over the last ``window`` updates plus an outlier cache,

    dJ_k / |dtheta_k|  ~  N( <g, dtheta_k / |dtheta_k|>,  s^2 ),     g = D^T beta,  beta ~ N(0, I / prior),

i.e. the return gained per unit step is ``|g| cos(g, step)``: the posterior of g is the model's belief
about the improvement direction and its likelihood is that of the observed cosines with the realised
improvements. D is an orthonormal basis of the last ``basis`` PPO proposals, in the actor's parameter
space (the step directions the learner actually explores).

Per update:

1. **Exploit.** If the improvement along the posterior mean direction is confidently positive
   (``z > z_min``), fox adds ``step_frac * |PPO step| * g / |g|`` to PPO's step.
2. **Outlier cache.** Observations that leave the window while unexplained (standardised residual >
   ``out_z``) are kept in a cache with weight ``out_weight``; a cached observation is dropped once the model
   explains it (residual < ``in_z``), or when the cache is full (oldest first).
3. **Sample cost.** Each update costs one batch of samples, ``cost_per_update`` in return units. Fox stops
   (marginal value theorem) when the optimistic predicted gain of another update,
   ``|step| (pred + kappa sd)``, has stayed below the cost for ``patience`` updates, after ``min_updates``.

Returns are measured from the rollout batches the learner collects anyway (mean reward per step times the
episode length, an episode-return estimate), so the model itself costs no extra samples.
"""
from dataclasses import dataclass, field

import numpy as np
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree


@dataclass
class FoxConfig:
    basis: int = 8
    window: int = 24
    cache_size: int = 16
    prior: float = 1.0                 # initial prior precision, relative to the data scale (then learned)
    noise_floor: float = 1e-6
    out_z: float = 2.0
    in_z: float = 1.0
    out_weight: float = 2.0
    z_min: float = 1.0
    step_frac: float = 0.5
    exploit: bool = True
    cost_per_update: float = 1.0       # return units per update (one batch of samples)
    kappa: float = 1.0
    min_updates: int = 10              # before exploiting (and the default stopping rule)
    patience: int = 3
    stop_rules: tuple = (10, 3)        # stopping points recorded for these minimum update counts


@dataclass
class Obs:
    step: np.ndarray        # the actor step actually applied
    dJ: float
    k: int                  # update index (age)


class ImprovementModel:
    """Bayesian regression of return gained per unit step on the step's direction, in a basis of PPO steps."""

    def __init__(self, cfg: FoxConfig):
        self.cfg = cfg
        self.proposals: list = []
        self.window: list = []
        self.cache: list = []
        self.D = None
        self.post = None

    def add_proposal(self, dtheta):
        n = np.linalg.norm(dtheta)
        if n > 0:
            self.proposals = (self.proposals + [dtheta / n])[-self.cfg.basis:]
            q, r = np.linalg.qr(np.stack(self.proposals, 1))
            keep = np.abs(np.diag(r)) > 1e-8
            self.D = q[:, keep].T                                         # (m, P) orthonormal rows

    def _data(self):
        obs = self.window + self.cache
        w = np.array([1.0] * len(self.window) + [self.cfg.out_weight] * len(self.cache))
        return obs, w

    def features(self, step):
        u = step / (np.linalg.norm(step) + 1e-12)
        return self.D @ u

    # --8<-- [start:improvement_model]
    def fit(self):
        obs, w = self._data()
        if self.D is None or len(obs) < 2:
            self.post = None
            return
        Phi = np.stack([self.features(o.step) for o in obs])
        y = np.array([o.dJ / (np.linalg.norm(o.step) + 1e-12) for o in obs])
        m = Phi.shape[1]
        # Evidence maximisation (MacKay): learn the prior precision alpha and the noise precision b from the
        # data, so the scale of "return per unit step" is not fixed a priori
        var_y = max(float(np.mean(y ** 2)), self.cfg.noise_floor)
        alpha, b = self.cfg.prior / var_y, 1.0 / var_y
        n_eff = float(w.sum())
        for _ in range(30):
            A = alpha * np.eye(m) + b * (Phi.T * w) @ Phi
            A_inv = np.linalg.inv(A)
            mu = b * A_inv @ (Phi.T * w) @ y
            gamma = float(np.clip(m - alpha * np.trace(A_inv), 1e-6, m))
            alpha = gamma / max(float(mu @ mu), 1e-12)
            rss = float(np.sum(w * (y - Phi @ mu) ** 2))
            b = max(n_eff - gamma, 1e-6) / max(rss, self.cfg.noise_floor * n_eff)
        A_inv = np.linalg.inv(alpha * np.eye(m) + b * (Phi.T * w) @ Phi)
        mu = b * A_inv @ (Phi.T * w) @ y
        self.post = (mu, A_inv, 1.0 / b)

    def predict(self, step):
        """Predicted return gain per unit step length along ``step``, and its standard deviation."""
        mu, A_inv, s2 = self.post
        phi = self.features(step)
        return float(phi @ mu), float(np.sqrt(phi @ A_inv @ phi))

    def direction(self):
        """Posterior mean improvement direction (unit, parameter space) and gain per unit step along it."""
        mu, A_inv, _ = self.post
        g = self.D.T @ mu
        n = np.linalg.norm(g)
        return (g / n if n > 0 else g), float(n)
    # --8<-- [end:improvement_model]

    def residual_z(self, o: Obs):
        mu, A_inv, s2 = self.post
        phi = self.features(o.step)
        y = o.dJ / (np.linalg.norm(o.step) + 1e-12)
        return abs(y - phi @ mu) / np.sqrt(s2 + phi @ A_inv @ phi)

    def observe(self, o: Obs):
        """Add an observation; move unexplained ones leaving the window to the cache; prune explained ones."""
        self.window.append(o)
        if len(self.window) > self.cfg.window:
            old = self.window.pop(0)
            if self.post is not None and self.D is not None and self.residual_z(old) > self.cfg.out_z:
                self.cache.append(old)
        self.fit()
        if self.post is not None and self.cache:
            self.cache = [c for c in self.cache if self.residual_z(c) >= self.cfg.in_z][-self.cfg.cache_size:]
            self.fit()


class FoxAdapter:
    """Wraps a PPO update function; call ``step()`` once per update. Only the actor is steered by the model."""

    def __init__(self, update, runner, opt, cfg: FoxConfig, episode_len: int):
        self.update, self.runner, self.opt, self.cfg = update, runner, opt, cfg
        self.model = ImprovementModel(cfg)
        self.episode_len = episode_len
        self.k = 0
        self.J_prev = None
        self.last_step = None
        rules = sorted(set(cfg.stop_rules) | {cfg.min_updates})         # the default rule is always recorded
        self.below = {m: 0 for m in rules}
        self.stops = {m: None for m in rules}
        self.stopped_at = None
        self.log = []

    def _flat_actor(self, params):
        return ravel_pytree(params["actor"])

    def step(self):
        theta, unravel = self._flat_actor(self.runner.params)
        theta = np.asarray(theta)
        runner, opt, stats = self.update(self.runner, self.opt)
        J = float(stats["mean_reward"]) * self.episode_len                # return estimate at theta
        if self.last_step is not None and np.linalg.norm(self.last_step) > 0:     # not while the actor is frozen
            self.model.observe(Obs(self.last_step, J - self.J_prev, self.k - 1))

        ppo_step = np.asarray(self._flat_actor(runner.params)[0]) - theta
        self.model.add_proposal(ppo_step)
        self.model.fit()
        extra = np.zeros_like(ppo_step)
        rec = dict(k=self.k, J=J, n_obs=len(self.model.window), n_cache=len(self.model.cache), exploit=0.0,
                   z=np.nan, gain=np.nan, pred_next=np.nan, pred_mean=np.nan)
        if self.model.post is not None:
            g, gain = self.model.direction()
            _, sd_g = self.model.predict(g)
            z = gain / (sd_g + 1e-12)
            rec.update(z=z, gain=gain)
            if self.cfg.exploit and self.k >= self.cfg.min_updates and z > self.cfg.z_min:
                extra = self.cfg.step_frac * np.linalg.norm(ppo_step) * g
                rec["exploit"] = 1.0
            # Marginal value of one more update of the same size, optimistic
            step = ppo_step + extra
            pred, sd = self.model.predict(step)
            rec["pred_next"] = np.linalg.norm(step) * (pred + self.cfg.kappa * sd)
            rec["pred_mean"] = np.linalg.norm(step) * pred           # for calibration: vs the realised dJ
            for m in self.stops:
                if self.k >= m:
                    self.below[m] = self.below[m] + 1 if rec["pred_next"] < self.cfg.cost_per_update else 0
                    if self.stops[m] is None and self.below[m] >= self.cfg.patience:
                        self.stops[m] = self.k + 1                        # updates spent
            self.stopped_at = self.stops.get(self.cfg.min_updates)
        applied = ppo_step + extra
        params = dict(runner.params, actor=unravel(jnp.asarray(theta + applied, dtype=theta.dtype)))
        self.runner, self.opt = runner._replace(params=params), opt
        self.last_step, self.J_prev = applied, J
        self.k += 1
        self.log.append(rec)
        return stats, rec
