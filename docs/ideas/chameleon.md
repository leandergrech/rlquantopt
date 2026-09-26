# 🐲 i03 chameleon: a manager with a map, a worker that follows

<span class="stamp stamp--archived">📦 archived</span> *2026-09-25, v2.17.0 · smoke test only, too slow to run at scale · superseded by [dragonfly](dragonfly.md)*

**Question.** Can a policy learn faster if it only follows latent tasks set by a jumpy world model, which
chooses between exploiting what it predicts and exploring in directions that never undo it?

**Answer.** Not testable at scale in this form: at about 47-55 s per update (a 20M-step run would take
more than a day), and in the smoke test its deterministic test episodes overshot the amplitude bound
within the first few steps. It was superseded by [dragonfly](dragonfly.md).

## Design (`agents/chameleon.py`)

- **Encoder** \(z = E(s)\), 16 dimensions, with VICReg variance and covariance terms, so the latent axes
  are decorrelated and unit-scale and similar states map to nearby latents.
- **History**: a GRU over \([z_t, a_{t-1}, r_{t-1}]\), reset at episode start: a summary of any length.
- **Jumpy model** \(F(z, c, a, \gamma)\): in one call, the displacement to the γ-discounted future
  latent \(y_t = \sum_{j \ge 1} (1-\gamma)\gamma^{j-1} z_{t+j}\), trained on γ ∈ {0.5, 0.8, 0.9, 0.95, 0.99}.
  Q and V heads regress the TD(λ) return of the env reward at that γ.
- **Planner**, every 8 steps: exploit direction \(u = \nabla_z V / |\nabla_z V|\); explore direction ξ⊥,
  a uniform random unit vector orthogonal to *u* (maximum entropy there, and exploring never undoes
  exploiting); task \(g = \sqrt{1-\epsilon}\,u + \sqrt{\epsilon}\,\xi_\perp\), with ε from a learned Beta
  gate trained by a clipped PPO surrogate on the GAE advantage of the env reward.
- **Worker** \(\pi(a \mid s, z, g)\) proposes 8 candidates; the one whose predicted jump best aligns with
  *g* is executed. Its reward: env reward plus \(\cos(z_{t+1} - z_t, g)\).
- **Schedule**: the model trains on every PPO minibatch; the first 5 updates train the model only.
  Leaky ReLU throughout.

## Smoke test (330k steps)

| | |
| --- | --- |
| Time per update | 47 s (machine about 2× oversubscribed); data collection 4.5 s of it, the model's training the rest |
| Mean reward per step | −14.7 at the start, best −4.7, −6.1 at the end |
| Exploration share ε | 0.50 → 0.08 (the gate learns to exploit) |
| Deterministic test episode | overshoots within 3 of 333 steps (return −18.8); without the search, within 9 |

## Outcome

- **The task reward pulls towards overshooting.** The latent includes the pulse amplitudes, so large
  amplitude changes are the easiest way to "move along the task"; the task reward pays a little every
  step, the overshoot costs −20 once. The search made it worse: it ranked candidates by alignment only,
  ignoring their value.
- **Most of the time went into a wasted computation**: the model's gradient was computed on every
  minibatch and then discarded outside its epochs (fixed later with `lax.cond`).
- Its useful parts (the jumpy γ-model, orthogonal exploration) went into [dragonfly](dragonfly.md).

**Records:** `results/i03_chameleon/` (smoke-test progress and summary).
