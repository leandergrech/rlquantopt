"""Idea iterations: every new idea gets a number and an animal codename, in alphabetical order.

``python -m rlquantopt.jx.train --idea axolotl ...`` writes to ``runs/i01_axolotl/<algo>_<timestamp>_s<seed>_<tag>/``,
so an idea's runs sit together and never land on an earlier idea's (or the pre-ideas runs in ``runs/``).
To start the next idea, add an entry with the next number and a codename starting with the next letter.
"""
from typing import NamedTuple


class Idea(NamedTuple):
    number: int
    summary: str


IDEAS = {
    "axolotl": Idea(1, "Autoregressive delta policy (one head, fed its own output K times per step) and a "
                       "sample judge: an MLP of (obs, action, reward) predicting mean and std of how well each "
                       "sample's PPO gradient agrees with the rest of the batch; confident verdicts reweight "
                       "the PPO loss through a separate guide term. Algo: ppo_judge."),
    "badger": Idea(2, "axolotl with a judge that also sees the lambda-return, is trained only every few policy "
                      "updates, and whose guide weight halves with its age since training; PPO early-stops "
                      "an update's epochs once approx KL exceeds 1.5 * target_kl. Algo: ppo_judge."),
    "chameleon": Idea(3, "Fresh start: a jumpy, gamma-conditioned latent world model with a GRU history sets latent "
                         "tasks g = sqrt(1-eps) u + sqrt(eps) xi (u = value-gradient direction, xi = random "
                         "orthogonal direction, eps from a learned Beta gate); the policy proposes candidate "
                         "actions, the model re-ranks them by predicted progress along g, and the policy is "
                         "rewarded for following g. The model trains on every PPO minibatch after a model-only "
                         "warm-up. Leaky ReLU throughout. Algo: chameleon."),
    "dragonfly": Idea(4, "Faster chameleon: history as parallel leaky-integrator traces (momenta of the latent); "
                         "exploration as a per-dimension Ornstein-Uhlenbeck walk (constant entropy, persistence "
                         "tau_d, spread sigma_d) set by a small learned guide that sees the measured latent "
                         "entropy, a pessimistic ensemble value and its position p relative to a canonical "
                         "golden cache of best reward/return; lower-bound Q vetoes risky candidates in the "
                         "search; self-imitation of golden actions. Algo: dragonfly."),
    "echidna": Idea(5, "Back to plain PPO, adding dragonfly's ingredients one at a time: OU (temporally correlated) "
                       "exploration noise with the policy conditioned on the previous noise, a pessimistic "
                       "Q-ensemble veto of risky candidates, and golden-cache self-imitation. Smoke-tested in "
                       "isolation on toy envs (toy_envs.py, toybench.py) before the gate env. Algo: ppo_plus."),
    "fox": Idea(6, "Sample-cost-aware per-device adaptation (black-box): PPO fine-tuning plus an improvement-"
                   "equivalent model, a Bayesian regression of return gained per unit step on the step direction "
                   "in a basis of recent PPO steps (the likelihood of |g| cos(g, step)); confident steps along g, "
                   "a cache of unexplained updates kept until explained, and a marginal-value stopping rule "
                   "against the cost of samples. Benchmarked on drifted toy devices first (foxbench.py)."),
    "gecko": Idea(7, "Hardware-measurable replication of the large-coupler-drift experiment: the reward is the "
                     "average gate fidelity to sqrt(iSWAP) after free virtual-Z corrections (leakage included), "
                     "and the agent observes readout populations of |01>, |10>, |11> and two-qubit Pauli "
                     "expectations of |0+>, |+0>, |+1> instead of state amplitudes (--objective sqrt_iswap "
                     "--obs-mode measured; exact expectations first, finite shots next). Algo: ppo."),
    "hippogriff": Idea(8, "Dragonfly's structure with fox's empiricism: a drift-conditioned PPO policy pi(a|s,z) "
                          "pretrained with domain randomisation (fox's model monitoring), then at deployment a "
                          "recursive Bayesian (EKF) belief over the drift whose uncertainty shrinks with every "
                          "interaction, information-gain probing inside a trust region that widens as the "
                          "uncertainty shrinks, sigma-point safety, then pi(a|s, belief). Grey-box (A, the known "
                          "model) and learned (B) measurement models. Toy first (hippogriff.py)."),
    "ibis": Idea(9, "gecko made data-lean: the pulse amplitude is clipped at its bound with a penalty on the excess "
                    "instead of ending the episode (--oob-mode clip); observations are finite-shot estimates, "
                    "N shots per measurement setting, 30 settings per step (--shots N); smaller policy networks "
                    "(--hidden). Question: how few shots per pulse and how small a network still give a good "
                    "warm start under the large coupler drift. Algo: ppo."),
}


def run_root(root: str, codename: str) -> str:
    if codename not in IDEAS:
        raise SystemExit(f"unknown idea {codename!r}; register it in rlquantopt/jx/ideas.py (known: {', '.join(IDEAS)})")
    return f"{root}/i{IDEAS[codename].number:02d}_{codename}"
