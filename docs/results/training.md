# Training RL agents

Which algorithm, how many samples and how much variation between seeds, with the paper's
environment and budget (20M environment steps, 8,192 steps per update, [128, 128] ReLU networks).

!!! info "In progress"
    PPO seeds 1-3 are training. This page will be updated in place with the seed statistics.

## TRPO vs PPO, seed 123

| | TRPO (paper algorithm) | PPO |
| --- | --- | --- |
| Best J_T | 9.8e-4 | 9.8e-4 |
| Step of the best checkpoint | 19.9M | 4.7M |
| 1 − U at the best step | 1.3e-3 | 1.3e-3 |
| Steps until J_T < 1e-2 | 2.4M | 3.1M |
| Wall time for 20M steps (laptop CPU) | 115 min | 61 min |

PPO reaches the same best gate as TRPO in a quarter of the samples and half the wall time, so
there is no degradation. Its evaluation J_T drifts upwards late in training (3e-3 at 20M steps), so
the best checkpoint, not the last, is the one to keep. For comparison, the paper's agent reached
9.5e-5; both of ours are limited by leakage, which [GRAPE refinement](robustness.md) removes.
