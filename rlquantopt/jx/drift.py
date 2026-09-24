"""Hardware-motivated ranges of qubit-frequency drift for fixed-frequency transmons.

Each range is a half-width in MHz applied independently to both qubit frequencies, with
the source it comes from. See docs/system/drift.md for the discussion.

  in_cooldown    ±0.1 MHz   Within one cooldown, hours to days. Dhieb et al. 2025 (arXiv:2512.18037):
                            "variations confined to a 100 kHz-interval" over 95 h. Consistent with
                            TLS-induced shifts of 5-140 kHz (Schlör et al., PRL 123, 190502, 2019) and
                            1-3 kHz typical, up to 20 kHz (Burnett et al., npj QI 5, 54, 2019).
  recool         ±5.7 MHz   Between cooldowns (thermal cycling to room temperature). Zhang et al.,
                            Sci. Adv. 8, eabi6690 (2022): "recool stability of 5.7 MHz" on IBM
                            multi-qubit processors. 5.7 MHz is 0.11 % / 0.10 % of our qubit
                            frequencies, i.e. the paper's ±0.1 % domain randomisation.
  fab_targeting  ±18.5 MHz  Setting a new device's frequencies after fabrication and laser annealing:
                            "frequency assignment precision of 18.5 MHz" (Zhang et al. 2022).

The numbers are reported as spreads (precision / stability); we use them as half-widths of a
uniform box, which covers about ±1 sigma.
"""
from typing import NamedTuple

import numpy as np


class DriftRange(NamedTuple):
    name: str
    half_width_mhz: float
    timescale: str
    source: str


DRIFT_RANGES = {
    "in_cooldown": DriftRange("in_cooldown", 0.1, "hours to days, one cooldown",
                              "Dhieb et al. 2025, arXiv:2512.18037"),
    "recool": DriftRange("recool", 5.7, "between cooldowns",
                         "Zhang et al., Sci. Adv. 8, eabi6690 (2022)"),
    "fab_targeting": DriftRange("fab_targeting", 18.5, "new device after fabrication",
                                "Zhang et al., Sci. Adv. 8, eabi6690 (2022)"),
}


def box_grid(half_width_mhz, n_per_axis=11):
    """Square grid of (Δω0, Δω1) in MHz covering ±half_width on both qubits."""
    d = np.linspace(-half_width_mhz, half_width_mhz, n_per_axis)
    return d, d


def range_stats(J, name):
    """Nominal-independent summary of a J_T map over one drift range."""
    J = np.asarray(J)
    return {f"{name}_mean_JT": float(J.mean()), f"{name}_worst_JT": float(J.max()),
            f"{name}_median_JT": float(np.median(J))}
