"""Numerical precision for rlquantopt.jx.

Double precision is on by default and is required for training: in float32 the
rounding error of 1000 propagation steps is ~1e-4 in J_T (the paper's RL pulse
reads J_T_min = 8e-6 instead of 9.5e-5), which is the scale the agent optimises.
RLQO_X64=0 switches to float32 for quick smoke tests only.
"""
import os

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", os.environ.get("RLQO_X64", "1") != "0")


def cdtype():
    return jnp.complex128 if jax.config.jax_enable_x64 else jnp.complex64


def fdtype():
    return jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
