"""Numerical precision for rlquantopt.jx.

Double precision is on by default because gate errors of 1e-4 accumulate over
1000 propagation steps. Set RLQO_X64=0 before importing to train in single
precision (much faster on consumer GPUs).
"""
import os

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", os.environ.get("RLQO_X64", "1") != "0")


def cdtype():
    return jnp.complex128 if jax.config.jax_enable_x64 else jnp.complex64


def fdtype():
    return jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
