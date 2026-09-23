"""Environment throughput: batched random-action rollouts of the JAX ZCQPEE (env steps per second)."""
import argparse
import time

import jax
import jax.numpy as jnp

from rlquantopt.jx import env as jenv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-envs", type=int, nargs="+", default=[1, 64, 512, 4096])
    p.add_argument("--n-steps", type=int, default=333)
    args = p.parse_args()
    cfg = jenv.EnvConfig()
    print(f"backend={jax.default_backend()} x64={jax.config.jax_enable_x64}")
    for n in args.n_envs:
        @jax.jit
        def run(key):
            keys = jax.random.split(key, n)
            _, states = jax.vmap(jenv.reset, in_axes=(0, None))(keys, cfg)

            def body(carry, k):
                states = carry
                acts = 0.1 * jax.random.normal(k, (n, cfg.act_dim))
                ks = jax.random.split(k, n)
                obs, states, r, *_ = jax.vmap(jenv.step_autoreset, in_axes=(0, 0, 0, None))(ks, states, acts, cfg)
                return states, r.mean()
            _, r = jax.lax.scan(body, states, jax.random.split(key, args.n_steps))
            return r
        run(jax.random.PRNGKey(0)).block_until_ready()
        t0 = time.perf_counter()
        run(jax.random.PRNGKey(1)).block_until_ready()
        dt = time.perf_counter() - t0
        print(f"n_envs={n:6d}  {n * args.n_steps / dt:12.0f} env steps/s")


if __name__ == "__main__":
    main()
