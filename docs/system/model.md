# Model and simulator: `rlquantopt/jx/physics.py`

## The model

Two fixed-frequency transmons (q0, q1) coupled through a tunable coupler (c), each truncated to
three levels, in a frame rotating at \(\omega_r = 6\) GHz (paper eq. 1):

\[
H(t) = \Delta_c b^\dagger b + \frac{\alpha_c}{2} b^{\dagger 2} b^2
     + \sum_{j=0,1}\Big[\Delta_j a_j^\dagger a_j + \frac{\alpha_j}{2} a_j^{\dagger 2} a_j^2
     + g_j (a_j b^\dagger + a_j^\dagger b)\Big] + u(t)\, b^\dagger b ,
\]

with \(\Delta = 2\pi(\omega - \omega_r)\). Frequencies are in GHz, energies in rad/ns, time in
ns. The control \(u(t)\) (rad/ns) moves the coupler frequency. Parameters, as in the paper run:

| | qubit 0 | qubit 1 | coupler |
| --- | --- | --- | --- |
| \(\omega/2\pi\) | 5.0311 GHz | 5.8899 GHz | 7.445 GHz |
| \(\alpha/2\pi\) | −324 MHz | −235 MHz | −230 MHz |
| \(g/2\pi\) | 100 MHz | 71.4 MHz | |

!!! warning "The paper's Table 1 pairs these differently"
    Table 1 lists 324 MHz and 100 MHz next to 5.8899 GHz. That pairing makes every stored pulse
    fail (J_T 1e-2 to 6e-2); the pairing above is the one in the code and the one that
    reproduces the paper's results.

## The key idea: excitation-number sectors

The Hamiltonian commutes with the total excitation number \(N = a_0^\dagger a_0 + a_1^\dagger a_1 + b^\dagger b\)
(the RWA removes all terms that change it). The four computational states live in fixed sectors:

| Computational state | N | Sector basis (kets \|q0 q1 c⟩) | Dimension |
| --- | --- | --- | --- |
| \|000⟩ | 0 | \|000⟩ | 1 |
| \|010⟩, \|100⟩ | 1 | \|001⟩, \|010⟩, \|100⟩ | 3 |
| \|110⟩ | 2 | \|002⟩, \|011⟩, \|020⟩, \|101⟩, \|110⟩, \|200⟩ | 6 |

So instead of a 27-dimensional ODE, each 50 ps sample is two small, exact matrix exponentials.
\|000⟩ is an eigenstate with energy 0 and never needs propagating. A test
(`test_hamiltonian_conserves_excitations`) checks that the full Hamiltonian really has no matrix
element between different N.

## Building the sector Hamiltonians

The full 27×27 drift and control are built with Kronecker products (as in v1 `ZCQubits`), then
restricted to the sector indices. `SECTOR1_IDX = [1, 3, 9]` and `SECTOR2_IDX = [2, 4, 6, 10, 12, 18]`
are the full-space indices \(9 q_0 + 3 q_1 + c\) of the sector kets.

```python
--8<-- "rlquantopt/jx/physics.py:sector_hamiltonian"
```

`params` may carry traced arrays (for example `omega_s` inside a `vmap`), which is how sweeps and
domain randomisation get their per-episode Hamiltonians.

## Exact propagation

For a piecewise-constant pulse, each sample applies \(e^{-i H(u_k) \Delta t}\). With
\(H = V \operatorname{diag}(w) V^\dagger\) from `eigh`, this is exact:

```python
--8<-- "rlquantopt/jx/physics.py:propagate"
```

Points to notice:

- `SectorState` holds `U1`, the 3×3 propagator of the N=1 sector (its columns are the images of
  \|001⟩, \|010⟩, \|100⟩), and `psi2`, the image of \|110⟩ in the N=2 sector.
- `jax.lax.scan` loops over the samples inside the compiled function; there is no Python loop.
- The control is diagonal in the Fock basis (\(b^\dagger b\)), so it enters as `jnp.diag(u * control)`.
- `eigh` is used instead of `jax.scipy.linalg.expm` because it is exact for Hermitian matrices,
  works on GPU and is differentiable.

## The realised gate

The v1 environment defines the realised 4×4 gate as \(G_{ij} = \langle \psi_i(t) | b_j \rangle\)
(QuTiP's `Qobj.overlap`), i.e. the complex conjugate of the usual matrix element. We keep that
convention so metrics match v1 exactly:

```python
--8<-- "rlquantopt/jx/physics.py:realised_gate"
```

Because propagation never mixes sectors, \(G = 1 \oplus V_{2\times2} \oplus e\): block diagonal.
This is what makes the [gate metrics](metrics.md) closed-form.

## Precision

`rlquantopt/jx/config.py` turns on float64 when the package is imported. Set `RLQO_X64=0` for
float32. Do not train in float32: over 1000 samples the rounding error is ~1e-4 in J_T, the level
the agent optimises. On the paper's RL pulse, float32 reports a minimum J_T of 8e-6 where the true
value is 9.5e-5.

## Verified against

| Test | Tolerance |
| --- | --- |
| Drift and control operators vs v1 `ZCQubits` | 1e-12 |
| Sector propagation vs full 27×27 `scipy.linalg.expm` | 1e-10 (1e-13 observed) |
| Sector propagation vs QuTiP `SESolver` stepping, as in v1 | 1e-7 (the ODE tolerance) |
