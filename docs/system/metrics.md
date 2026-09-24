# Gate metrics: `rlquantopt/jx/metrics.py`

The reward and GRAPE both use the paper's cost (eq. 2):

\[
J_T = 1 - \frac{w_C\, C + w_U\, U}{w_C + w_U}, \qquad w_C = 1,\; w_U = 3,
\]

where \(C\) is the gate concurrence (1 for a perfect entangler) and \(U = \operatorname{Tr}(G^\dagger G)/4\)
is the unitarity (1 when nothing leaks out of the computational subspace). The step reward is
\(-\log_{10} J_T\), so a reward of 4 means \(J_T = 10^{-4}\).

## Weyl-chamber coordinates in closed form

The concurrence comes from the Weyl-chamber coordinates \((c_1, c_2, c_3)\), which v1 computes
with `weylchamber.c1c2c3` (Childs et al., PRA 68, 052311). That algorithm needs the eigenvalues of
\(U \tilde U / \sqrt{\det U}\) with \(\tilde U = (\sigma_y\otimes\sigma_y) U^T (\sigma_y\otimes\sigma_y)\).
A general 4×4 `eig` is not available on GPU in JAX, and it is awkward to differentiate.

Our gate is block diagonal, \(G = \operatorname{diag}(g_{00}, V, e)\) with
\(V = \begin{pmatrix} a & b \\ c & d \end{pmatrix}\). Working out \(\tilde G\) gives
\(\operatorname{diag}\big(e, \begin{pmatrix} d & b \\ c & a\end{pmatrix}, g_{00}\big)\), so

\[
G\tilde G = \operatorname{diag}\Big(g_{00} e,\;
\begin{pmatrix} ad+bc & 2ab \\ 2cd & ad+bc \end{pmatrix},\; g_{00} e\Big),
\]

whose eigenvalues are \(g_{00} e\) (twice) and \(ad + bc \pm 2\sqrt{abcd}\). The rest of the
algorithm (phases, sorting, the \(n = \mathrm{round}(\sum S)\) shift, mapping into the chamber)
follows `weylchamber` line by line:

```python
--8<-- "rlquantopt/jx/metrics.py:weyl"
```

- `digits=8` reproduces `weylchamber`'s rounding exactly. That rounding has zero gradient, so GRAPE
  calls it with `digits=None`.
- \(\sqrt{abcd}\) has an infinite slope when an off-diagonal element is exactly 0 (for example at
  \(u = 0\)); GRAPE replaces the resulting NaN gradients with 0.
- Inside the perfect-entangler polyhedron the concurrence is exactly 1 and flat, so the gradient
  there comes only from the unitarity.

## Concurrence, unitarity and J_T

```python
--8<-- "rlquantopt/jx/metrics.py:cost"
```

!!! note "Why the concurrence can look noisy"
    When \(|b|, |c| \ll 1\) (early in a pulse), the eigenvalue phases are ill-conditioned and the
    concurrence changes by ~1e-5 for tiny changes of the state. The v1-equivalence test therefore
    compares concurrence at 3e-4 while amplitudes are compared at 1e-5. On exact gates the metric
    agrees with `weylchamber` to 1e-8.

The perfect-entangler target and its named-gate alternatives are discussed in
[Named gates and a richer model](../next/named-gates.md).
