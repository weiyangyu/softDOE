# Historical optimization measurements

These measurements precede the public Q/final A-SOFT-score release update.
They document the earlier optimization pass; current release and CUDA evidence
will be recorded in verification documentation. Paths and test counts below
refer to that preserved historical checkout.

# softDOE 0.2.0 performance and verification

Both `soft()` structures and `a_soft()` now use compiled CPU search and a
native Metal implementation. This removes the remaining Python/PyTorch
control overhead. The package version remains **0.2.0**. PyDOE's centered
maximin LHD (`criterion="centermaximin", iterations=100`) is retained.

## Why Python was slower than MOFAT

MOFAT's expensive work runs in compiled C++/Metal code. Choosing Python
instead of R does not automatically make those calculations faster.
The original Python strict-Metal implementation launched hundreds of
thousands of small tensor operations. The first fix fused the Q kernels,
but search control still crossed Python/PyTorch boundaries thousands of
times, with repeated allocations and buffer uploads.

For the reported 20-factor, 6-base-run case, the original profile recorded
663,520 `torch.log1p` calls, 22,308 scalar readbacks, and 3,422 CPU score
recomputations. An isolated original run took 39.465 seconds. The first
fused-kernel fix reduced this to roughly 3.5–4.7 seconds. The current native
Metal search takes roughly 1.5 seconds on the same Apple M4.

## Changes to Q evaluation

- **CPU:** compiled search and augmentation loops, reusable Cholesky/solve
  workspaces, and cached one-dimensional kernel factors for SOFT's fixed
  coordinate grid. The cache is capped at 64 MiB. macOS uses system Accelerate
  consistently across Python/SciPy versions. Other platforms use SciPy's
  exported Cython BLAS/LAPACK capsules with checked integer ABI signatures.
  Small Cholesky factorizations use a fixed recursive panel order. Normal
  correlations use direct column scaling; underflow falls back to log scaling.
- **Metal:** compiled C++ search controls fused Metal kernels directly.
  Shared buffers persist across candidates, without PyTorch tensor
  allocation or per-step Python calls. An accepted incremental update and
  the next pair are submitted together. Both standard and strict search,
  and augmentation, use this path. PyTorch is no longer required on macOS.
- **CUDA:** correlation calculations batch dimensions with bounded
  temporary tensors. Strict left/right candidates share batched solves,
  one score readback, and only the accepted full state is materialized.
  CUDA still uses optional PyTorch float64. Hardware speed is unverified.
- **All paths:** stable log scaling and incremental Schur/Cholesky updates
  are retained. Final updates whose Q values will never be used are
  omitted, while candidate factorization still checks numerical validity.

Small Metal correlation factorizations use CPU double precision; the
large scaled solves use GPU float32, as in MOFAT. This is part of Q
calculation. Neither SOFT structure uses CPU rescoring to break GPU ties.
Both Metal and CUDA retain MOFAT's tolerance on log-Q:
`sqrt(float32 epsilon) * max(1, abs(a), abs(b))`. Strict ties choose the
left block; tied swaps keep the incumbent. CPU comparisons are unchanged.
A-SOFT base generation follows these rules; its separate augmentation-point
selection retains CPU-double checks of near-tied candidates.

All 286 strict constructions and the original column-swap search remain
in the 20-by-6 optimization. No iteration count or evaluation set was reduced.

## Public-function benchmarks

Apple M4; Python 3.14.7, NumPy 2.5.3, SciPy 1.18.1, pydoe 1.5.0.
Baseline is the previous 0.2.0 build **after** the first strict-Metal fix,
not the original 40-second implementation. Calls use `seed=123` and
include PyDOE preparation and public-function overhead, after runtime
initialization. Baseline has two repetitions; updated has three. Tables
show medians. The one cold PyDOE import at p=4 was excluded in each build.
These are local observations, not statistical speed guarantees.

For p=20, l=6, m=500; generated A-SOFT adds 10 points from N=10,000:

| Function | Structure | Backend | Previous seconds | Updated seconds | Speedup |
| --- | --- | --- | ---: | ---: | ---: |
| soft | standard | CPU | 0.785 | 0.381 | 2.06× |
| soft | strict | CPU | 2.720 | 1.289 | 2.11× |
| a_soft | standard | CPU | 0.856 | 0.420 | 2.04× |
| a_soft | strict | CPU | 2.931 | 1.352 | 2.17× |
| soft | standard | Metal | 1.109 | 0.339 | 3.27× |
| soft | strict | Metal | 4.629 | 1.474 | 3.14× |
| a_soft | standard | Metal | 1.133 | 0.342 | 3.32× |
| a_soft | strict | Metal | 3.395 | 1.478 | 2.30× |

Supplied-base A-SOFT (126-by-20 base, 20 added points, N=10,000):

| Backend | Previous seconds | Updated seconds | Speedup |
| --- | ---: | ---: | ---: |
| CPU | 0.0957 | 0.0528 | 1.81× |
| Metal | 0.1027 | 0.0225 | 4.57× |

For p=4, l=4, m=64; A-SOFT adds 4 points from N=512:

| Function | Structure | Backend | Previous seconds | Updated seconds |
| --- | --- | --- | ---: | ---: |
| soft | standard | CPU | 0.00284 | 0.00217 |
| soft | strict | CPU | 0.00759 | 0.00273 |
| a_soft | standard | CPU | 0.00371 | 0.00273 |
| a_soft | strict | CPU | 0.00826 | 0.00330 |
| soft | standard | Metal | 0.05401 | 0.00829 |
| soft | strict | Metal | 0.03438 | 0.01811 |
| a_soft | standard | Metal | 0.04563 | 0.00766 |
| a_soft | strict | Metal | 0.04256 | 0.01709 |

All 18 complete benchmark designs match the previous build exactly.
CPU is still faster for some sizes, including this strict 20-factor case.
GPU submission overhead and the amount of parallel work determine the
crossover. `auto` preserves MOFAT's platform policy; it does not time the
backends to select the fastest one. Explicit `backend="cpu"` is available.

Raw repetitions and environment information are saved in
`benchmarks/2026-10-02.json`. To repeat the public-function measurements:

```bash
python tools/benchmark_workflows.py --backend cpu metal --repeats 3 --output /tmp/softdoe-workflows
```

Use `--backend cpu cuda` on a Linux NVIDIA host. Run benchmarks sequentially.

## Controlled MOFAT comparison

A fresh sequential run (R 4.5.2, MOFAT 2.0) gives these median optimizer
times, excluding discovery and PyDOE preparation:

| Backend | Updated softDOE | MOFAT |
| --- | ---: | ---: |
| CPU | 1.296 s | 1.457 s |
| Metal | 1.541 s | 3.112 s |

The strict comparison uses identical A and Sobol matrices, bypassing
cross-language differences in the random LHD provider. The complete
126-by-20 Metal design matches the MOFAT reference within 2e-14 coordinate
tolerance. Updated log-Q is -105.07073974609375; MOFAT reports
-105.07074737548828 (difference 7.63e-6). Independent CPU evaluation of
that design gives -105.07075754299358. These pass the 3e-5 log-score tolerance.
The standard p=10, l=4, m=64 Metal reference also matches in full.
CPU fixtures retain their 2e-12 score tolerance and full-design agreement.

```bash
python tools/benchmark_strict.py --backend metal --output /tmp/softdoe-bench
Rscript tools/benchmark_strict.R /tmp/softdoe-bench /tmp/mofat-bench.json
```

The R script only reads the installed MOFAT package and writes its specified
output. Equal R/Python random seeds alone do not imply equal designs.

## Verification and limits

- Final source-archive installation tested outside the checkout on Python
  3.14 with actual Metal: **167 passed, 10 skipped**. Skips were the opt-in
  historical diagnostic and nine absent-CUDA-device cases. The earlier
  source-tree run also passed (166 tests, before the extra batching case).
- Source archive and Python 3.9 wheel build and `twine check` passed;
  installed dependencies passed `uv pip check` (24 packages).
- Python 3.9 source suite without PyTorch: 151 passed, 24 skipped. Native
  Metal, both public structures, augmentation, and MOFAT parity passed;
  optional PyTorch checks skipped.
- Independent SciPy tests cover full and incremental native Q, cached
  standard/strict search, duplicate rows, underflow, and supplied designs.
- Both standard and strict native Metal tests block PyTorch imports.
  CPU fallback tests verify that failures restart the original workflow.
- CUDA batching passes portable host-tensor arithmetic and search tests.
  Actual CUDA performance and device execution are unverified on this Mac.
- Metal uses float32, so exact CPU/GPU scores and near-tied design choices
  are not universally identical. CPU stays float64. Cross-platform BLAS
  versions can also affect extremely close comparisons.

All 162 MOFAT reference files match their baseline SHA-256 hashes. Previous
0.2.0 artifacts are preserved in `dist/archive/before-full-optimization/`.

## Installation

These measurements describe a historical build. Use the current
[installation guide](../README.md#install) and
[release verification](verification.md) for installation and testing.

## File scope

Input/output project:
the softDOE repository.
Read-only reference:
the separate MOFAT repository.
Output: `dist/0.2.0/softdoe-0.2.0.tar.gz` and the Python 3.9 macOS wheel.

Changed in this optimization pass:

- Runtime additions: `_native.hpp`, `_native.cpp`, `_native_metal.mm`,
  `_metal.py`, `_cuda.py` under `src/softDOE/`.
- Runtime edits: `_soft.py`, `_augment.py`, `_engine.py`, `_backend.py`,
  `_strict.metal`, `_strict_metal.py` in the same directory.
- Packaging: `setup.py`, `MANIFEST.in`.
- Tests: added `test_native_cpu.py`, `test_native_metal.py`,
  `test_cuda_batching.py`; updated `test_strict_metal.py`'s fallback hook.
- Evidence/tools: `tools/benchmark_workflows.py`, `benchmarks/2026-10-02.json`.
- Docs: this report, `README.md`/`.html`, `tests/tests.md`/`.html`,
  `IMPLEMENTATION_REPORT.md`, `task_plan.md`, `notes.md`.

Unchanged: all MOFAT files, `_core.cpp`, public signatures and version,
PyDOE initialization, Sobol generator/data, screening measures, validation,
and existing MOFAT numerical fixtures. Nothing was uploaded to PyPI.
