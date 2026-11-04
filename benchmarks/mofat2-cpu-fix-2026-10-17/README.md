# CPU Q scoring correction against MOFAT 2.0

The corrected runtime passes **888/888 CPU/Metal comparisons** and **444/444
Python 3.9 CPU comparisons**. Fresh wheel/source installs on Python 3.9 and
3.14 each pass **334 tests with 53 optional skips**.

This audit investigates and corrects the three CPU final-score disagreements
reported in the [original Mac audit](../mofat2-mac-2026-10-17/README.md).
The original reference package, saved fixtures, comparison tolerances, and
strict CPU search acceptance rule were preserved.

## Cause and correction

Both implementations used the same shared initial ranks and initial-search
QMC points. Arithmetic differences in Python's native CPU scorer changed
the treatment of near-tied candidates:

1. Python computed scaled correlations through `exp(log(K)-max(log(K)))`.
   MOFAT 2.0 uses `K/max(K)` for representable ordinary correlations, avoiding
   a log/exp round trip. Direct scaling alone corrected case 0069.
2. Python summed squared solved entries through one accumulation chain.
   MOFAT/Armadillo uses two chains for these small vectors. Matching that
   summation order corrected 0066 and its structure-fallback duplicate 0427.
3. The first full rerun revealed case 0118 (`p=5,l=4,m=64,N=128,n_aug=4,
   seed=42`, CPU, standard). Vendor Cholesky factors differed by 1–2 floating-point
   units. Fixed recursive Cholesky panels for matrices of at most 64 rows matched
   the reference on Python 3.14, but Python 3.9's SciPy/OpenBLAS still diverged.
4. macOS now uses system Accelerate consistently, independent of SciPy's bundled
   BLAS. That also resolves 0118 on Python 3.9. Other platforms retain SciPy's
   BLAS/LAPACK capsules with checked integer ABI signatures.

The recursive algorithm splits each panel in half, solves the lower block,
updates the trailing symmetric block, and recurses. It follows the algorithm
specified by [LAPACK DPOTRF2](https://www.netlib.org/lapack/explore-html/d1/dd3/group__potrf2_ga7a1158271be5fac6e3d89b7ca8d71a07.html).
Larger matrices retain the backend's existing factorization routine.

The implementation change is in `src/softDOE/_native.hpp`, in
`Linalg`, `CpuState::initialize`, and `CpuState::norm`; `setup.py` links the
built-in Accelerate framework on macOS. Normal alpha=1 correlations use
direct scaling. Any subnormal/zero correlation triggers recomputation of the
entire affected evaluation column in log space. General alpha retains log
scaling. This preserves the existing underflow handling.

This is a correction of numerical evaluation order with a material effect on
greedy search outcomes. It does not change the Q formula, replace the reference
with Python output, add a tie epsilon, or relax final-score acceptance.
Floating-point behavior may still vary across other BLAS libraries or platforms;
the audit establishes the results of the recorded runs, not universal bitwise parity.

## Focused reproduction

All cases use `a_soft`, CPU, `p=3,l=4,m=64,N=128`, and shared integer ranks.

| Case | Structure | Seed | n_aug | Before Python Q | Corrected Python and reference R Q |
| --- | --- | ---: | ---: | ---: | ---: |
| `0066_a_soft_grid` | standard | 1 | 4 | 0.0003705904610785128 | 0.00045444284379107417 |
| `0069_a_soft_grid` | standard | 42 | 1 | 0.0020001412403873446 | 0.0016213145570992992 |
| `0427_a_soft_structure_fallback` | invalid → standard | 1 | 4 | 0.0003705904610785128 | 0.00045444284379107417 |

The [before-results](before-results.csv) and [before-summary](before-summary.json)
record a fresh reproduction of all three failures before changing the runtime.
The new regression file is [test_cpu_q_parity.py](../../tests/test_cpu_q_parity.py),
using a [fixture captured from unchanged R](../../tests/fixtures/mofat_cpu_near_ties.json).
All three regression cases failed before the change and passed after it.
The additional 0118 reference Q is `3.131660324529821e-08`; its intermediate
failing Python Q was `2.0535923945083884e-08`. It now passes on both tested
Python versions. These four trajectory fixtures explicitly target macOS;
seven new analytical/independent-solve tests run on every platform.

The corrected [0066](0066-diagnostic.json), [0069](0069-diagnostic.json),
and [0118](0118-diagnostic.json) diagnostics have `first_decision_divergence: null`.
Both implementations' native searches agree with their step-by-step traces;
the original accept/reject divergences are gone. The final Q values above
match the unchanged R results. The original diagnostics remain in the older
audit directory for comparison.

Run from the repository root after installing the current checkout:

```bash
python -m pytest -q tests/test_cpu_q_parity.py
python tests/run_mofat_experiments.py --profile full --backends cpu --case 0066_a_soft_grid --case 0069_a_soft_grid --case 0427_a_soft_structure_fallback --repeats 3 --output experiment-results/cpu-q-correction
```

Use a new output directory for each run. The ordinary suite also retains independent
kernel-solve, underflow, extreme-input, native/reference, backend and live R checks.

## Independent precision check

An 80-digit `mpmath` evaluation of the mathematical kernel criterion on the
recorded binary input coordinates finds equal scores at the first differing
0066 and 0069 decisions (within `8.5e-81` and exactly at working precision,
respectively). In 0118, candidate minus incumbent log Q is about `-8.48e-16`;
double-precision Python had reported `+1.78e-15` and R reported zero.
Run [precision_check.py](precision_check.py) with `mpmath` installed to repeat
this independent calculation. The [recorded values](high-precision-results.json)
support the diagnosis of
rounding-sensitive search. They do not establish that the R greedy result is a
global optimum or that the correction always increases final Q.

## Verification

- Current Python 3.14.7 suite with actual Metal, optional PyTorch, and live R:
  **353 passed, 34 CUDA-only skips**; [test log](pytest-py314.log).
- Python 3.9.6 full CPU matrix, one timing repetition: **444 passed, 0 failed**;
  [summary](py39-cpu-summary.json), [rows](py39-cpu-results.csv).
- Four analytical one-point cases cover normal, subnormal, and underflowed
  correlations at p=1,142,143,160. Three independent dense solves cover the
  recursive/vendor Cholesky boundary at 63,64,65 rows.

- Final CPU/Metal matrix, three timed repeats: **888 passed, 0 failed**;
  all 888 workflow/input/independent-score checks passed. See [summary](summary.json)
  and [all comparison rows](results.csv).
- Of 826 shared-input comparisons, 825 designs match within `2e-14`; one has an
  equivalent score with a different design. The other valid comparisons use
  32 native initializers. All 30 invalid-input checks pass.
- Maximum Python/R CPU discrepancy on the same design: `1.422e-14` relative Q.
  Maximum Python Metal returned Q versus independent R CPU Q: `2.513e-5`.
- The complete [manifest](manifest.json) is identical to the original audit,
  including all inputs, repeats and backend selections. Captured R function
  bodies are also unchanged. Their raw printed hashes differ because R prints
  process bytecode addresses; [reference-comparison.json](reference-comparison.json)
  records equality after normalizing only those addresses.

Fresh wheel and source installations on Python 3.9.6 and 3.14.7 each passed
**334 tests with 53 optional skips** (PyTorch absent, CUDA unavailable). Strict
metadata, dependencies, assets, isolated imports, live R, actual Metal, examples,
and README checks all passed. All 29 release input hashes match the checkout
and both source archives. See [release-verification.json](release-verification.json).
Local ready-to-review artifacts are in `dist/release-cpu-fix-2026-10-17/`;
nothing was published.

Windows/Linux/CUDA hardware,
Intel macOS, and the optional stress profile are not covered by this Apple M4 run.

## Timing and evidence

The run recorded 1,716 warmups and 5,148 timed `soft`/`a_soft` calls across both
implementations. [timings.json](timings.json) retains all repetitions; R rescoring
is recorded separately from workflow-only timing. Builds and other numerical
jobs were kept out of the timed run.

Representative shared-initializer workloads: `p=20,l=6,m=500,seed=1`; augmentation
uses `n_aug=10,N=10000`. Times are medians of three calls, in seconds.

| Function | Structure | Backend | Python | R workflow | R with scores |
| --- | --- | --- | ---: | ---: | ---: |
| `soft` | standard | cpu | 0.2827 | 0.6670 | 0.6670 |
| `soft` | standard | metal | 0.3652 | 0.5119 | 0.5120 |
| `a_soft` | standard | cpu | 0.3260 | 0.6965 | 0.7377 |
| `a_soft` | standard | metal | 0.3582 | 0.5491 | 0.5887 |
| `soft` | strict | cpu | 1.1896 | 1.3606 | 1.3606 |
| `soft` | strict | metal | 1.6135 | 3.3430 | 3.3430 |
| `a_soft` | strict | cpu | 1.2298 | 1.3391 | 1.4099 |
| `a_soft` | strict | metal | 1.6283 | 3.4091 | 3.4410 |

These local timings do not establish a general speed guarantee. The standard CPU
examples improved relative to the earlier audit; strict examples were similar.
Metal is not uniformly faster than CPU. The selected output scores are in the CSV.

Raw outputs and logs remain at `.work/q-fix/mac-accelerate-final/`. The incomplete
`.work/q-fix/mac-full-final/` run was stopped when Python 3.9 still failed 0118;
it is not acceptance evidence. The first attempted correction's complete run
passed 887/888 and is retained at `.work/q-fix/mac-full-01/` as diagnostic history.
The original audit directory and its checksums were not modified.

## Change scope

- Runtime/build inputs: `src/softDOE/_native.hpp`, `setup.py`.
- New regression files: `tests/test_cpu_q_parity.py`,
  `tests/fixtures/mofat_cpu_near_ties.json`.
- Documentation: `README.md`, `tests/test_README.md`, `docs/experiments.md`,
  `docs/verification.md`, `docs/performance.md`.
- Durable investigation records: `task_plan.md`, `notes.md`, and this audit folder.
- Ignored build/test artifacts: `.work/q-fix/`, new `dist/release-checks/` runs,
  `dist/release-cpu-fix-2026-10-17/`, and rebuilt local extensions.

The installed/sibling MOFAT reference, pre-existing fixtures, historical benchmark
folder, and unrelated working-tree edits were preserved. The user's original
`.venv` editable target is restored; final package checks used fresh isolated
installations. Comparison scripts/gates and CPU/GPU search acceptance rules
were not edited. No commit, push, or publication was performed.
