# Mac comparison audit: MOFAT 2.0 and softDOE 0.2.0

The expanded experiment completed **444 configurations on CPU and Metal**:
**888 paired runs, 885 passed and 3 final-score disagreements**. All 888
workflow/input validation checks passed. Every Python score was independently
verified by R on that Python design. The three final-score disagreements remain
failures; the experiment exits with status 1 and no tolerance was relaxed.

## Scope and reproducibility

- 105 valid `soft`, 324 valid `a_soft` (73 supplied-base), and 15 invalid-input
  configurations per backend; 858 valid backend/configuration combinations total.
- One untimed warmup and three timed repetitions in each implementation:
  **5,148 timed public-function calls**, plus 1,716 warmups. Extra independent
  Q and diagnostic calls are not counted as timed workflow repetitions.
- The full matrix covers both structures, two seeds, initializer branches,
  the corner-policy boundary at p=10/11, small/default candidate pools,
  pool exhaustion, supplied boundaries/duplicates and input rejections.
- The optional 460-case stress profile and Windows/Linux/CUDA hardware were
  not run. See [the experiment guide](../../docs/experiments.md) for their commands.

Run from a checkout with the installed Python package and R MOFAT 2.0:

```bash
python tests/run_mofat_experiments.py --profile full --backends cpu metal --repeats 3 --output experiment-results/mac
```

The recorded machine was Apple M4, with Python 3.14.7, R version 4.5.2 (2025-10-31).
The metadata files record dependency versions, backend devices, import/package
paths, BLAS thread environment settings, runtime source hashes, and captured
R functions. CPU threads were limited to one; implementation stages ran sequentially.
This is a descriptive local measurement, not a platform-independent speed guarantee.

## Scores and design agreement

- Maximum relative difference between Python/R CPU evaluation of the **same**
  Python design: `1.435e-14`.
- Maximum relative difference between Python Metal's returned Q and independent
  R CPU evaluation of that design: `2.513e-5`.
- Among 826 shared-input, same-backend comparisons: 822 designs agreed within
  `2e-14`; one different design had an equivalent score; three final scores
  disagreed. The other 32 valid comparisons use independent native initializers
  and are not expected to produce identical designs or scores.
- All 30 invalid-input checks produced the expected rejection in both packages.

Three CPU disagreements (all p=3, l=4, m=64, N=128):

| Case | Seed | n_aug | Python Q | R Q | Normalized-Q difference relative to R |
| --- | ---: | ---: | ---: | ---: | ---: |
| `0066_a_soft_grid` | 1 | 4 | 0.0003705904611 | 0.0004544428438 | -6.573% |
| `0069_a_soft_grid` | 42 | 1 | 0.00200014124 | 0.001621314557 | +7.250% |
| `0427_a_soft_structure_fallback` | 1 | 4 | 0.0003705904611 | 0.0004544428438 | -6.573% |

Case 0427 repeats 0066 through invalid-structure fallback to standard. These
are two distinct search paths. Both CPU implementations use strict `>` to
accept a candidate, matching the baseline. At their first different decisions,
identical candidate designs receive score changes differing by one floating-point
unit: Python `+8.88e-16` versus R `0` (0066), and Python `+8.88e-16` versus
R `-8.88e-16` (0069). Greedy search then follows different paths.

The native optimizer and a step-by-step reference trace agree within each
implementation. Replaying augmentation from either selected base in both
packages agrees within `1.11e-16` in coordinates and `1.03e-15` relative Q.
Thus this evidence identifies sensitivity of the initial CPU search path to
rounding, rather than incorrect Q calculation or a different augmentation rule.
It does **not** establish universal cross-implementation design parity.
Runtime implementation and reference code were preserved.

Reproduce the diagnostics (use a new output filename each time):

```bash
python tests/diagnose_mofat_case.py experiment-results/mac --case 0066_a_soft_grid --output experiment-results/0066-diagnostic.json
python tests/diagnose_mofat_case.py experiment-results/mac --case 0069_a_soft_grid --output experiment-results/0069-diagnostic.json
```

## Representative runtime and score results

Shared initialization; p=20, l=6, m=500, seed=1. Augmentation uses n_aug=10,
N=10000. All times are medians of three calls in seconds. R `a_soft` does not
return Q: its workflow-only and workflow-plus-Q times are both shown. Python
returns Q within the function. For `soft`, the returned scores are already
included in the workflow time. Shared starting-matrix generation is excluded
from these rows; initializer-inclusive timings are labeled `native` in the CSV.

| Function | Structure | Backend | Python seconds | R workflow seconds | R with scores seconds | Python normalized Q | R normalized Q |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `soft` | standard | cpu | 0.3962 | 0.6401 | 0.6401 | 0.00445435894 | 0.00445435894 |
| `soft` | standard | metal | 0.3347 | 0.4914 | 0.4914 | 0.00464000243 | 0.0046400042 |
| `a_soft` | standard | cpu | 0.4373 | 0.6857 | 0.7346 | 0.00154116501 | 0.00154116501 |
| `a_soft` | standard | metal | 0.3595 | 0.5825 | 0.6109 | 0.0015655172 | 0.001565516 |
| `soft` | strict | cpu | 1.1853 | 1.6204 | 1.6204 | 0.00514940249 | 0.00514940249 |
| `soft` | strict | metal | 1.6202 | 3.2307 | 3.2307 | 0.00512873718 | 0.00512873914 |
| `a_soft` | strict | cpu | 1.2238 | 1.4403 | 1.4813 | 0.00212836048 | 0.00212836048 |
| `a_soft` | strict | metal | 1.6533 | 3.1965 | 3.2293 | 0.00213374548 | 0.00213374548 |

Metal is not uniformly faster than CPU: the strict workloads above take longer
on Metal in both implementations. Small workloads can favor R; consult all rows
before drawing a speed conclusion. The scored timing ratio compares the same
returned information, but includes a separate R scoring pass for augmentation.

## Files and verification

- [results.csv](results.csv): all 888 rows, raw/normalized Q, independent-score
  discrepancies, design equality, statuses and timing ratios.
- [timings.json](timings.json): every timing repetition and associated scores.
- [manifest.json](manifest.json): exact arguments and shared/supplied input matrices.
- [summary.json](summary.json): counts, tolerances and the analysis-script fingerprint.
- [0066 diagnostic](0066-diagnostic.json) and [0069 diagnostic](0069-diagnostic.json):
  full candidate traces and fixed-base augmentation checks.
- Environment JSON files and `reference-functions.txt`: provenance.

The original run is retained locally at `.work/experiments/mac-full-02/`, with
raw designs and R progress logs. The execution-script hash and later analysis
hash are kept separately: report-only integrity/status columns were finalized
after the timed run; all recorded outputs were checked with that final analyzer.
The earlier `.work/experiments/mac-full-01/` run is diagnostic evidence only:
its fractional shared ranks exposed an R-index truncation issue in the harness.
The corrected manifest uses exact integer ranks, with dedicated transport tests.

The ordinary suite, including live R and 37 experiment-runner checks, passed
**342 tests with 34 CUDA-only skips**. This green suite does not erase the three
flagged comparisons in the broader experiment matrix.
