# SOFT and A-SOFT comparison experiments

**2026-10-22 CUDA update:** all 444 full-profile Python CUDA configurations passed
on Google Colab's RTX PRO 6000 Blackwell GPU. All 429 valid GPU-produced designs
also passed independent R MOFAT 2.0 CPU rescoring on the Mac. This checks GPU
execution and numerical results; R CUDA optimizer parity was not run. See the
[dated audit](../benchmarks/verification-2026-10-22/README.md) for exact scope and evidence.

The small live test suite has 12 pytest items, including eight generated
`a_soft` comparisons and one supplied-design comparison. It is a regression
suite, not an exhaustive parameter study. The experiment runner below adds a
specified coverage matrix, repeated timings, and independent R rescoring.
No finite matrix covers every valid integer, seed, floating-point input, or GPU.

## Files and coverage

The experiment files are in `tests/`:

- `experiment_cases.py`: deterministic case definitions, including invalid inputs.
- `run_mofat_experiments.py`: portable Python CLI, validation, timings and reports.
- `reference_experiments.R`: MOFAT 2.0 reference generation and independent rescoring.
- `test_experiment_runner.py`: tests of the experiment runner's failure detection.
- `diagnose_mofat_case.py` and `.R`: trace standard CPU search decisions and
  replay augmentation from both selected bases for a flagged case.

| Profile | Configurations | Valid soft | Valid a_soft | Invalid inputs |
| --- | ---: | ---: | ---: | ---: |
| `smoke` | 32 | 6 | 11 | 15 |
| `full` | 444 | 105 | 324 | 15 |
| `stress` | 460 | 113 | 332 | 15 |

Counts are per backend. `full --backends cpu metal` runs 888 paired comparisons,
including 858 valid workflows and 30 invalid-input checks. Each valid workflow
runs once to warm up and three more times by default **in each implementation**.
The counts describe configurations, not individual assertions or timed repetitions.

The full matrix includes:

- Core grid: `p = 2, 3, 5, 10, 11, 20`; `l = 2, 4, 6`; both `standard` and
  `strict`; seeds `1, 42`; `m=64`. Each configuration runs `soft` and generated
  `a_soft` with `n_aug = 1, 4, 12`, `N=128`.
- Supplied designs at `p = 2, 5, 11, 20`, `l = 2, 4, 6`, `n_aug = 1, 12`:
  OFAT designs, matrices touching 0 and 1, and matrices with duplicate rows.
  The last two are accepted augmentation inputs, not necessarily screening designs.
- The minimum `p=l=2`, `m=1`, and candidate exhaustion with `n_aug=N=1` or `8`.
- Below/equal/above `4!` factors (`p=23,24,25`, `l=4`) to exercise initializer
  branches; the `l=2` grid also exercises exact and remainder branches.
- `m=1,17,500`; non-power-of-two pools `N=17,33,257`; default-size pools
  `N=10000`; `l=8`; augmentation sizes `3,5,10,24` in additional cases.
- Default-sized workloads at `p=5,20`, `l=4,6`, both structures, `m=500`, and
  `n_aug=10,N=10000`, with both shared and native initializers.
- Invalid structure fallback, and supplied designs overriding invalid generation
  arguments; 15 explicit rejection cases for invalid dimensions, noninteger
  counts, odd `l`, zero counts, `n_aug>N`, malformed rows and out-of-bounds inputs.

`stress` adds `p=50,100,256,1024`, `l=2`, both structures, `m=32`, and generated
augmentation with `n_aug=16,N=512`. It is opt-in: the largest dense designs can
require substantial memory and time. These are additional scale tests, not a
promise that every machine can finish them. Ordinary `pytest` also covers
validation, numerical extremes, backend fallback and API properties.

List exact settings without importing softDOE or starting R:

```bash
python tests/run_mofat_experiments.py --profile full --list
```

## Prerequisites

Use a checkout containing these scripts. Install this Python source and verify
its import location; a pre-existing editable install may point to another checkout:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
python -c "import softDOE; print(softDOE.__version__); print(softDOE.__file__); print(softDOE.backend_info())"
```

Install **MOFAT 2.0** from the reference source/archive supplied by its authors,
including its R dependencies. An older CRAN package is not a substitute for
this baseline. The runner does not download or modify MOFAT. In R:

```r
install.packages(c("jsonlite", "gtools", "Rcpp", "RcppArmadillo", "SFDesign",
                   "scales", "SLHD", "spacefillr", "MaxPro", "RcppHungarian", "withr"))
# If MOFAT 2.0 is not already installed, use:
# install.packages("/path/to/MOFAT_2.0.tar.gz", repos=NULL, type="source")
stopifnot(as.character(packageVersion("MOFAT")) == "2.0")
MOFAT::mofat_backend_info()
```

`Rscript` must be on PATH. Otherwise pass its executable path with `--rscript`,
for example `--rscript "C:/Program Files/R/R-4.x.y/bin/Rscript.exe"` after replacing
the installed version. Building R packages from source requires the matching
R build tools (Rtools on Windows). No rpy2 is used.

Start with:

```bash
python tests/run_mofat_experiments.py --profile smoke --backends cpu --repeats 1 --output experiment-results/smoke
```

Each output directory must be new; use a different name for reruns, or omit
`--output` to create a timestamped directory. A run fails with a nonzero exit
status for a failed check, missing reference package, version mismatch, missing
requested GPU, or unexpected fallback. Check both the exit status and
`summary.json`; a partial run is not a successful run. Do not use `python -O`
or `PYTHONOPTIMIZE`; the runner rejects disabled assertions. The default R-stage
timeout is four hours; change `--r-timeout` for longer stress experiments.

## macOS

On an Apple GPU with both packages' Metal backends available:

```bash
python tests/run_mofat_experiments.py --profile full --backends cpu metal --repeats 3 --output experiment-results/mac
```

An Intel Mac without an available Metal backend can run `--backends cpu`.
The native Python Metal implementation needs no PyTorch.

## Windows and Linux CPU

The same command works in Windows PowerShell and a Linux shell:

```bash
python tests/run_mofat_experiments.py --profile full --backends cpu --repeats 3 --output experiment-results/cpu
python -m pytest -q
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1` and use
Visual Studio Build Tools with C++ support for the Python source build.
Native Windows softDOE supports CPU; installing CUDA PyTorch alone does not
change softDOE's platform dispatch.

## NVIDIA CUDA on Linux

Use a Python version supported by the CUDA-enabled PyTorch build, and install
the appropriate wheel from the [official PyTorch installation selector](https://pytorch.org/get-started/locally/).
Check availability before running:

```bash
python -c "import torch, softDOE; print(torch.__version__, torch.version.cuda); print(softDOE.backend_info())"
python tests/run_mofat_experiments.py --profile full --backends cpu cuda --repeats 3 --output experiment-results/cuda
python -m pytest -q
```

For a same-backend comparison, MOFAT must also be built with its CUDA backend
available. MOFAT's source declares a CUDA Toolkit 12.x/13.x and supported host
compiler requirement on Linux. Make `nvcc` available on PATH, or set `CUDA_HOME`
to the toolkit directory, then build the reference archive:

```bash
R CMD INSTALL MOFAT_2.0.tar.gz
Rscript -e 'stopifnot(MOFAT::mofat_backend_info(refresh=TRUE)$cuda$available)'
```

The archive's `inst/GPU_INSTALLATION.md` documents `NVCC_CCBIN` and
`MOFAT_CUDA_ARCH` overrides when needed.
Python's runtime uses CUDA-enabled PyTorch; its strict kernel uses NVRTC when
available. Both packages must report CUDA available, and every timed call must
actually use CUDA. Python CUDA timing synchronizes the device before and after
each call. R returns host results after its backend work completes.

If R only supports CPU on the machine, explicitly run:

```bash
python tests/run_mofat_experiments.py --profile full --backends cpu cuda --r-backend cpu --repeats 3 --output experiment-results/cuda-vs-r-cpu
```

This still requires real Python CUDA execution, but labels the reference as R
CPU. Its timing ratio compares different devices and must not be described as
a Python-versus-R implementation speedup on the same hardware backend.
WSL2 is usable only if softDOE detects a working Linux CUDA backend there;
this repository has not verified WSL2 execution.

For a larger opt-in run, replace `full` with `stress`. A single failing case can
be reproduced with `--case <id>` using an ID from the manifest; repeat `--case`
to select several cases. Keep the same profile because IDs depend on the profile.

## What is compared

Most generated cases supply the same initial integer rank matrix to both implementations.
PyDOE ranks are rounded to their intended integers before transport, avoiding
sub-ulp rank errors being truncated by R indexing.
The runner temporarily replaces their internal initializer in the current
Python/R process; it never edits the R package. These timings include public
function validation, evaluation/candidate construction, optimization and output,
but exclude generating the shared starting matrix. Native-initializer cases
use each package's actual initializer, reset its seed before every call, and
include that initialization cost. SFDesign and PyDOE differ: identical seeds
are not expected to yield identical designs or Q in those cases.

For every valid case the runner checks shape, bounds, OFAT blocks for generated
bases, preservation of supplied rows, membership and uniqueness of augmentation
points, deterministic repetitions, backend use, finite scores, and independent
Q calculations. R checks supplied-prefix preservation before JSON serialization;
Python checks it exactly too. Transported R coordinates allow `2e-14` absolute
error. Each Python output is rescored in a separate R process using R CPU `Q()`.
Each R output is also rescored on R CPU outside the timed calls. All Q comparisons
use the same evaluation set and kernel scale; raw and normalized Q are both kept.
The `soft` evaluation count is checked against the actual pool size.

The fixed relative tolerances for returned scores are `1e-8` on CPU/CUDA and
`3e-3` on Metal (float32). They apply even to very small positive Q; there is no
absolute tolerance that can hide underflow. Same-base, same-backend final scores
must also agree within these tolerances. Coordinate agreement is reported
separately: rounding in near-tie decisions can produce different designs with
equivalent scores. Native-initializer or different-backend comparisons report
quality differences without requiring equality. These tolerances are acceptance
limits, not statements of observed error; inspect the recorded differences.

CPU search uses MOFAT's strict `>` comparison, with no near-tie tolerance.
Different floating-point evaluation orders can move a candidate score by one
unit in the last place and change the selected path. Later augmentation can
amplify this into a material final-Q difference. The runner deliberately flags
such final-score disagreements and exits nonzero; it does not increase the
tolerance to make them pass. A flagged final-score comparison must be examined
separately from the independent correctness check on each returned design's Q.
The CPU diagnostic scripts can trace standard-structure cases and test whether
augmentation agrees when both implementations receive the same fixed base:

```bash
python tests/diagnose_mofat_case.py experiment-results/mac --case 0066_a_soft_grid --output experiment-results/0066-diagnostic.json
```

The diagnostic requires a valid shared-base standard CPU `a_soft` case and a
new output filename. It records the first different accept/reject decision,
both native/reference traces, and augmentation replay from each selected base.
It does not change the experiment's status or tolerances.

## Runtime interpretation and outputs

Every valid workload has an untimed warmup followed by three repetitions by
default. BLAS/OpenMP thread environment variables are set to one by default
before numerical libraries start; use `--threads` to change them. R and Python
stages run sequentially. Timings exclude process startup, JSON I/O, validation
of outputs and independent reference rescoring. The runner retains every
repetition and reports medians, not a fastest sample.

**R MOFAT 2.0 `a_soft` does not return Q.** Python does. Consequently:

- `r_workflow_seconds`: R design generation/augmentation only.
- `r_scored_seconds`: that R workflow plus an explicit R `Q()` call on the same
  backend and full candidate pool, returning the outputs Python supplies.
- `python_seconds`: Python public function, including its returned scores.
- Both R/Python time ratios are provided. The scored ratio compares the same
  deliverables but includes an extra R scoring pass; it does not isolate language overhead.

Small workloads can be dominated by dispatch and timing overhead. Ratios depend
on dimensions, initializer, backend, dependencies and machine load. Three local
repetitions are descriptive measurements, not a statistical speed guarantee.

Each run writes:

- `manifest.json`: every argument, seed and shared/supplied input matrix.
- `python-environment.json`, `r-environment.json`: versions, backend/device
  information, import paths, thread settings and source/function fingerprints.
- `python-results.jsonl`, `r-results.jsonl`: designs, scores, repetition timings,
  diagnostics and explicit errors.
- `r-python-scores.jsonl`: independent R scores for Python designs.
- `results.csv`, `summary.json`: one row per configuration/backend, pass/fail,
  Q differences, design equality and timing ratios.
- `r-baseline.log`, `r-score.log`: R diagnostics and progress.

Inspect failed rows before interpreting runtime. Results from Windows/Linux/CUDA
are only evidence after running on those environments; the scripts themselves
are not proof of those platforms' correctness.

## Historical Mac results before the CPU correction: 2026-10-17

The full profile ran on Apple M4 CPU and native Metal with three timed repeats:
**444 configurations × 2 backends = 888 comparisons**. All workflow/input and
independent-score checks passed; **885 comparisons passed and 3 final-Q
agreements failed**. Those three CPU cases remain explicit failures in the
report and produce exit status 1. They are two distinct `a_soft` search paths
plus one repeated invalid-structure fallback case, traced to one-unit rounding
differences in strict CPU accept/reject comparisons. Fixed-base augmentation
agrees in both directions. This is not a claim of universal R/Python design parity.

See the [Mac audit and runtime tables](../benchmarks/mofat2-mac-2026-10-17/README.md),
[all comparison rows](../benchmarks/mofat2-mac-2026-10-17/results.csv), and
[every timing repetition](../benchmarks/mofat2-mac-2026-10-17/timings.json).
The largest relative discrepancy when independently rescoring the same design
on Python/R CPU was `1.435e-14`; the largest Metal returned-Q discrepancy against
R CPU rescoring was `2.513e-5`. The ordinary suite, including live R and runner
integrity tests, passed **342 tests**, with **34 CUDA-only skips**.

The stress profile, Windows, Linux, WSL2 and CUDA hardware were not executed
in this audit. Their scripts and commands are provided for those environments.

## CPU numerical correction: 2026-10-17

The [correction audit](../benchmarks/mofat2-cpu-fix-2026-10-17/README.md)
reproduced the original three failures and an additional near-tie case discovered
during testing. Direct kernel scaling, paired squared-norm accumulation, fixed
small Cholesky panels, and a consistent macOS Accelerate backend resolve those
four cases on Python 3.9 and 3.14. QMC inputs were identical; the differences
came from score rounding and the subsequent greedy paths. Reference scores,
strict search decisions, and experiment tolerances were preserved.

The corrected CPU/Metal run passed **888/888 comparisons**, with all 888
validation checks passing and the original gates unchanged. The Python 3.9 full
CPU sweep also passed **444/444**. The full Python 3.14 suite passed **353 tests**
with **34 CUDA-only skips**. Exact settings, unchanged reference evidence, and
per-case results are in the correction audit. Fresh wheel/source installations
on Python 3.9.6 and 3.14.7 each passed 334 tests with 53 optional skips, plus
metadata, dependency, example, and README checks. Other platforms were unverified
in that October 17 record; the October 22 audit above adds Python CUDA workflows
and independent R CPU rescoring.
