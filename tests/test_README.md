# softDOE test settings, results, and reproduction

**Update, 2026-10-22:** the same package inputs were retested on Apple M4 macOS
and Google Colab with an NVIDIA RTX PRO 6000 Blackwell GPU. macOS passed
**353 tests with 34 CUDA-only skips**; Colab passed **339 tests with 48
Mac/live-R skips**, including **all 34 actual-device CUDA tests without skips**.
Fresh wheel and source installs on Python 3.9 and 3.14 each passed **334 tests
with 53 optional skips**. See the [current verification record](../benchmarks/verification-2026-10-22/README.md)
for source hashes, per-test coverage, environments, and additional workflow checks.
The dated October 17 results and limitations below are retained as historical evidence.

Recorded on **2026-10-17** for **softDOE 0.2.0**, using **R MOFAT 2.0** as
the reference. Run all commands below from the **repository root**, not `tests/`.

The corrected macOS suite passed **353 tests with 34 CUDA-only skips**. The full
CPU/Metal experiment now passes **888/888 comparisons**, including all validation
checks and the four near-tie regressions. Python 3.9 also passes its complete
**444-case CPU matrix**. Fresh wheel and source installations on both Python
versions each passed **334 tests with 53 optional skips**.
The historical run below retains its original three failures for comparison.

Windows CPU, Linux CUDA, WSL2, and the optional stress profile have **not been
executed for this revision**. Their instructions below are reproduction procedures,
not claims of platform verification.

Quick links: [test files](#1-test-files-and-responsibilities),
[settings](#2-recorded-environment-and-settings), [results](#3-verified-results),
[Windows CPU](#5-windows-cpu-powershell-instructions),
[Linux CUDA](#6-linux-cuda-shell-instructions),
[potential issues](#8-potential-issues-and-maintenance-notes).

## 1. Test files and responsibilities

### Regression and implementation tests

| File | Settings or behavior checked |
| --- | --- |
| [test_soft.py](test_soft.py) | Standard/strict design dimensions, bounds, OFAT blocks, seeded repeatability, invalid generation arguments. |
| [test_public_api.py](test_public_api.py) | Public return contracts, PyDOE initialization, supplied-design overrides, structure fallback, QMC ordering, RNG independence, log-space scores, incremental scoring. |
| [test_measure.py](test_measure.py) | Hand-calculated `mustar` and `t`; invalid shapes, response lengths, and nonreal/nonnumeric responses. |
| [test_user_workflows.py](test_user_workflows.py) | Factor/response units, factor labels and paired row permutations, external odd-size screening blocks, immutable inputs, preserved screening rows after augmentation, parallel seeded CPU calls, minimum pool, invalid OFAT contrasts. |
| [test_release_regressions.py](test_release_regressions.py) | Extreme finite responses/steps, tiny contrasts, undefined total effects for constant responses, Q overflow/underflow, GPU numerical fallback, legacy unseeded RNG. |
| [test_q.py](test_q.py) | R fixtures and independent kernel solves, general kernel parameters, final augmented-design scores, underflow normalization, validation and fallback. |
| [test_mofat_parity.py](test_mofat_parity.py) | Saved MOFAT 2.0 results for QMC, shared-base `soft`/`a_soft`, supplied augmentation, documentation examples, and Friedman screening effects. No R installation needed. |
| [test_mofat_live.py](test_mofat_live.py) | 12 live CPU regression items against installed MOFAT 2.0; requires `--run-r`. Includes eight generated `a_soft` comparisons and one supplied-design comparison. |
| [test_pydoe_compat.py](test_pydoe_compat.py) | Current/legacy PyDOE import spelling and initialization, dependency error handling, restoration of global RNG state. |
| [test_backends.py](test_backends.py) | Capability reporting, dispatch, explicit CPU isolation, GPU failure recovery, actual available devices, and tensor paths on the host. |
| [test_cpu_q_parity.py](test_cpu_q_parity.py) | Four macOS final-Q regressions (0066, 0069, 0427, 0118), four analytical normal/subnormal/underflow cases, and three independent solves around the recursive Cholesky boundary (63/64/65 rows). Only the four macOS trajectory fixtures skip on other platforms. |
| [test_native_cpu.py](test_native_cpu.py) | Native Q/append operations against SciPy; native search and augmentation against reference implementations; duplicate rows and underflow. |
| [test_native_metal.py](test_native_metal.py) | Native Metal without PyTorch, MOFAT fixture agreement, augmentation choices/prefix preservation, and finite underflow results. |
| [test_strict_metal.py](test_strict_metal.py) | Fused strict-search updates, log scaling, `p=20,l=6` reference cases, and kernel failure recovery. Some paths require PyTorch. |
| [test_native_cuda.py](test_native_cuda.py) | Compiled strict CUDA search against tensor search, public dispatch, underflow normalization, and invalid-kernel recovery. Requires Linux CUDA. |
| [test_cuda_batching.py](test_cuda_batching.py) | Chunked correlations, paired score/state updates, search and augmentation against unbatched state. Requires CUDA. |
| [test_gpu_ties.py](test_gpu_ties.py) | GPU incumbent/left-choice rules near ties, common CUDA/Metal tolerances, and standard Metal reference agreement. |
| [test_experiment_runner.py](test_experiment_runner.py) | 37 checks of report integrity: incomplete/duplicate records, missing backends, QMC size, timing samples, underflow, integer-rank transport, strict failures, and case coverage. |

Hardware-dependent tests may skip on unavailable devices. Use `pytest -ra` to
see why; skips are not successful GPU executions. Many fallback tests deliberately
simulate unavailable hardware and do not establish real-device correctness.

### Reference data, experiments, and release checks

| File | Purpose |
| --- | --- |
| [conftest.py](conftest.py), [reference_mofat.R](reference_mofat.R) | Live-test options, exact MOFAT version guard, and isolated R reference generation. |
| [fixtures/mofat_reference.json](fixtures/mofat_reference.json), [fixtures/mofat_q.json](fixtures/mofat_q.json) | Saved CPU generation, screening, QMC, and Q results. |
| [fixtures/mofat_cpu_near_ties.json](fixtures/mofat_cpu_near_ties.json) | Unmodified MOFAT 2.0/macOS Accelerate results for the four near-tie regression cases; includes shared ranks and reference provenance. |
| [fixtures/mofat_standard_metal.json](fixtures/mofat_standard_metal.json), [fixtures/mofat_strict_metal.json](fixtures/mofat_strict_metal.json) | Saved Metal reference cases. |
| [experiment_cases.py](experiment_cases.py) | Deterministic smoke/full/stress matrices and invalid-input cases. |
| [run_mofat_experiments.py](run_mofat_experiments.py) | Portable experiment CLI, repeated timings, validation, independent rescoring, and reports. |
| [reference_experiments.R](reference_experiments.R) | MOFAT 2.0 workflows and independent R CPU Q evaluation in separate processes. |
| [diagnose_mofat_case.py](diagnose_mofat_case.py), [diagnose_mofat_case.R](diagnose_mofat_case.R) | Standard CPU search traces and augmentation replay for a flagged shared-base case. |
| [../tools/verify_release.py](../tools/verify_release.py) | Fresh wheel/sdist builds, isolated installations, metadata/assets/dependencies, pytest, README Python examples, and the example script. |
| [../tools/generate_mofat_fixtures.R](../tools/generate_mofat_fixtures.R), [../tools/generate_q_fixtures.R](../tools/generate_q_fixtures.R) | Reproducible R fixture generation. Existing fixtures remain the saved baseline. |
| [../.github/workflows/tests.yml](../.github/workflows/tests.yml) | Configured Ubuntu/macOS/Windows × Python 3.9/3.14 release checks. Remote execution is not established by this local record. |

## 2. Recorded environment and settings

| Setting | Recorded value |
| --- | --- |
| Machine / OS | Apple M4, arm64, macOS 26.5.2 |
| Python / softDOE | Python 3.14.7 and 3.9.6 / softDOE 0.2.0 |
| CPU linear algebra | System Accelerate on macOS; fixed recursive Cholesky panels for matrices up to 64 rows |
| Python numerical packages | Python 3.14: NumPy 2.5.4, SciPy 1.18.1, PyDOE 1.5.0; fresh Python 3.9: NumPy 2.0.2, SciPy 1.13.1, PyDOE 0.9.3 |
| R / reference | R 4.5.2 / MOFAT 2.0 |
| Selected R dependencies | spacefillr 0.4.0, SFDesign 0.1.5, Rcpp 1.1.1 |
| Experiment devices | Python CPU vs R CPU; Python native Metal vs R Metal |
| Timing | One untimed warmup, then three timed repetitions per valid workload in each implementation; report medians |
| Threads / scheduling | BLAS/OpenMP environment limits set to one; implementation stages run sequentially |
| Coordinates | Absolute agreement tolerance `2e-14` for transported R coordinates |
| Experiment scores | Relative tolerance `1e-8` on CPU/CUDA, `3e-3` on Metal; no absolute Q tolerance |

Full versions, device checks, import paths, source/script hashes, and R session
details are in the corrected-run [Python environment](../benchmarks/mofat2-cpu-fix-2026-10-17/python-environment.json)
and [R environment](../benchmarks/mofat2-cpu-fix-2026-10-17/r-environment.json).

The 12-item live regression suite uses `(p,l) = (2,4), (3,2), (4,2), (5,4)`,
both structures, `m=32`, and generated augmentation `n_aug=3,N=64`. R selects
each shared base with seed `100+p+l`. Additional items cover QMC, Q, supplied
augmentation, and Friedman screening on `MOFAT::mofat(10,4)` with seed 1.
Live score tolerance is `2e-12`; design tolerance is `2e-14`. These small
regression checks have different tolerances and scope from the broad experiments.

### Experiment profiles

Counts below are **configurations per backend**, not assertions or repetitions.

| Profile | Total | Valid `soft` | Valid `a_soft` | Invalid inputs | Recorded execution |
| --- | ---: | ---: | ---: | ---: | --- |
| `smoke` | 32 | 6 | 11 | 15 | CPU + Metal; one timed repetition |
| `full` | 444 | 105 | 324 | 15 | CPU + Metal; three timed repetitions |
| `stress` | 460 | 113 | 332 | 15 | Not run |

The full matrix includes:

- Core grid: `p={2,3,5,10,11,20}`, `l={2,4,6}`, `standard`/`strict`, seeds
  `{1,42}`, `m=64`; generated augmentation `n_aug={1,4,12}`, `N=128`.
- 73 supplied-design configurations, including OFAT inputs, boundary values
  0/1, duplicate rows, and supplied-input overrides of generation arguments.
- Minimum sizes and exhausted candidate pools (`n_aug=N=1` or `8`), initializer
  branches around `4!` (`p=23,24,25`), and additional `l=8` cases.
- Additional `m={1,17,500}`, `N={17,33,257,10000}`, and
  `n_aug={3,5,10,24}` settings; these are targeted cases, not a full cross-product.
- Default-sized workloads at `p={5,20}`, `l={4,6}`, both structures, `m=500`,
  `n_aug=10,N=10000`, using both shared and native initializers.
- 15 invalid-input configurations, including bad dimensions/counts, odd `l`,
  `n_aug>N`, malformed rows, and out-of-bounds supplied coordinates.

The stress extension adds `p={50,100,256,1024}`, `l=2`, both structures, `m=32`,
and augmentation `n_aug=16,N=512`. Large dense designs can require substantial
memory and time. No finite matrix exhausts all valid inputs or hardware.

For the full CPU/Metal run, 444 configurations × 2 backends gives 888 comparisons:
858 valid workflows and 30 invalid-input checks. Across both implementations,
the valid workflows generated **5,148 timed `soft`/`a_soft` calls and 1,716
warmups**. Separate R rescoring calls are additional.

List every case without importing softDOE or starting R:

```bash
python tests/run_mofat_experiments.py --profile full --list
```

### What a successful comparison establishes

Valid cases check dimensions, bounds, generated OFAT blocks, preserved supplied
rows, augmentation candidate membership/uniqueness, repeatability, actual backend
use, finite scores, and QMC evaluation counts. Python and R preserve their supplied
prefixes exactly before serialization. Every Python design is independently
rescored by R CPU Q; each R design is also rescored outside the timed workflow.

Shared-initializer cases use the same **integer rank matrix** in both processes.
The temporary initializer replacement does not edit either installed package.
Native-initializer cases use PyDOE and SFDesign normally; equal seeds do not imply
equal starting designs. Their invariants and scores are validated, but equal final
designs/scores are not required. Different-backend comparisons likewise do not
require final-score equality.

## 3. Verified results

The corrected-source audit is [mofat2-cpu-fix-2026-10-17](../benchmarks/mofat2-cpu-fix-2026-10-17/README.md).
| Corrected-source run | Result | Evidence |
| --- | --- | --- |
| Python 3.14.7, PyTorch, actual Metal, live MOFAT 2.0 | **353 passed, 34 CUDA-only skips**; 13.37 s | [test log](../benchmarks/mofat2-cpu-fix-2026-10-17/pytest-py314.log) |
| Full CPU/Metal matrix, three timing repetitions | **888 passed, 0 failed**; all 888 validation checks passed | [summary](../benchmarks/mofat2-cpu-fix-2026-10-17/summary.json), [888 rows](../benchmarks/mofat2-cpu-fix-2026-10-17/results.csv) |
| Python 3.9.6, full CPU matrix, one timing repetition | **444 passed, 0 failed** | [summary](../benchmarks/mofat2-cpu-fix-2026-10-17/py39-cpu-summary.json) |

Among 826 shared-input comparisons, 825 designs agree and one has an equivalent
score with a different design. Another 32 use native initializers, and 30 are
expected invalid-input rejections. Maximum same-design Python/R CPU Q relative
error is `1.422e-14`; maximum returned Metal Q error against R CPU is `2.513e-5`.
Fresh wheel and source installations on **Python 3.9.6 and 3.14.7** each passed
**334 tests with 53 optional skips**, including live MOFAT 2.0 and actual Metal.
Strict metadata, dependencies, package assets, examples, and README blocks passed.
All **29 release input hashes** match the checkout and each source archive.
The [distribution summary](../benchmarks/mofat2-cpu-fix-2026-10-17/release-verification.json)
records all four results. Local complete reports: `dist/release-checks/run-cr09hhm8/`
(Python 3.9) and `dist/release-checks/run-wjpahnno/` (Python 3.14).
Verified artifacts are consolidated in `dist/release-cpu-fix-2026-10-17/`.
No package was published.

### Historical results before the correction

These are saved results from 2026-10-17. Paths beginning `.work/` or `dist/` are
local artifacts and may not exist in a fresh clone. The benchmark folder contains
the durable experiment record.

| Run | Result | Evidence |
| --- | --- | --- |
| Ordinary suite with optional PyTorch, real Metal, and live R | **342 passed, 34 CUDA-only skipped**; 13.12 s | Local `.work/experiments/final-pytest.log` |
| Final CPU/Metal smoke | **64 passed, 0 failed**; one timed repetition | Local `.work/experiments/final-smoke/summary.json` |
| Full CPU/Metal matrix | **888 validation passes; 885 overall passes, 3 final-score agreement failures**; nonzero exit status | [summary.json](../benchmarks/mofat2-mac-2026-10-17/summary.json), [all 888 rows](../benchmarks/mofat2-mac-2026-10-17/results.csv) |
| Fresh wheel installation, Python 3.14.7 | **323 passed, 53 optional skips**; 19.31 s; live R enabled, PyTorch absent | Local `dist/release-checks/run-gqg822k9/wheel-tests.log` |
| Fresh sdist installation, Python 3.14.7 | **323 passed, 53 optional skips**; 19.19 s; live R enabled, PyTorch absent | Local `dist/release-checks/run-gqg822k9/sdist-tests.log` |
| Release build/metadata/dependencies/examples | Passed for both fresh installations | Local `dist/release-checks/run-gqg822k9/report.json` and associated logs |

Of 826 shared-input, same-backend valid comparisons, 822 produced matching
designs within `2e-14`, one produced different designs with equivalent scores,
and three failed final-score agreement. Another 32 valid comparisons used native
initializers. All 30 invalid-input comparisons rejected the inputs as expected.

Maximum relative error between Python CPU Q and independent R CPU Q for the
**same design** was `1.435e-14`. The maximum for returned Python Metal Q versus
R CPU rescoring was `2.513e-5`. These observed errors are distinct from the
larger final-design quality differences below.

### Original three final-score disagreements (reproduced and corrected)

All are CPU cases with `p=3,l=4,m=64,N=128` and standard structure; the last
case selects standard through invalid-structure fallback.

| Case ID | Seed | `n_aug` | Python Q | R Q | Normalized-Q difference relative to R |
| --- | ---: | ---: | ---: | ---: | ---: |
| `0066_a_soft_grid` | 1 | 4 | 0.0003705904611 | 0.0004544428438 | -6.573% |
| `0069_a_soft_grid` | 42 | 1 | 0.002000141240 | 0.001621314557 | +7.250% |
| `0427_a_soft_structure_fallback` | 1 | 4 | 0.0003705904611 | 0.0004544428438 | -6.573% |

Case 0427 repeats the search path in 0066. The two distinct paths first diverge
on a one-unit floating-point difference in a strict `>` accept/reject decision:
Python log-Q change `+8.88e-16` versus R `0` for 0066, and Python `+8.88e-16`
versus R `-8.88e-16` for 0069. Candidate designs are identical at that point.

Native and step-by-step optimizers agree within each implementation. When both
packages augment the same selected base, results agree within `1.11e-16` in
coordinates and `1.03e-15` relative Q. The evidence therefore locates these
differences in initial-search sensitivity to rounding, rather than the Q formula
or augmentation rule. The strict comparison and acceptance tolerances were kept;
the old records retain their failures. The corrected source passes all three
without changing those gates. See the historical [0066](../benchmarks/mofat2-mac-2026-10-17/0066-diagnostic.json)
and [0069](../benchmarks/mofat2-mac-2026-10-17/0069-diagnostic.json) diagnostics.

### Cause and regression coverage

Both languages received identical initial QMC points and shared ranks. Python's
native scorer added a log/exp round trip when scaling normal kernel products;
it also used a different sum-of-squares reduction order. Direct scaling and
paired accumulation remove those avoidable differences while retaining a full
log-space fallback for underflowed columns.

The first complete rerun exposed a fourth case: `0118_a_soft_grid`, CPU,
`p=5,l=4,m=64,N=128,n_aug=4,seed=42`, standard structure. Small Cholesky panels
and BLAS implementations rounded differently. Fixed recursive panels and system
Accelerate on macOS make this path consistent across the tested Python/SciPy
versions. All four cases now match the unchanged R final Q at `rtol=1e-8`.
The CPU strict `>` rule, GPU tie policy, and experiment tolerances are unchanged.

An independent 80-digit calculation finds the first differing candidates in
0066 and 0069 tied to working precision; 0118's difference is about `-8.48e-16`
in log Q. This supports a rounding-sensitive search diagnosis, not different
QMC samples or an incorrect Q formula. Different numerical libraries can still
choose different paths; the four stored trajectory regressions explicitly target
macOS. The seven analytical/solve regressions run on all platforms.

### Historical runtime findings

The [benchmark report](../benchmarks/mofat2-mac-2026-10-17/README.md) provides
eight representative rows; [timings.json](../benchmarks/mofat2-mac-2026-10-17/timings.json)
retains every timed repetition. For example, shared-initializer standard
`a_soft(p=20,l=6,m=500,n_aug=10,N=10000,seed=1)` on CPU took median **0.4373 s**
in Python, **0.6857 s** for R's workflow, and **0.7346 s** for R with scores.

MOFAT 2.0 `a_soft` returns the design without Q; Python returns both. Thus
`r_workflow_seconds` measures R design construction, while `r_scored_seconds`
adds an explicit Q evaluation. `python_seconds` includes its returned scores.
The scored ratio compares equivalent output information but includes an extra
R pass; it does not isolate programming-language overhead.

Warmups, process startup, JSON transport, output checks, and independent R CPU
rescoring are excluded from the timed samples. Shared-base generation is excluded;
native initialization is included in native cases. CUDA timing synchronizes the
Python device before/after each call. Metal is not uniformly faster than CPU,
and small workloads can favor R. Three local repetitions are descriptive evidence,
not a general speed guarantee.

## 4. Run the ordinary and live suites

Use a repository checkout: the installed distribution does not include this test
directory. Use an isolated environment and install the current checkout:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
python -c "import softDOE; print(softDOE.__version__); print(softDOE.__file__); print(softDOE.backend_info())"
python -m pytest -q -ra
```

The ordinary suite uses saved fixtures and needs neither R nor rpy2. For live
comparisons, first install **MOFAT 2.0 from its reference source/archive** and
dependencies. An older MOFAT release is not the baseline. In R:

```r
install.packages(c("jsonlite", "gtools", "Rcpp", "RcppArmadillo", "SFDesign",
                   "scales", "SLHD", "spacefillr", "MaxPro", "RcppHungarian", "withr"))
# Replace this path with the actual MOFAT 2.0 source archive, if not installed:
install.packages("/path/to/MOFAT_2.0.tar.gz", repos = NULL, type = "source")
stopifnot(as.character(packageVersion("MOFAT")) == "2.0")
MOFAT::mofat_backend_info()
```

Then, with `Rscript` on PATH:

```bash
python -m pytest -q -ra tests/test_mofat_live.py --run-r --mofat-version=2.0
python -m pytest -q -ra --run-r --mofat-version=2.0
```

The first command runs the 12 live items; the second runs the whole suite including
them. Missing R, missing dependencies, or the wrong MOFAT version fail an explicitly
requested live run. The tests use separate R processes and avoid the rpy2 dynamic
library import problem.

For isolated release verification, which builds and installs both artifacts:

```bash
python -m pip install build twine
python tools/verify_release.py --run-r --mofat-version=2.0
```

Omit `--run-r` if R is unavailable. This release script creates fresh environments
without optional PyTorch; its skip count can differ from the development environment.

## 5. Windows CPU: PowerShell instructions

Prerequisites: Python 3.9 or later, Visual Studio Build Tools with C++ support for
the Python extensions, and R plus matching Rtools for MOFAT's source installation.
Install the R dependencies/reference described above. Native Windows softDOE
supports CPU only; PyTorch is unnecessary for this run.

From the repository root, create a new environment:

```powershell
py -3 -m venv .venv-tests
.\.venv-tests\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
python -c "import softDOE; print(softDOE.__version__); print(softDOE.__file__); print(softDOE.backend_info())"
python -m pytest -q -ra
```

If PowerShell prevents activation, use `.\.venv-tests\Scripts\python.exe` in
place of `python` in subsequent commands. Verify R is reachable:

```powershell
Rscript -e "stopifnot(as.character(packageVersion('MOFAT')) == '2.0')"
python -m pytest -q -ra tests/test_mofat_live.py --run-r --mofat-version=2.0
python tests/run_mofat_experiments.py --profile smoke --backends cpu --repeats 1 --output experiment-results/windows-smoke
$LASTEXITCODE
python tests/run_mofat_experiments.py --profile full --backends cpu --repeats 3 --output experiment-results/windows-full
$LASTEXITCODE
```

If needed, add R's actual `bin` directory to PATH for live pytest, or append
`--rscript "C:/Program Files/R/R-4.x.y/bin/Rscript.exe"` to the **experiment**
commands, replacing the version/path. The pytest live option has no `--rscript`
argument; it requires PATH configuration.

Inspect the outcome even when the process exits nonzero:

```powershell
Get-Content experiment-results/windows-full/summary.json
Import-Csv experiment-results/windows-full/results.csv | Where-Object { $_.status -ne "passed" }
```

## 6. Linux CUDA: shell instructions

Prerequisites: an NVIDIA GPU/working driver, a C++17 compiler and Python build
headers, and a Python version supported by your CUDA-enabled PyTorch build.
softDOE's optional GPU dependency declares `torch>=2.8`. For an R CUDA comparison,
MOFAT also needs CUDA Toolkit 12.x/13.x and an NVCC-compatible GCC/Clang compiler,
as described in the reference archive's `inst/GPU_INSTALLATION.md`.

Create the Python environment from the repository root:

```bash
python3 -m venv .venv-tests
source .venv-tests/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

Install CUDA-enabled PyTorch 2.8 or newer into this environment using the
[official installation selector](https://pytorch.org/get-started/locally/): select
Linux, Pip, Python, and the CUDA build suitable for your driver. Use
`python -m pip` in place of `pip`/`pip3` in its command to target this environment.
A CPU-only PyTorch build cannot run these CUDA tests.

Check both the package location and a working CUDA backend:

```bash
nvidia-smi
python -c "import torch, softDOE; print(torch.__version__, torch.version.cuda); print(softDOE.__version__, softDOE.__file__); print(softDOE.backend_info()); assert torch.cuda.is_available(); assert softDOE.backend_info()['cuda']['available']"
```

Install the R dependencies listed above. For **R CUDA**, put `nvcc` on PATH or set
`CUDA_HOME` to the toolkit directory, then build the actual reference archive:

```bash
nvcc --version
R CMD INSTALL /path/to/MOFAT_2.0.tar.gz
Rscript -e 'stopifnot(as.character(packageVersion("MOFAT")) == "2.0"); print(MOFAT::mofat_backend_info(refresh=TRUE)); stopifnot(MOFAT::mofat_backend_info()$cuda$available)'
```

Replace the archive path. MOFAT's GPU guide describes `NVCC_CCBIN` and
`MOFAT_CUDA_ARCH` if compiler/architecture overrides are necessary. Python uses
CUDA-enabled PyTorch and its NVRTC runtime; the separate toolkit requirement
here is for building the R reference backend.

Once both CUDA availability checks succeed:

```bash
python -m pytest -q -ra --run-r --mofat-version=2.0
python tests/run_mofat_experiments.py --profile smoke --backends cpu cuda --repeats 1 --output experiment-results/cuda-smoke
echo $?
python tests/run_mofat_experiments.py --profile full --backends cpu cuda --repeats 3 --output experiment-results/cuda-full
echo $?
```

Review any CUDA skips in pytest: they do not count as CUDA verification. The
experiment runner requires the requested device and rejects unexpected CPU
fallback. Its default `--r-backend same` pairs Python CPU with R CPU and Python
CUDA with R CUDA.

If only R CPU is available, explicitly use:

```bash
python tests/run_mofat_experiments.py --profile full --backends cpu cuda --r-backend cpu --repeats 3 --output experiment-results/cuda-vs-r-cpu
```

This still requires real Python CUDA, but the CUDA timing ratio compares different
devices. Do not describe it as a same-backend Python/R speedup. WSL2 execution is
unverified here; it qualifies only if the Linux CUDA backend actually works.

## 7. macOS reproduction, diagnostics, and output interpretation

For the recorded CPU/Metal configuration, after installing both packages:

```bash
python tests/run_mofat_experiments.py --profile full --backends cpu metal --repeats 3 --output experiment-results/mac
python tests/diagnose_mofat_case.py experiment-results/mac --case 0066_a_soft_grid --output experiment-results/0066-diagnostic.json
python tests/diagnose_mofat_case.py experiment-results/mac --case 0069_a_soft_grid --output experiment-results/0069-diagnostic.json
```

Native Metal needs no PyTorch. Both packages must have access to a Metal device;
headless processes can differ. Use `--backends cpu` on a Mac without Metal.
Diagnostics support shared-base, standard CPU `a_soft` cases and leave report
statuses unchanged.

For every platform:

- Use a **new output directory** each time, or omit `--output` for a timestamped
  directory. Existing experiment directories and diagnostic files are not overwritten.
- `--case <id>` selects a case and can be repeated. Keep its original profile
  because IDs depend on the profile; exact arguments are in `manifest.json`.
- Change `--threads` if needed and record it. Default R-stage timeout is 14,400 s;
  `--r-timeout` changes it. Do not use `python -O` or `PYTHONOPTIMIZE`.
- `--profile stress` opts into the larger matrix. It has not been run for this
  record and may exceed available resources.
- Check exit status, `summary.json`, and individual failed rows. A missing summary,
  partial run, unavailable backend, or version failure is not a successful experiment.

| Output | Contents |
| --- | --- |
| `manifest.json` | Exact arguments, seeds, backend selections, and shared/supplied matrices. |
| `python-environment.json`, `r-environment.json` | Versions, devices, imported paths, thread settings, and source provenance. |
| `python-results.jsonl`, `r-results.jsonl` | Raw returned designs/scores, every timing repetition, validation errors. |
| `r-python-scores.jsonl` | Independent R CPU scoring of Python designs. |
| `results.csv` | One row per case/backend, scores, timing medians/ratios, design agreement, and statuses. |
| `summary.json` | Counts, tolerances, overall result, and analyzer hash. |
| `r-baseline.log`, `r-score.log` | R execution diagnostics. |

`validation_status` describes workflow invariants and returned-score correctness.
`agreement_status` describes required cross-implementation agreement; native
initializers/different-backend runs do not require equal final scores. Overall
`status` remains failed when a required final-Q comparison fails, even if validation
passes. Investigate new failures rather than automatically treating them as the
historical near-tie cases.

## 8. Potential issues and maintenance notes

- **Wrong Python installation:** an existing editable environment can point to an
  older checkout. Check `softDOE.__file__` and `softDOE.__version__` before testing.
- **Reference identity:** use MOFAT 2.0. The runner does not download a reference
  package or silently substitute another version.
- **Earlier harness issue:** `.work/experiments/mac-full-01/` used fractional
  shared ranks that R could truncate. The corrected `mac-full-02` uses exact
  integer ranks and is the canonical pre-correction run. Transport tests cover this;
  the earlier run must not be reported as current implementation results.
- **Numerical search sensitivity:** the original three failures and the additional
  0118 regression pass on the tested macOS configurations after correction. Other
  BLAS libraries can still change near-tie decisions; universal design parity is
  not established. The full experiment CLI retains strict failure reporting.
- **Screening inputs:** `measure` requires paired OFAT blocks. After augmentation,
  evaluate screening effects on the preserved original OFAT rows and responses;
  arbitrary added rows are not additional OFAT contrasts.
- **Timing interpretation:** retain workflow-only and scored R timings, backend
  identities, native/shared initialization, and all repetitions when comparing speed.
- **Coverage limits:** Windows, Intel Mac, WSL2, R CUDA optimizer parity, and the
  stress profile remain unverified. Linux/Colab CUDA execution is recorded in the
  October 22 audit. The configured CI matrix does not install live MOFAT by default or prove GPU
  coverage. Check actual CI artifacts before claiming a platform pass.

The current platform evidence is [benchmarks/verification-2026-10-22/](../benchmarks/verification-2026-10-22/).
The CPU correction evidence is [benchmarks/mofat2-cpu-fix-2026-10-17/](../benchmarks/mofat2-cpu-fix-2026-10-17/),
with full raw Mac designs/logs in `.work/q-fix/mac-accelerate-final/`.
The historical evidence is [benchmarks/mofat2-mac-2026-10-17/](../benchmarks/mofat2-mac-2026-10-17/),
including its `SHA256SUMS`. Full raw designs/logs remain locally in
`.work/experiments/mac-full-02/`. Execution and final analysis hashes are recorded
separately because report integrity/status columns were finalized after timing;
the final analyzer checked all saved outputs without changing timings.

For a new run, retain a new output directory and append its date, platform/device,
versions, exact command/profile, repeats/threads, pass/fail/skip counts, and evidence
path here. Keep unresolved failures and unexecuted platforms explicit. Additional
method details are in [docs/experiments.md](../docs/experiments.md).
