# Verification

Latest executed verification: [2026-10-22 macOS and Colab CUDA audit](../benchmarks/verification-2026-10-22/README.md).
The current 387-item suite passes across the two applicable platforms: 353
passes on macOS and 339 on Colab, with all 34 actual-device CUDA items passing.
Platform skips are recorded separately. Fresh Python 3.9 and 3.14 wheel/source
installations also pass. Older dated audits below remain historical records.

Run development commands from the repository root in an activated virtual
environment. Tests and examples live in the Git repository; release archives
contain the runtime package, build inputs, README, and license notices.

## Python tests and examples

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
python -m pytest -q
python examples/mofat_examples.py
```

The tests cover all six public functions, standard/strict block structure,
independent Q calculations, input validation, numerical extremes, supplied-base
preservation, reproducibility, and backend fallback. They also compare against
saved R outputs without requiring R or rpy2. Check the import path when using
an existing environment:

```python
import softDOE
print(softDOE.__version__)
print(softDOE.__file__)
print(softDOE.backend_info())
```

An editable install should point to this checkout. A release test must instead
point inside its isolated environment. A green test run against another
checkout does not validate this source.

## Live R comparisons

The reference baseline is **MOFAT 2.0**, as declared by its R `DESCRIPTION` and
`packageVersion("MOFAT")`. This is distinct from Python softDOE 0.2.0. The public
CRAN MOFAT 1.0 package does not provide the SOFT APIs used by these tests.
Install the intended MOFAT reference archive in R and its dependencies, plus
`jsonlite`. Make `Rscript` available on PATH. First check in R:

```r
packageVersion("MOFAT")
stopifnot(as.character(packageVersion("MOFAT")) == "2.0")
stopifnot(requireNamespace("jsonlite", quietly = TRUE))
```

Then run in the Python checkout:

```bash
python -m pytest -q tests/test_mofat_live.py --run-r --mofat-version=2.0
```

`--run-r` starts a temporary R process and requires the **exact requested
version**; missing R, missing packages, and mismatched versions fail the run.
Without this flag, only the live tests skip. No rpy2 installation is needed.
The subprocess changes the initial base generator in memory to export a shared
base; it never changes the R package on disk.

The live suite checks:

- standard and strict `soft`, including both full-permutation and remainder
  branches, with identical initial bases in R and Python;
- `a_soft` with generated and supplied bases, and final scores independently
  recomputed using R's `Q` on the full augmentation candidate pool;
- `Q` at multiple kernel exponents;
- `qmc_generate` across corner policies and at the 1024-factor limit;
- `measure` on SOFT structures and an actual `MOFAT::mofat` design.

R's SFDesign and Python's PyDOE use different initializers; equal seeds do
not imply equal designs. Design comparisons use shared base matrices with
absolute tolerance 2e-14. CPU scores use relative tolerance 2e-12. Sobol
coordinates agree after the reference's float32 rounding. `backend_info()` is
covered by Python tests; availability depends on the host and is not expected
to match R's distinct GPU dependencies.

Historical fixtures in `tests/fixtures/` retain their version/source hashes.
The read-only regeneration scripts are for maintainers with the MOFAT source:

```bash
Rscript tools/generate_mofat_fixtures.R /path/to/MOFAT tests/fixtures/mofat_reference.json
Rscript tools/generate_q_fixtures.R /path/to/MOFAT tests/fixtures/mofat_q.json
```

Replace `/path/to/MOFAT` with that checkout. Run from the Python repository
root. Rcpp compiles a string copy in a temporary directory; the scripts reject
output inside the reference checkout. Do not regenerate references just to
make failing Python tests pass.

## Broad generator experiments

The 12 live test items are supplemented by a 444-configuration `soft`/`a_soft`
experiment matrix, with repeated timing, varied dimensions and augmentation
sizes, and independent R Q rescoring. See [experiments.md](experiments.md) for
coverage, macOS/Windows/Linux/CUDA commands, output fields and current results.
The runner is opt-in; ordinary pytest does not launch the benchmark matrix.

## Release checks

Install the build/check tools, then use the portable verification script:

```bash
python -m pip install --upgrade pip
python -m pip install build twine
python tools/verify_release.py
```

To include the live R suite in both artifact checks:

```bash
python tools/verify_release.py --run-r --mofat-version=2.0
```

Each invocation creates a new directory under `dist/release-checks/` and:

1. Builds an sdist and builds a wheel from that sdist using isolated build dependencies.
2. Runs `twine check --strict` and checks archive contents and runtime assets.
3. Creates separate virtual environments for wheel and sdist installations.
4. Copies tests/examples away from the source tree, verifies the actual imported
   path/version, runs `pip check`, the tests, the example script, and every Python
   block in README order.
5. Writes command logs, dependency/backend details, artifact SHA-256 hashes,
   file lists, and pass/fail status to `report.json`.

This requires internet access for Python dependencies and the platform's C++17
build tools. It does not install into your active environment, alter R, or
publish anything. A failing stage returns a nonzero exit status and retains
its logs. The output wheel is platform/Python specific; a macOS wheel cannot
be installed on Linux or Windows. The sdist includes all platform build inputs.

## GPU and platform checks

Metal tests require an accessible Apple GPU. CUDA tests require Linux, a
supported NVIDIA GPU/driver, and CUDA-enabled PyTorch. The isolated release
verifier installs base dependencies only; it does not install PyTorch. For CUDA
validation, prepare a separate GPU environment with CUDA-enabled PyTorch,
install `python -m pip install -e ".[test,gpu]"`, and run `python -m pytest -q`.
Tests skip unavailable hardware explicitly. Mocked failures and CPU tensor tests validate fallback or
arithmetic, not actual GPU execution. Confirm `backend_used` when testing GPU
workflows; a successful call can be a successful CPU fallback.

The CI workflow builds and tests source installs on Linux, macOS, and Windows
with Python 3.9 and 3.14. GPU availability is not assumed on hosted runners.
A workflow definition is not evidence that those jobs have run; release reports
must distinguish executed local/remote checks from configured checks.

Earlier hardware measurements and their original inputs are retained in
[performance.md](performance.md) and `benchmarks/`. They describe historical
artifacts, not a guarantee about a newly rebuilt artifact. For new benchmarks:

```bash
python -m pip install threadpoolctl
python tools/benchmark_workflows.py --backend cpu metal --repeats 3 --blas-threads 1 --output benchmarks/local
```

Use `--backend cpu cuda` on supported Linux hardware. Warm each exact workflow,
synchronize CUDA, record hardware/runtime versions, and report medians over
repeated calls. Do not count a fallback as GPU execution or assume CPU/GPU
near-tie choices are identical.

## Local release audit: 2026-10-17

The release inputs before the expanded experiment documentation and software-citation update were verified on an Apple M4 with native Metal:

| Installation | Python | Result |
| --- | --- | --- |
| Fresh wheel and sdist, base dependencies, live R enabled | 3.9.6 | 286 passed, 53 skipped per artifact |
| Fresh wheel and sdist, base dependencies, live R enabled | 3.14.7 | 286 passed, 53 skipped per artifact |
| Installed wheel plus optional PyTorch, live R enabled | 3.14.7 | 305 passed, 34 CUDA-only skips |
| Editable development install, ordinary tests | 3.9.6 | 274 passed, 65 skips (including 12 opt-in live R tests) |

Both artifact builds passed strict metadata checks, archive-content checks,
all README Python examples, and the example script. All 29 release input files
matched the tested source archives and then-current checkout byte-for-byte.
The subsequent README/CITATION changes supersede those archived documentation
hashes; the runtime sources are unchanged by the experiment harness. The optional
tensor run measured 91% Python statement coverage; this does not measure C++
or GPU-kernel coverage. Live comparisons and regenerated CPU fixtures matched
the confirmed MOFAT 2.0 reference baseline.

No Windows, Linux, or NVIDIA hardware run was performed in this audit. The CI
matrix is configured but has not been run remotely for these changes. Earlier
CUDA records remain historical evidence only. No package was uploaded.

Local artifacts and SHA-256 sums are in `dist/release-2026-10-17/`.
Detailed per-command reports are under `dist/release-checks/run-pgr3cwy0/`
and `dist/release-checks/run-lbf2r8y4/`; optional coverage and editable-install
logs are under `.work/release-audit/`. These generated directories are not
included in release archives.

## Expanded experiment verification: 2026-10-17

The subsequent generator study ran 444 configurations on CPU and Metal:
888 comparisons, 885 passed and three final-Q disagreements retained as failures.
All own-design score/invariant/input checks passed; the disagreements are traced
CPU greedy-search decisions differing by one floating-point unit. See the
[full audit](../benchmarks/mofat2-mac-2026-10-17/README.md) for exact cases,
independent rescoring, fixed-base augmentation diagnoses and runtime tables.
The matrix is deliberately not reported as universal design parity.

The ordinary suite with PyTorch, Metal and live R passed 342 tests with 34
CUDA-only skips. The 37 runner integrity/transport tests also passed on Python
3.9.6. The final CLI smoke run passed all 64 CPU/Metal checks. Python static
checks, R syntax checks, CFF schema validation and diff whitespace checks passed.

Fresh Python 3.14.7 wheel and sdist installs after the README/software-citation
update each passed 323 tests with 53 optional skips, plus strict metadata,
archive contents, dependencies, examples and README Python blocks. All 29
release input hashes matched that pre-correction revision. Its build/report is
`dist/release-checks/run-gqg822k9/`; previous consolidated artifacts are retained
as historical builds. No package was published.

## CPU correction and fresh release verification: 2026-10-17

The corrected native CPU arithmetic resolves the original three final-Q
comparisons and a fourth near-tie case discovered in the first rerun. The
[correction audit](../benchmarks/mofat2-cpu-fix-2026-10-17/README.md) records the
causes, fixed recursive Cholesky panels, macOS Accelerate choice, independent
80-digit calculations, and unchanged reference inputs/comparison tolerances.

| Check | Result |
| --- | --- |
| Full CPU/Metal matrix, three repeats | 888 passed, 0 failed; all 888 validations passed |
| Python 3.9 full CPU matrix, one repeat | 444 passed, 0 failed |
| Python 3.14 with PyTorch, real Metal, live R | 353 passed, 34 CUDA-only skips |
| Fresh Python 3.9.6 wheel and source installs | 334 passed, 53 optional skips per installation |
| Fresh Python 3.14.7 wheel and source installs | 334 passed, 53 optional skips per installation |

Both builds passed strict metadata, dependencies, required assets, isolated
imports, live MOFAT 2.0, actual Metal, examples, and README Python blocks.
All 29 release input hashes match the current checkout and both tested archives;
artifact hashes also match. Reports: `dist/release-checks/run-cr09hhm8/report.json`
(Python 3.9) and `dist/release-checks/run-wjpahnno/report.json` (Python 3.14).
Current artifacts and SHA256 sums are at `dist/release-cpu-fix-2026-10-17/`.
All older release directories above are historical and do not contain this fix.

Seven new analytical/kernel-solve tests are portable; four exact near-tie
trajectory tests explicitly target macOS Accelerate. The full experiment CLI
retains strict cross-implementation failure reporting on every platform.
At the time of this October 17 audit, Windows, Linux, CUDA, Intel macOS, and
stress execution were unverified. The October 22 audit above adds Linux/Colab
CUDA verification. No commit, push, or publication was performed.
