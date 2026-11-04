# rpy2 linkage repair

Installed rpy2 3.6.9 / rpy2-rinterface 3.6.7 in .venv (Python 3.9).
R --version reports 4.5.2, arm64; R RHOME reports /Library/Frameworks/R.framework/Resources.
otool -L confirms API extension arm64 links to absent R.framework/Versions/4.6 libraries.
Import reproduces warning and falls back to ABI mode.
An initial version-print probe used an invalid ListVector attribute; use robjects.r("R.version.string") instead.
Reference: https://github.com/rpy2/rpy2 identifies rpy2-rinterface as the R C API binding package.

## Completed repair

Built source distribution of rpy2-rinterface==3.6.7 with R_HOME=/Library/Frameworks/R.framework/Resources, RPY2_CFFI_MODE=BOTH, ARCHFLAGS='-arch arm64'.
Installed local wheel using pip --no-index --no-deps --force-reinstall.
Input/output: .venv/lib/python3.9/site-packages rpy2-rinterface distribution files.
Backup: /tmp/softdoe-rpy2-repair/rpy2-rinterface-before.tar.gz (excludes external cached bytecode).
Built wheel: /tmp/softdoe-rpy2-repair/rpy2_rinterface-3.6.7-cp39-cp39-macosx_11_0_universal2.whl.
Verification: forced API and default imports pass without fallback warnings; base and stats load; sum(1:5)=15; BLAS matrix-vector multiplication returns [7,10]; pip check passes; otool links to installed 4.5-arm64 libraries.
Unchanged: application source, dependency declarations, installed dependency versions, system R, existing user edits.

# measure symmetry investigation

Inputs: src/softDOE/_measure.py, tests/test_measure.py, tests/test_mofat_parity.py, installed MOFAT 2.0, user's Friedman function with mofat(10,4).
Default Python imports .venv/src/softdoe/src/softDOE; this older checkout has the same measure formulas but weaker input validation. Current source tested explicitly with PYTHONPATH=src.
Live parity for R seeds 1, 42, 123: Python and installed R agree to <=3.4e-16. Both imported and current repository implementations checked.
Seed 1: mustar first two = [7.99860699880763, 6.11264098119259]; t first two approximately [0.690007784,0.371023393].
Seed 42: mustar first two approximately [8.424460355,8.424460355]; t [0.519269092,0.519269092].
Seed 123: mustar first two approximately [7.544097813,7.544097813]; t [0.242359630,0.242359630].
There are 44 runs but only 4 OFAT contrasts per factor. Functional symmetry does not force finite, non-exchange-balanced contrasts to agree.
Swapping columns 0/1 swaps measures; appending the swapped design WITH corresponding OFAT-block reordering yields identical first two estimates, checked for all 3 seeds.
Seed 1 exchange-balanced estimates: mustar [7.05562399,7.05562399], t [0.536684374,0.536684374].
Verification: PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_measure.py tests/test_mofat_parity.py -k measure => 8 passed, 23 deselected.
Initial pytest without PYTHONPATH used older checkout: 5 passed, 3 input-validation failures. Not related to symmetry; formulas identical on valid input.
Reference definition: https://cran.r-universe.dev/MOFAT/doc/manual.html (public version 1.0; local installed version 2.0 source inspected directly).
No implementation defect found for supplied example. No source, tests, dependencies, or installed checkout modified; only task_plan.md and notes.md updated.

# Release-hardening goal evidence

- Current worktree 0.2.0, PyPI API on 2026-10-17 lists only0.1.0. Public clone is not a proven route to0.2.0; README now requires actual0.2.0 source archive or matching checkout.
- Verified https://arxiv.org/abs/2606.02533: title Space-Filling One-Factor-At-A-Time Designs, authors Wei-Yang Yu and V. Roshan Joseph, year2026, doi10.48550/arXiv.2606.02533. README citation and CITATION.cff use those facts.
- Local ../MOFAT/DESCRIPTION and installed package both2.0. User clarification requested because goal said0.2.0; do not silently equate versions.
- Added explicit optional live R suite, 12 tests passed against local installed2.0. Shared A_fun handles independent Python/R RNGs. Initial fixture JSON serialized NULL as{}, corrected with null="null".
- Independent user tester found release-source clone instructions unverified and measure response scaling instability; developer fixing numerical extremes and Q extreme kernels.
- Fresh release environments, sdist/wheel verification script and cross-platform CI added. CI configuration is not execution evidence.

## Final current-turn verification

Final runs: dist/release-checks/run-pgr3cwy0 (Python3.14.7), run-lbf2r8y4 (Python3.9.6), both status passed. Each wheel and sdist run:286 passed,53 skipped, README and examples pass, pip check and twine strict pass. Reports contain every command/path/hash. Current29 runtime/build/README/license inputs match archives byte-for-byte.
Optional torch2.14.1 installed into .work/release-venv only:305 passed,34 CUDA-only skips, Python statement coverage90.5% (rounded91%). Real Apple M4 Metal execution confirmed in package backend_info and device tests. No new CUDA/Linux/Windows hardware results claimed.
README development command required pip upgrade with CLTools Python3.9 bundled pip21.2.4; upgraded isolated editable-check pip26.0.1 then editable install succeeded. Ordinary tests274passed65skips (12optionalR +53optionalGPU); example succeeds. Existing user's .venv remains linked to old checkout intentionally; release validations use their own environments.
Source fixture regenerate output matches saved references numerically. Three R source hashes differ from historical fixture capture (soft.R,soft_augment.R,soft_measure_lib.R), but fresh live and source-regenerated outputs match. Existing fixture files were not overwritten.
Schema validation of CITATION.cff against official1.2.0 schema passes. Static ruff E9/F and git diff --check pass.
Deliverables: dist/release-2026-10-17 includes sdist, cp39 universal2 Mac wheel, cp314 arm64 Mac wheel, two reports and SHA256SUMS. No publication/push.
Remaining: ask user to resolve MOFAT0.2.0 vs local2.0. This is a baseline identity question, not permission to make routine changes.

## Reference identity follow-up
Read-only git history of ../MOFAT confirms version changes at7c501e1 (1.0), d63f205 (1.1),6b7dcd3 (2.0). The local MOFAT_2.0.tar.gz DESCRIPTION also says2.0. No local0.2.0 baseline found. Both final release reports and hashes revalidated unchanged. An overbroad parent-folder filename scan encountered OneDrive .Trash timeouts; it was interrupted (exit130), and search was narrowed to the MOFAT checkout. No package/R source was modified in this follow-up.

## Baseline confirmed and release audit complete (2026-10-17)
User explicitly selected MOFAT 2.0 as the reference. This resolves the prior version-identity blocker. Updated docs/verification.md, task_plan.md, notes.md and .work/release-audit/summary.md; runtime, tests, reference source and release artifacts were preserved.
Final command: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .work/release-venv/bin/python -m pytest -q tests/test_mofat_live.py --run-r --mofat-version=2.0. Result: 12 passed in 1.11s; log .work/release-audit/final-mofat-2.0.log. Import path confirmed inside .work/release-venv/lib/python3.14/site-packages/softDOE, version 0.2.0.
Rechecked both final release reports, all 29 source/archive input hashes per report, every built artifact hash, and delivered SHA256SUMS: all pass. Fresh-install results remain 286 passed/53 skipped for each wheel and sdist on Python 3.9.6 and 3.14.7; optional PyTorch/Metal result remains 305 passed/34 CUDA-only skips. No release input changed, so no rebuild was needed. Windows/Linux/CUDA execution remains unverified for this revision; no publication performed.

# Broad experiment matrix (2026-10-17)
Previous 12 live pytest items covered 8 shared-base soft configurations, 8 generated and 1 supplied a_soft configurations, plus Q/qmc/measure checks. They were not a runtime matrix.
New full matrix: 444 configurations, comprising 429 valid (105 soft, 324 a_soft, including 73 supplied-base) and 15 invalid-input checks, repeated per requested backend. Default CPU+Metal means 888 comparisons, one warmup and three timed calls for every valid implementation/backend configuration. Shared-base and native-initializer workloads are distinguished.
R MOFAT2.0 a_soft has no returned scores. Runner records R workflow-only time and workflow-plus-Q time separately; independently evaluates each Python design with R CPU Q after timing. User software citation and NSF acknowledgment updated by developer agent in README/CITATION only.
Harness errors found before full run: R reserved-word access s$function required s[["function"]]; JSON transport creates sub-ulp differences, so transported coordinate checks use 2e-14 while Python supplied preservation and internal R supplied preservation remain exact. CPU+Metal smoke3repeats:64 passed0failed at .work/experiments/smoke-02.
Full run started: .work/experiments/mac-full-01, console .work/experiments/mac-full-01-console.log. R/Python processes run sequentially to avoid concurrent timing contention.

## Full run 1 findings and correction
mac-full-01 completed:677passed211failed. All workflow and independent own-design Q checks passed; failures were shared-base final-score comparisons.208 mismatches used l=6 shared PyDOE ranks such as3.9999999999999996. Python's level transform rounds, whereas R's index semantics truncate. Corrected only experiment input construction to np.rint shared ranks, and R now rejects noninteger shared ranks. Four affected CPU/Metal diagnostic configurations pass (8comparisons) in rank-check.37 runner integrity/transport tests pass; CFF schema, lint and whitespace checks pass.
Three remaining CPU comparisons in mac-full-01 (0066,0069,0427) are two distinct p3l4a_soft trajectories plus invalid-structure fallback repeating0066. Independent developer traced identical candidate matrices with logQ differences at most1ULP:0066 Rdelta0 vs Python+8.88e-16;0069 Rdelta-8.88e-16 vs Python+8.88e-16. Both use strict CPU > per baseline, so legitimate optimization paths differ. Python augmentation from R's chosen base exactly reproduces R finaldesign/Q; R augmentation from Python's chosen base exactly reproduces Python finaldesign/Q (maxcoordinate difference1.11e-16). No runtime changes or relaxed tolerances are justified. Preserve these as explicit final-score divergences, not hidden passes. Probe files: /tmp/softdoe-tie-probe.json, /tmp/softdoe-tie-probe.R, /tmp/softdoe-stage2-probe.R.
Final canonical rank run started .work/experiments/mac-full-02, console .work/experiments/mac-full-02-console.log (session21586). No competing benchmarks/tests while timing.


## Final expanded experiment evidence
- Canonical full run .work/experiments/mac-full-02 completed:444configurations ×2backends =888 comparisons. All888 validation/input checks passed;885comparisonspassed,3final-Qagreementfailures (0066CPU,0069CPU,0427CPU) deliberately retained. Among826shared-input/same-backend valid cases:822designsagree,1equivalent-scoredifferentdesign,3divergentfinalscores.32native-initializer cases report qualitydifferences without claimingidenticaldesigns;30invalid-input checks passed.
- Maximum same-design independentPython/RCPU Qrelativeerror1.434587683684614e-14; maximumPythonMetalreturnedQ vsRCPUrescoring2.5125791517351815e-5.
- Portable diagnostics for0066 and0069 reproducefirstdivergentsteps[3,4,2] and[2,4,3] (one-based), identicalcandidate matrices, ±8.88e-16scoredecisiondeltas. Bothnativeoptimizers match theirreference traces. Both-directionfixed-baseaugmentation differs atmost1.11e-16coordinates and1.024e-15relativeQ. No runtimechanges/tolerancerelaxation justified.
- Durable benchmark evidence:benchmarks/mofat2-mac-2026-10-17 (888-rowCSV,1716timingrecords with5148samples,manifest,environment/sourcefingerprints,diagnoses,README andchecksums). Originalrawdesigns/logsremainmac-full-02. Executionandlateranalysis fingerprintskeptseparate;finalanalyzerrecheckedallrecordeddata.
- CurrentordinarysuitewithliveR/PyTorch/actualMetal:342passed34CUDA-onlyskips. Runnerintegritytests:37passedonPython3.9.6 aswellas3.14.7. FinalCLI smokeCPU/Metal:64passed. RuffE9/F,Rparsing,CFFschema,grant/softwarecitationcontent,andgitdiff--checkpass.
- Freshupdatedrelease:dist/release-checks/run-gqg822k9,Python3.14.7wheelandsdist each323passed53skips;strictmetadata,pipcheck,allREADMEPythonexamplesandexamplescriptpass.All29releaseinputhashesmatchcurrentcheckout;archivecontainssoftwarecitation,NSFgrantandexperimentcommands.
- Changedthisgoal:6newtest/experiment/diagnosticfilesin tests/,README.md,CITATION.cff,docs/experiments.md,docs/verification.md,.gitignore,task_plan.md,notes.md,andbenchmarkevidencefolder. Runtime implementation,siblingRsource/installedRpackage,existingfixturesanduserworktreechangespreserved. NoWindows/Linux/CUDA/stresshardwareexecutionclaimed;no publication.

# CPU final-Q correction investigation (2026-10-17)

- Reproduced all three original failures on Python3.14.7/installed MOFAT2.0 in
  `.work/q-fix/before`; exit1, all own-design validation passes. New focused tests
  against freshly captured unmodified R results fail3/3 before runtime edits;
  log `.work/q-fix/regression-before.log`.
- Concrete arithmetic mismatch: R `scaled_rhs_fast` divides direct kernel products
  by their column maximum on the normal alpha=1 path. Python `CpuState::initialize`
  previously logged every product then exponentiated log differences. Testing a
  direct-scaling correction with entire-column log fallback for subnormal/zero
  products. Search comparison and experiment acceptance tolerances unchanged.
- Rebuild uses existing isolated `.work/release-venv`; its import is being switched
  from the old wheel to this checkout for development. Historical wheels unchanged.
- Tool error: sandbox pip download could not resolve PyPI. The failed process ended;
  permitted network escalation installed pybind113.1.0 in the isolated environment.
  No user environment or R package was modified.
- Direct scaling alone fixes0069 but leaves0066/0427 failing. Armadillo's small
  sum-of-squares uses two accumulation chains; Python used one. Applying paired
  accumulation fixes all3 regression cases. Both full search diagnostics now have
  no differing accept/reject decisions; both native/reference paths agree and the
  final Q values match R. Files `.work/q-fix/paired-0066.json` and `paired-0069.json`.
- Full suite with real Metal/PyTorch/live R:345passed34CUDA-onlyskips,13.58s,
  `.work/q-fix/full-pytest-01.log`. Full888matrix with3repeats now running in
  `.work/q-fix/mac-full-01`, execsession77604. Avoid competing builds/tests while
  timing. The full matrix and fresh distribution checks remain pending.
- Exploratory probe corrections: initial Rcpp header paths omitted the installed
  `current/` subdirectory; located with rg. First intermediate-score probe used
  final augmentation beta1/144 rather than initial-search beta1/64; corrected.
  Such probe outputs are diagnostic scratch only, not acceptance evidence.
- Added analytic one-point Q checks at dimensions1/142/143/160 to exercise normal,
  subnormal and fully underflowed correlation columns, including a simultaneous
  normal column. These use the closed-form kernel score, no R or mpmath dependency.
- Source change remains limited to CpuState::initialize and CpuState::norm in
  src/softDOE/_native.hpp. CPU strict > and GPU decision policies are unchanged.
  R's small-vector sum(square()) is paired through Armadillo op_dot; its larger
  reductions may use BLAS, so the correction is not a promise of bitwise universal
  hardware/BLAS parity. The original score tolerances are retained in regression
  tests and the full experiment runner.
- Full matrix mac-full-01 finished887pass1fail: new CPU0118_a_soft_grid,
  p5l4m64N128seed42n_aug4standard, normalizedQdifference-8.093%. All888
  validation checks still pass. Added this case from the unchanged R results to
  the regression fixture; do not finalize the correction while it remains.
- Python3.9's rebuilt native extension also passes the original3 and the4
  analytical tests, but fails0118. A build command accidentally changed the user's
  .venv editable target; immediately reinstalled its original .venv/src/softdoe
  target and verified the original import path was restored. Future tests use
  .work/editable-check or fresh isolated distribution environments.
- 0118's R and Python correlation matrices are identical, but vendor Cholesky
  factors/solutions differ by1-2ULP. R's bundled reference LAPACK uses recursive
  DPOTRF2 panels; SciPy's optimized vendor POTRF uses a different order.
  A Python probe implementing the recursive TRSM/SYRK sequence matched R L and Y
  bit-for-bit (raw R doubles exported as hex to exclude JSON rounding). Reference:
  https://www.netlib.org/lapack/explore-html/d1/dd3/group__potrf2_ga7a1158271be5fac6e3d89b7ca8d71a07.html
- Implemented fixed recursive lower-Cholesky panels for n<=64 through existing
  SciPy BLAS capsules (added dsyrk); larger matrices keep vendor POTRF. All four
  case regressions and numerical/related tests pass:121passed38skips. New0118
  diagnostic has no decision divergence and zero final Q discrepancy. CPU full
  matrix running at .work/q-fix/cpu-recursive-01 (session60479),1repeat.
- Corrected another scratch probe: Python3.14 built-in sum is compensated and
  cannot represent naive summation; replaced with explicit scalar loops when
  distinguishing summation orders. No scratch-probe shortcut is used as test proof.

- Recursive-panel source passed the full444CPU matrix and353pytest checks on
  Python3.14, but0118 still failed on3.9/OpenBLAS. Stopped the premature
  mac-full-final timing run. Its incomplete records are not acceptance evidence.
- Compensated-norm scratch experiment fixes0118 onbothversions but reintroduces
 0066/0427, so not adopted. More accurate arithmetic alone cannot guarantee an
  identical heuristic path when comparison uses strict > at a tie.
- Independent80digit mpmath calculation:0066 first-divergent scores agree within
 8.5e-81;0118 candidate-incumbent is-8.483970443837e-16 while doublePython reports
 +1.776e-15 andR0. InputQMC is shared. These are rounding-dependent decisions,
 not a wrong Q formula. Evidence:.work/q-fix/high-precision-results.json.
- A scratch extension using system Accelerate onPython3.9 reproduces allfour
 reference finalscores. Implemented macOS CBLAS calls with checked typed headers,
 retaining SciPy capsule BLAS onotherplatforms andexisting vendorlargePOTRF.
 setup.py links the built-in Accelerate framework. No new Python dependency.

- Final macOS Accelerate source: Python3.9 regression11/11pass, fullCPU444/444
 (1repeat), andPython3.14 fullsuite353pass34CUDA-onlyskips(13.37s).
 Logs:.work/q-fix/accelerate-focused-py39.log,cpu-accelerate-py39/,
 full-pytest-accelerate.log. StartedfullCPU/Metal3repeat mac-accelerate-final.
- One earlierPython3.9 focused invocation ranbeforethebuildfinished and sawtheold
 extensionfailure; the completedbuildandexplicitrerun abovearethevalidresult.
- New exacttrajectoryfixtures aremarkedmacOS-only. Their referencepaths depend
 onAccelerate; treating themasuniversalBLAS outputs wouldbeincorrect. Allseven
 newclosed-form/kernel-solve checks runacrossplatforms. The existingexperiment
 harnessisunchangedandcontinuesreporting materialfinalQ differencesasfailures.

- Final full-run manifest is identical to the original canonical 888-case audit.
  R's printed-function MD5 differs only because `capture.output` includes process
  bytecode addresses. After normalizing only those addresses, all captured
  function bodies/namespaces are identical; stable SHA256 and raw hashes are
  recorded in the new audit's reference-comparison.json. R/dependency versions
  also match. No R package or experiment gate was changed.

## Completed CPU correction audit

- Final full run `.work/q-fix/mac-accelerate-final`: 888 passed, 0 failed,
  all 888 validations passed, three timed repeats. Among 826 shared-input
  comparisons, 825 designs agree and one has equivalent score; 32 use native
  initializers, and all 30 invalid-input comparisons pass. Maximum same-design
  Python/R CPU Q relative error: 1.421577599204407e-14. Maximum returned Metal Q
  error against R CPU: 2.5125791517351815e-5.
- Final diagnostics for 0066, 0069, and 0118 show no differing search decision.
  Python 3.9's independent 0118 diagnostic also has no divergence. Independent
  80-digit checks are reproducible with the audit's precision_check.py.
- Fresh Python 3.14.7 wheel/source installs: 334 passed, 53 optional skips each;
  report `dist/release-checks/run-wjpahnno/report.json`. Fresh Python 3.9.6
  wheel/source installs: same counts; report `run-cr09hhm8/report.json`.
  All strict metadata, assets, isolated imports, dependency checks, live R,
  actual Metal, README blocks, and examples passed. All 29 checkout/source
  archive input hashes and artifact hashes match. Consolidated current artifacts:
  `dist/release-cpu-fix-2026-10-17/`; older builds are historical.
- Updated README, test_README, experiment/verification/performance docs and the
  new audit. Syntax, Ruff E9/F, whitespace, Markdown fences, and local links pass.
  Historical benchmark SHA256SUMS still validate unchanged. New audit has
  28 checksummed files. Original user .venv import target is restored and checked.
- Runtime changes are limited to `_native.hpp` and macOS linkage in `setup.py`.
  New tests comprise four macOS trajectory fixtures and seven portable analytical
  or independent-solve checks. No search policy, comparison gate, R package,
  old fixture, unrelated working-tree edit, commit, push, or publication changed.
- Limits remain explicit: this verifies Apple M4/macOS configurations, not
  Windows/Linux/CUDA, Intel Mac, stress inputs, or universal bitwise BLAS parity.

## Current platform verification (2026-10-22)
- Current Python 3.14 macOS full suite: 353 passed, 34 CUDA-only skips (13.41s), including live R. Fresh Python 3.14 wheel/source release checks both pass.
- All 29 release input hashes exactly match the verified October 17 corrected revision, including README. The October 2 Colab notebook results are historical and insufficient for current verification.
- Colab connected to RTX PRO 6000 Blackwell, Python 3.13.15, PyTorch 2.11.0+cu130, CUDA 13.0. Current 256024-byte bundle prepared with SHA256 3d0799067576caca775beeb8518fdf08f20025118874b04877d1115128873aaa.
- Initial upload blocked by automatic review; user explicitly approved source/test bundle upload to Colab. Browser file chooser event timed out; native picker fallback in progress.
- First Python 3.9 release verification stopped before build; inspect console and install missing build tools in isolated audit environment as needed.

## Final October 22 platform verification

- Mac full suite:353pass34CUDA-onlyskips. Colab RTX PRO6000 Blackwell/CUDA13/PyTorch2.11/Python3.13:339pass48Mac/liveRskips. JUnit identity union confirms387/387unique items passed on applicable hardware, including34/34actualCUDAitems.
- Both fresh wheel/source installs onPython3.9and3.14:334pass53optional skips each; build/metadata/assets/dependencies/isolatedimports/examples/README checks pass.
- Colab full workflow sweep:444/444 configurations (429valid15expectederrors); warmup+1repeat, no fallback, structural/repeatability/candidate/Q checks. IndependentMacRMOFAT2CPUrescoring:429/429pass;maxQrelativeerror2.8441214342104595e-14,maxnormalizederror1.0895459443866998e-15.
- Colab latestPyDOE initially upgradedNumPy pastNumba's constraint; pinnedNumPy2.2.6 andinstalledmissingJedi. Finalpipcheckclean; wholeCUDApytest rerunpasses. OriginalNumPy2.5.4run also passed. No softDOE defect found.
- Source archive SHA256 ad8e8781e4a8fd5876f32fdc2c5a7ae8f4f0906d516f314d500fbeab1b3cbfc5. All29sourceinputs and37validationfiles verified beforeColabinstall andagainafteraudit. Historical28checksums pass. Downloadedevidence SHA256 f15f851c439cca3ded431b1359e12eb9f5f2a1f067d537e8b6a43e69290c79cd verified locally.
- Evidence:benchmarks/verification-2026-10-22/README.md,combined-tests.json,per-test-platform-results.csv,colab/cuda-full/,colab-result.jpg. Current notebook:https://colab.research.google.com/drive/1PeoSvm4QLXx976Qj05izFsEFFL56aWl7. Appendeddatedcells only; oldcells notrerun.
- Changed:docs/verification.md,docs/experiments.md,tests/test_README.md,task_plan.md,notes.md,newbenchmarkfolder;isolatedaudittooldependencies andephemeralColabenvironment. Untouched:src/,testcode/fixtures,packaging,user.venv,R,historicalevidence. NoGitwrites/publication. Stress/otherGPUs/Windows/IntelMac/WSL2/R-CUDAoptimizer notverified.

## User-facing README rewrite (2026-10-22)

User requested a professional package README centered on purpose, installation,
functions, examples, citation, acknowledgement, and license. Removed release
migration commentary, developer verification commands, MOFAT comparison protocols,
and historical numerical-fix narrative. Those records remain in docs/benchmarks.

README.md: 287 to 207 lines; 1624 to 1008 words. Installation targets the upcoming
0.2.0 release with `python -m pip install "softDOE>=0.2.0"`; no claim that publication
has occurred. Preserved the software citation/CFF agreement and NSF DMS-2310637.

Verification: six Python examples pass on 3.9.6 and 3.14.7; PyPI Markdown rendering,
repository link checks, strict Twine checks, and exact README inclusion in freshly
built wheel/sdist all pass. All 28 other package inputs match the prior audit.
Evidence and before-copy: `.work/readme-release-2026-10-22/`.
Only README.md/task_plan.md/notes.md changed; source, tests, metadata, existing
README.html, and detailed verification evidence preserved. Nothing published.
