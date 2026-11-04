# Task Plan: Repair rpy2 R linkage

- [x] Reproduce import warning and inspect installed R and binary linkage.
- [x] Rebuild rpy2-rinterface 3.6.7 against installed R 4.5.2, preserving dependencies.
- [x] Verify API-mode import, R package loading, numeric R operations, and dependency consistency.

Scope: repaired .venv only; preserved all existing repository changes and system R.
Status: complete. Restart any existing Python process or notebook kernel.

Errors resolved: sandbox network restriction required approved network access for source download; initial backup encountered external cached bytecode, corrected to archive only files inside .venv.

# Active task: Investigate measure symmetry report

- [ ] Inspect Python and installed MOFAT measure implementations and design ordering.
- [ ] Reproduce the supplied Friedman-function example with fixed R seeds; compare Python/R on identical data.
- [ ] Check factor-exchange symmetry and finite-design estimates; fix a confirmed defect if present.
- [ ] Run focused verification and explain formulas and findings.

Status: inspecting implementations. Preserve existing user changes.

## Completed investigation
- [x] Inspected Python and installed MOFAT implementations; identical formulas on valid input.
- [x] Compared identical X/y at R seeds 1, 42, 123 in both Python checkouts.
- [x] Verified column-swap equivariance and exact equality on exchange-balanced designs.
- [x] Current-source targeted tests: 8 passed. Explanation prepared; no formula change justified.

Status: complete. Distinct environment finding: default import uses .venv/src/softdoe, not this working tree; older validation explains initial 3 validation-test failures.

# Active goal: Prepare softDOE for public package release

## Required outcomes
- README links verified paper and provides citation, general installation and runnable guides.
- All six public functions audited and tested, with fixes for confirmed defects.
- R MOFAT 2.0 is the reference, as confirmed by the user: establish exact source/version and test provenance before claiming parity.
- Independent developer and user-test agents challenge implementation and user instructions.
- Clean sdist/wheel build and fresh-environment installation, examples, tests, metadata, and distribution contents verified.
- Keep existing user changes; do not publish or push.

## Execution
- [x] P0: establish reference version, public API coverage, baseline tests, and release requirements.
- [x] P0: implementation agent audits/fixes functions; user agent probes failures independently.
- [x] P0: revise README/install/citation and packaging/verification instructions.
- [x] P0: add reproducible live R comparison coverage for all mathematical public APIs.
- [x] P0: build artifacts; test fresh wheel and sdist installs plus documented examples/commands.
- [x] P1: complete cross-platform CI configuration and document unverified hardware honestly.
- [x] P0: final confirmation run against MOFAT 2.0 and release evidence audit.

Status: complete. User confirmed MOFAT 2.0; final live comparisons and release evidence audit passed.

## Release goal audit (2026-10-17)
- [x] Verified paper metadata; README paper link/BibTeX and schema-valid CITATION.cff added.
- [x] Replaced personal archive installation with general source/archive/environment instructions; distinguished unpublished0.2.0 from public0.1.0.
- [x] Two independent agents implemented fixes and adversarial user-workflow tests; all six APIs exercised.
- [x] Fixed measure overflow/underflow while preserving tiny contrasts; stable extreme Q kernels; legacy unseeded RNG behavior.
- [x] Added exact-version live R tests: 12 comparisons against available local MOFAT2.0. Regeneration scripts reproduce existing results, without modifying reference files.
- [x] Built clean sdist/wheels; fresh wheel and sdist installs on Python3.9.6 and3.14.7 each:286 passed,53 skipped; all README/example commands passed.
- [x] Optional PyTorch plus real Metal:305 passed,34 CUDA-only skips; Python statement coverage91%.
- [x] Corrected old-pip editable-install failure; documented sequence passes (274 passed,65 expected skips).
- [x] Static E9/F checks, diff whitespace, dependency checks, CFF schema and strict package metadata pass.
- [x] Cross-platform CI configured; Windows/Linux/CUDA execution explicitly unverified for this revision.
- [x] Deliverables consolidated under dist/release-2026-10-17; all29 release input hashes match current checkout and tested archives.
- [x] Baseline identity resolved: user confirmed "Use MOFAT 2.0 as the reference"; installed R and source versions match.

Current status: release candidate and local verification complete against the confirmed MOFAT 2.0 baseline. No publish, push, or commit performed.

## Historical continuation audit 2 (2026-10-17; blocker subsequently resolved)
Previous goal turn: progress (implementation, documentation, tests and release artifacts completed).
Current evidence: both final release reports remain passed; all29 release inputs and artifact hashes still match. Installed R and sibling source/archive are MOFAT2.0. Available local git history has DESCRIPTION versions1.0 ->1.1 ->2.0, with no0.2.0. This narrows the remaining action to user clarification or a different reference location; no further implementation change is justified by current evidence.
Same baseline-identity blocker now observed in two consecutive goal turns. Leave goal active; blocked threshold of three is not met.

## Historical continuation audit 3 (2026-10-17; blocker subsequently resolved)
Previous turn gathered new reference-history evidence but could not resolve the baseline identity. Revalidated now: installed/source MOFAT remain2.0, both release reports pass, and all release input hashes remain unchanged. No user clarification or0.2.0 reference has arrived. The same blocker has persisted across three consecutive goal turns, and no meaningful remaining work can establish the requested0.2.0 comparison without user input or a new reference. Mark goal blocked pending confirmation of2.0 or location of0.2.0. Source, tests and artifacts remain preserved.

## Final baseline confirmation (2026-10-17)
User confirmed: "Use MOFAT 2.0 as the reference." Updated the canonical verification document and audit summary accordingly. Final live R suite against the installed softDOE 0.2.0 wheel: 12 passed in 1.11 seconds; log at .work/release-audit/final-mofat-2.0.log. Both release reports remain passed; all 29 release input hashes match the current checkout and each source archive. All built artifact hashes and delivered SHA256SUMS match. Only documentation/status files changed in this follow-up, so tested release artifacts remain current. All requested outcomes complete within the documented platform limits.

# Active goal: Broad MOFAT 2.0 experiments and software citation

- [x] Inspect existing comparison coverage, generator arguments, R backends, and timing conventions.
- [x] Add a documented scenario matrix and portable experiment scripts under tests/ for soft/a_soft, varying p/l/n_aug and other meaningful arguments; require requested hardware and exact R baseline.
- [x] Run broad CPU and native Metal comparisons on this Mac, recording repeated runtime, returned Q, independently checked Q, design invariants, backend use, and reproducible inputs.
- [x] Add Windows/Linux/CUDA instructions and experiment commands to README; replace paper BibTeX with package citation and add U.S. NSF grant DMS-2310637 acknowledgment.
- [x] Verify new scripts, updated documentation/citation, and affected existing tests; inspect experiment results and address defects.
- [x] Report actual scenario counts, numerical/runtime findings, and remaining platform limits with paths to results.

Status: complete. Corrected canonical-rank full run mac-full-02 finished:888 validation checks passed;885 comparisons passed and3 final-Q disagreements remain explicit strict failures. Two distinct CPU trajectories diverge on1ULP decisions; fixed-base augmentation and independentQ agree, so baseline strict-comparison semantics were preserved. Evidence saved in benchmarks/mofat2-mac-2026-10-17/. Ordinary suite342passed34CUDA-onlyskips;37runner tests also pass onPython3.9.6;finalsmoke64passed. Fresh3.14.7wheel/sdist each323passed53skipped, current29releaseinputhashesmatch. Windows/Linux/CUDA and opt-in stress hardware runs not claimed; portable scripts/instructions provided as requested. Runtime/reference code and earlier artifacts preserved; no publish/push.

# Active goal: Resolve CPU final-Q disagreements before release

User requests investigation and a justified correction for cases 0066, 0069, and
0427 against MOFAT 2.0, with regression/release verification. Prior statements that
rounding required no change are hypotheses to revisit, not completion evidence.

- [x] Reproduce the three failures with current source and confirm environment/reference identity.
- [x] Compare native Python/R scoring, arithmetic order, and search decisions; use independent high-precision evidence where needed.
- [x] Implement a general correction supported by the diagnosis, retaining meaningful final-Q agreement gates; add focused regression tests.
- [x] Run the original failing cases, full CPU/Metal matrix, ordinary/live-R tests, and fresh distribution verification.
- [x] Update test_README and verification evidence with exact results, implementation limits, and release status.

Status: complete. Original cases 0066/0069/0427 and newly discovered 0118 pass
on both tested Python versions. Final CPU/Metal run: 888/888, three repeats;
Python 3.9 CPU run: 444/444. Full Python 3.14 suite: 353 passed, 34 CUDA-only
skips. Fresh wheel and source installs on Python 3.9.6 and 3.14.7 each passed
334 tests with 53 optional skips, plus strict metadata, dependency, asset,
example, and README checks. All 29 release inputs and archives match by hash.

Evidence: `benchmarks/mofat2-cpu-fix-2026-10-17/` (28 checksummed files),
`tests/test_README.md`, `dist/release-cpu-fix-2026-10-17/`. Reports:
`dist/release-checks/run-cr09hhm8/` and `run-wjpahnno/`. Original gates, R,
old fixtures, and the historical audit remain unchanged. Windows/Linux/CUDA,
Intel Mac, and stress execution remain unverified. No publication or git writes.

# Current task: macOS and Google Colab CUDA verification (2026-10-22)

- [ ] Audit test coverage, skip gates, source identity, and previous evidence.
- [ ] Rebuild and run current checkout tests on macOS CPU/Metal, including live R.
- [ ] Run the exact current source and CUDA-specific tests on Google Colab GPU; retain logs, source hashes, device identity, and skipped-test reasons.
- [ ] Verify distributions where appropriate and write a dated verification report with explicit limitations.

Scope: preserve existing worktree edits; test current softDOE 0.2.0 rather than public Git HEAD or an older editable install. Evidence directory: `.work/verification-2026-10-22/`.

2026-10-22 progress:
- [x] Audited current tests, actual-device assertions, skip conditions, and historical source identities.
- [x] macOS full suite: 353 passed / 34 CUDA-only skips; fresh Python 3.9 and 3.14 wheel and source installs each 334 passed / 53 optional skips.
- [x] Colab current source: 339 passed / 48 expected Mac/live-R skips; all 34 real-device CUDA tests passed. Upload explicitly approved by user after initial automatic review block.
- [ ] Run additional 444-case CUDA workflow matrix and independently rescore returned designs with local R CPU.
- [ ] Download Colab evidence, compare per-test coverage across platforms, and finalize report.

Setup issues: Python 3.9 audit initially lacked build; installed build/twine in `.work/editable-check` and rerun passed. Native upload picker required KP_Enter. Installing current PyDOE upgraded Colab NumPy beyond preinstalled Numba's constraint; checking compatible NumPy 2.2.6 and rerunning CUDA suite before finalizing environment.

## Completed current verification (2026-10-22)

- [x] Downloaded and checksum-verified Colab evidence; native Finder copied the archive after macOS Downloads privacy blocked shell reads.
- [x] Compared JUnit test identities: exactly 387 unique tests, all passed on at least one applicable platform; 34/34 Mac CUDA skips passed on Colab.
- [x] Full CUDA workflow matrix: 444/444 (429 valid, 15 invalid-input rejections), no unexpected CPU fallback.
- [x] Independently rescored all 429 CUDA-produced designs using unmodified local R MOFAT 2.0 CPU: all passed, max Q relative error 2.8441214342104595e-14.
- [x] Saved durable evidence, Colab screenshot, per-test matrix, release reports, and dated README in `benchmarks/verification-2026-10-22/`; updated verification/experiment docs and test README.

Status: complete. Implementation, tests, fixtures, numerical gates, R, and user `.venv` unchanged. Optional stress, Windows/Intel Mac/WSL2, other GPUs, and R CUDA optimizer parity remain outside executed coverage. No publish, push, or commit. A computer-use tool unexpectedly stalled during evidence transfer; completed test results were unaffected.

# Current task: user-facing release README (2026-10-22)

- [ ] Rewrite installation and usage around the six public functions; remove migration, developer-audit, and MOFAT experiment details from the README.
- [ ] Preserve software citation, NSF acknowledgement, license, and essential GPU/input guidance.
- [ ] Execute all Python examples on Python 3.9 and 3.14; build distribution metadata and check Markdown rendering/links.

Scope: prepare README.md for the upcoming 0.2.0 upload; do not publish. Detailed verification records remain in docs/ and benchmarks/. Backup and validation outputs: `.work/readme-release-2026-10-22/`.

## Completed README rewrite (2026-10-22)

- [x] Replaced migration/release-status installation prose with release-facing pip installation and concise compiler/GPU prerequisites.
- [x] Removed development verification and MOFAT experiment sections; retained user-facing examples for all six functions, software citation, NSF acknowledgement, and licenses.
- [x] Ran all six Python examples successfully on Python 3.9.6 and 3.14.7.
- [x] Rendered with PyPI's Markdown renderer, checked repository links, built fresh sdist/wheel, passed strict Twine checks, and verified packaged README bytes.

Status: complete. README reduced from 1624 to 1008 words (38%). Source/test code, fixtures, packaging settings, citation metadata, README.html, and detailed experiment records remain untouched. No publication. Only README.md and planning notes changed; temporary evidence is in `.work/readme-release-2026-10-22/`.
Setup issues resolved: audit environment lacked `wheel` and the optional Markdown renderer; installed both there. A scratch HTML assertion initially omitted the renderer's heading ID/anchor attributes; corrected the assertion and confirmed rendering passed.
