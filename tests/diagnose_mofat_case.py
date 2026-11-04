"""Trace a standard CPU A-SOFT disagreement without changing comparison gates.

Example:
  python tests/diagnose_mofat_case.py experiment-results/mac --case 0066_a_soft_grid \
      --output experiment-results/0066-diagnostic.json

Requires the installed Python package, Rscript, MOFAT 2.0, and jsonlite.
The output is a new JSON file; existing experiment evidence is never modified.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from unittest.mock import patch


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def compare_designs(left, right):
    import numpy as np
    left, right = np.asarray(left), np.asarray(right)
    return dict(exactly_equal=bool(np.array_equal(left, right)),
                max_absolute_difference=float(np.max(np.abs(left-right))))


def score_errors(left, right):
    return {key: abs(left[key]-right[key])/max(abs(left[key]), abs(right[key]))
            if left[key] or right[key] else 0.0 for key in ("Q", "normalized_Q")}


def python_trace(case):
    import numpy as np
    from softDOE import _native
    from softDOE._engine import ArrayOps
    from softDOE._soft import _prepare, _optimize, _standard_ofat_constr
    fixed = {**case["base"], "A": np.array(case["base"]["A"], dtype=float)}
    with patch("softDOE._soft._a_fun", return_value=fixed):
        prepared = _prepare(case["p"], case["l"], case["m"], case["seed"],
                            2*case["l"]+case["n_aug"])

    def evaluate(base):
        design = _standard_ofat_constr(base, case["p"], prepared["delta"])
        score = float(_native.evaluate_cpu(design, prepared["qmc"], prepared["beta"]).min())
        return design, score

    base = prepared["A"].copy()
    current_design, current = evaluate(base)
    initial = dict(A=base.tolist(), design=current_design.tolist(), log_Q=current)
    trace = []
    p0 = case["p"] if prepared["remainder"] is None else prepared["remainder"]
    p1 = 0 if prepared["quotient"] is None else prepared["factorial_l"]*prepared["quotient"]
    for i in range(case["l"]-1):
        for j in range(i+1, case["l"]):
            for k in range(1, p0):
                column = p1+k
                candidate = base.copy()
                candidate[[i, j], column] = candidate[[j, i], column]
                design, score = evaluate(candidate)
                accepted = score > current
                trace.append(dict(step=[i+1, j+1, column+1], incumbent_log_Q=current,
                                  candidate_log_Q=score, delta=score-current,
                                  accepted=accepted, A=candidate.tolist(), design=design.tolist()))
                if accepted:
                    base, current_design, current = candidate, design, score
    native_design, native_score = _optimize(prepared, "standard", ArrayOps())
    result = dict(initial=initial, trace=trace,
                  reference=dict(design=current_design.tolist(), log_Q=current),
                  native=dict(design=native_design.tolist(), log_Q=native_score),
                  reference_native=compare_designs(current_design, native_design))
    return result, prepared


def historical_result(directory, filename, case_id, replay):
    path = directory/filename
    if not path.exists():
        return dict(available=False)
    for line in path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if item["id"] == case_id and item["backend"] == "cpu":
            result = dict(available=True, recorded_status=item["status"])
            if "design" in item:
                result["design"] = compare_designs(replay["design"], item["design"])
                result["Q_relative_errors"] = score_errors(replay, item)
            return result
    return dict(available=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment_dir", type=Path, help="directory containing manifest.json")
    parser.add_argument("--case", required=True, help="exact case ID from the manifest")
    parser.add_argument("--output", required=True, type=Path, help="new diagnostic JSON file")
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--r-timeout", type=int, default=300)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new file")
    if args.r_timeout < 1:
        parser.error("r-timeout must be positive")
    manifest = json.loads((args.experiment_dir/"manifest.json").read_text(encoding="utf-8"))
    if manifest.get("mofat_version") != "2.0":
        parser.error("this diagnostic requires a MOFAT 2.0 experiment")
    case = next((case for case in manifest["cases"] if case["id"] == args.case), None)
    if case is None:
        parser.error("case is absent from the manifest")
    if (case["function"] != "a_soft" or case["structure"] == "strict"
            or case.get("expect_error") or case.get("supplied_design") is not None
            or case.get("base") is None):
        parser.error("supported cases: valid standard CPU A-SOFT with a shared base, including structure fallback")
    rscript = shutil.which(args.rscript)
    if rscript is None:
        parser.error("Rscript was not found; use --rscript with its executable path")
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[variable] = "1"
    import numpy as np
    import softDOE
    from softDOE import _native
    from softDOE._qmc import _sobol_points, _unique_rows_preserve_order

    python, prepared = python_trace(case)
    with tempfile.TemporaryDirectory(prefix="softdoe-diagnostic-") as temporary:
        payload, reference = Path(temporary)/"input.json", Path(temporary)/"reference.json"
        write_json(payload, dict(case=case, python=python))
        command = [rscript, "--vanilla", str(Path(__file__).with_suffix(".R")), str(payload), str(reference)]
        completed = subprocess.run(command, text=True, capture_output=True, timeout=args.r_timeout)
        if completed.returncode:
            raise RuntimeError("R diagnostic failed:\n" + completed.stderr + completed.stdout)
        r = json.loads(reference.read_text(encoding="utf-8"))

    first = None
    for index, (left, right) in enumerate(zip(python["trace"], r["trace"])):
        if left["step"] != right["step"]:
            raise RuntimeError("Python and R enumerated different candidate steps")
        if left["accepted"] != right["accepted"] and first is None:
            first = dict(trace_index=index, step=left["step"],
                         python_delta=left["delta"], r_delta=right["delta"],
                         python_accepted=left["accepted"], r_accepted=right["accepted"],
                         candidate_design=compare_designs(left["design"], right["design"]))
        right["python_log_Q"] = float(_native.evaluate_cpu(
            np.asarray(right["design"]), np.asarray(r["evaluation"]), prepared["beta"]).min())
    if len(python["trace"]) != len(r["trace"]):
        raise RuntimeError("Python and R produced different trace lengths")
    boundary = .25/(2*case["l"]+case["n_aug"])
    pool = _unique_rows_preserve_order(boundary+(1-2*boundary)*_sobol_points(case["N"], case["p"]))
    beta = (2*case["l"]+case["n_aug"])**-2
    replays = {}
    for owner, base in (("python", python["native"]["design"]), ("r", r["native"]["design"])):
        result = softDOE.a_soft(n_aug=case["n_aug"], N=case["N"], ofat_design=base, backend="cpu")
        scored = softDOE.Q(1, beta, result["design"], pool, backend="cpu")
        replay = dict(design=result["design"].tolist(), Q=result["Q"], normalized_Q=result["normalized_Q"],
                      returned_score_relative_errors=score_errors(result, scored))
        reference = r["augmentation"][owner]
        replays[owner+"_base"] = dict(python=replay, r=reference,
            design=compare_designs(replay["design"], reference["design"]),
            Q_relative_errors=score_errors(replay, reference))
    report = dict(case=case, created_at=datetime.now(timezone.utc).isoformat(),
        experiment_directory=str(args.experiment_dir.resolve()),
        python_version=softDOE.__version__, python_package=str(Path(softDOE.__file__).resolve()),
        r_environment=r.pop("environment"),
        initial_design=compare_designs(python["initial"]["design"], r["initial"]["design"]),
        evaluation=compare_designs(prepared["qmc"], r.pop("evaluation")),
        first_decision_divergence=first, python=python, r=r, augmentation=replays,
        historical_python=historical_result(args.experiment_dir, "python-results.jsonl", args.case,
                                            replays["python_base"]["python"]),
        historical_r=historical_result(args.experiment_dir, "r-results.jsonl", args.case,
                                       replays["r_base"]["r"]),
        interpretation="Diagnostic evidence only; no experiment status or comparison tolerance is changed.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(f"Wrote {args.output}; first decision divergence: {first['step'] if first else 'none'}")


if __name__ == "__main__":
    main()
