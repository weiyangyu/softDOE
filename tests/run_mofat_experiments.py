"""Run reproducible SOFT/A-SOFT comparisons with an installed R MOFAT 2.0.

Examples (run from repository root after installing this checkout):
  python tests/run_mofat_experiments.py --profile smoke --backends cpu
  python tests/run_mofat_experiments.py --profile full --backends cpu metal
  python tests/run_mofat_experiments.py --profile full --backends cpu cuda

No rpy2 is needed. Explicit GPU requests must execute on that GPU, not fall back.
Each output directory is new; existing experiment evidence is never overwritten.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from unittest.mock import patch

from experiment_cases import case_counts, experiment_cases


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_lines(path: Path) -> list:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def append_json(path: Path, value) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", choices=["smoke", "full", "stress"], default="full")
    parser.add_argument("--backends", nargs="+", choices=["cpu", "metal", "cuda"], default=["cpu"])
    parser.add_argument("--r-backend", choices=["same", "cpu"], default="same",
                        help="same compares the same backend; cpu compares Python GPUs with R CPU")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--mofat-version", default="2.0")
    parser.add_argument("--rscript", default="Rscript", help="Rscript command or full executable path")
    parser.add_argument("--r-timeout", type=int, default=14400, help="seconds allowed per R stage")
    parser.add_argument("--output", type=Path, help="new output directory; default experiment-results/<timestamp>")
    parser.add_argument("--list", action="store_true", help="print case counts and arguments without running")
    parser.add_argument("--case", action="append", help="run just these exact case IDs (repeatable)")
    args = parser.parse_args()
    if min(args.repeats, args.threads, args.r_timeout) < 1:
        parser.error("repeats, threads and r-timeout must be positive")
    args.backends = list(dict.fromkeys(args.backends))
    cases = experiment_cases(args.profile)
    if args.case:
        known = {c["id"] for c in cases}
        if set(args.case) - known:
            parser.error("unknown case IDs: " + str(set(args.case) - known))
        cases = [c for c in cases if c["id"] in args.case]
    return args, cases


def prepare_cases(cases: list) -> list:
    import numpy as np
    import softDOE
    from softDOE._soft import _a_fun
    for case in cases:
        case["base"] = None
        case["supplied_design"] = None
        supplied = case.get("supplied")
        if supplied:
            p, l = case["p"], case["l"]
            if supplied == "ofat":
                design = softDOE.soft(p, l, m=16, seed=case["seed"], backend="cpu")["design"]
            else:
                design = np.random.default_rng(case["seed"]).random((l*(p+1), p))
                if supplied == "boundary":
                    design[0] = 0; design[1] = 1
                elif supplied == "duplicate":
                    design[-1] = design[0]
                elif supplied == "bad_rows":
                    design = design[:-1]
                elif supplied == "out_of_bounds":
                    design[0, 0] = 1.01
            case["supplied_design"] = design.tolist()
        elif not case.get("expect_error") and case["initialization"] == "shared":
            base = _a_fun(case["p"], case["l"], case["seed"])
            # R indexes by truncation; PyDOE ranks can be sub-ulp below integers.
            # Share exact ranks so both level transforms receive the intended levels.
            case["base"] = {**base, "A": np.rint(base["A"]).tolist()}
    return cases


def arguments(case: dict, backend: str) -> dict:
    import numpy as np
    kwargs = {key: case[key] for key in ["p", "l", "structure", "m", "seed"]}
    kwargs["backend"] = backend
    if case["function"] == "a_soft":
        kwargs.update(n_aug=case["n_aug"], N=case["N"])
        if case["supplied_design"] is not None:
            kwargs["ofat_design"] = np.array(case["supplied_design"], dtype=float)
            kwargs["ofat_design"].setflags(write=False)
        if case["tag"] == "supplied_override":
            kwargs.update(p=-1, l=-1)
    return kwargs


def evaluation(case: dict):
    import softDOE
    from softDOE._qmc import _sobol_points, _unique_rows_preserve_order
    if case["function"] == "soft":
        return softDOE.qmc_generate(case["m"], case["p"])
    boundary = .25/(2*case["l"] + case["n_aug"])
    return _unique_rows_preserve_order(boundary+(1-2*boundary)*_sobol_points(case["N"], case["p"]))


def validate_design(case: dict, design, points, transported: bool = False) -> None:
    import numpy as np
    p, l = case["p"], case["l"]
    n_aug = case.get("n_aug", 0)
    assert design.shape == (l*(p+1)+n_aug, p), "incorrect output shape"
    assert np.isfinite(design).all() and ((0 <= design) & (design <= 1)).all(), "invalid coordinates"
    base = design[:l*(p+1)]
    if case["supplied_design"] is not None:
        if transported:
            np.testing.assert_allclose(base, case["supplied_design"], rtol=0, atol=2e-14)
        else:
            np.testing.assert_array_equal(base, case["supplied_design"])
    else:
        bound = .25/(2*l+n_aug)
        assert (design >= bound-1e-14).all() and (design <= 1-bound+1e-14).all()
        blocks = base.reshape(p+1,l,p)
        pairs = zip(blocks[:-1],blocks[1:]) if case["structure"] == "strict" else (
            (blocks[0], block) for block in blocks[1:])
        factors = []
        for left, right in pairs:
            changed = np.abs(left-right) > 1e-12
            assert np.all(changed.sum(axis=1) == 1), "not one factor per paired row"
            columns = np.flatnonzero(changed.any(axis=0))
            assert len(columns) == 1, "mixed factors in a block"
            factors.append(int(columns[0]))
        assert sorted(factors) == list(range(p)), "missing/repeated factor"
    if n_aug:
        added = design[-n_aug:]
        assert len(np.unique(added,axis=0)) == n_aug, "repeated augmentation candidate"
        available = {tuple(row) for row in points}
        if transported:
            assert all(np.any(np.max(np.abs(points-row),axis=1) <= 2e-14) for row in added), "row outside candidate pool"
        else:
            assert all(tuple(row) in available for row in added), "row outside candidate pool"


def compare_scores(actual: dict, expected: dict, backend: str) -> dict:
    import numpy as np
    # Fixed before collecting results. Metal scoring uses float32; CPU/CUDA use float64.
    tolerance = 3e-3 if backend == "metal" else 1e-8
    errors = {}
    for key in ["Q", "normalized_Q"]:
        a, b = float(actual[key]), float(expected[key])
        assert np.isfinite(a) and np.isfinite(b) and a >= 0 and b >= 0, key
        # Tiny Q values must not pass merely because an absolute tolerance dwarfs them.
        scale = max(abs(a),abs(b))
        error = abs(a-b)/scale if scale else 0.0
        errors[key+"_relative_error"] = error
        assert error <= tolerance, f"{key} relative error {error:.3g} > {tolerance}"
    return errors


def run_python_case(case: dict, backend: str, repeats: int) -> dict:
    import numpy as np
    import softDOE
    base = case["base"]
    context = nullcontext()
    if base is not None:
        fixed = {**base, "A": np.array(base["A"],dtype=float)}
        context = patch("softDOE._soft._a_fun", return_value=fixed)
    kwargs = arguments(case, backend)
    call = getattr(softDOE,case["function"])
    with context:
        if case.get("expect_error"):
            try:
                call(**kwargs)
            except (ValueError, TypeError) as error:
                return dict(status="expected_error",error=str(error))
            raise AssertionError("Expected invalid arguments to be rejected")
        def synchronize():
            if backend == "cuda":
                import torch
                torch.cuda.synchronize()
        warm = call(**kwargs)
        assert warm["backend_used"] == backend, warm.get("fallback_reason")
        seconds = []
        for _ in range(repeats):
            synchronize()
            start = time.perf_counter()
            result = call(**kwargs)
            synchronize()
            seconds.append(time.perf_counter()-start)
            assert result["backend_used"] == backend, result.get("fallback_reason")
            np.testing.assert_array_equal(result["design"],warm["design"])
            for key in ["Q", "normalized_Q"]:
                assert result[key] == warm[key], "non-reproducible score"
    telemetry = dict(status="ok", seconds=seconds, design=result["design"].tolist(),
                     Q=result["Q"],normalized_Q=result["normalized_Q"],
                     device=result["device"],backend_used=result["backend_used"],
                     fallback_reason=result["fallback_reason"],qmc_size=result.get("qmc_size"))
    try:
        validate_result(case, backend, result, telemetry)
    except Exception as error:
        telemetry.update(status="error",error=f"{type(error).__name__}: {error}")
    return telemetry


def validate_result(case: dict, backend: str, result: dict, telemetry: dict) -> None:
    import softDOE
    points = evaluation(case)
    validate_design(case, result["design"], points)
    levels = 2*case["l"]+case.get("n_aug",0)
    cpu = softDOE.Q(1,1/levels**2,result["design"],points,backend="cpu")
    telemetry["cpu_score"] = cpu
    errors = compare_scores(result,cpu,backend)
    if case["function"] == "soft":
        assert result["qmc_size"] == len(points)
    telemetry.update(cpu_score=cpu,self_score_errors=errors)


def run_r(args, output: Path, mode: str) -> None:
    command = [args.rscript,"--vanilla",str(Path(__file__).with_name("reference_experiments.R").resolve()),
               mode,str(output/"manifest.json"),str(output)]
    print(f"R {mode}: progress in {output / ('r-'+mode+'.log')}",flush=True)
    with (output/('r-'+mode+'.log')).open("w",encoding="utf-8") as log:
        result = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=args.r_timeout)
    if result.returncode:
        raise RuntimeError(f"R {mode} failed; see {output / ('r-'+mode+'.log')}")


def validate_times(values, repeats: int, label: str) -> None:
    import numpy as np
    values = np.atleast_1d(np.asarray(values,dtype=float))
    if values.ndim != 1 or len(values) != repeats or not np.isfinite(values).all() or not (values > 0).all():
        raise ValueError(f"{label} timings must contain {repeats} finite positive repetitions")


def index_records(records: list, expected: set, label: str) -> dict:
    indexed = {}
    for record in records:
        key = (record["id"],record["backend"])
        if key in indexed:
            raise ValueError(f"Duplicate {label} result: {key}")
        indexed[key] = record
    if set(indexed) != expected:
        raise ValueError(f"Incomplete {label} results: missing={expected-set(indexed)}, unexpected={set(indexed)-expected}")
    return indexed


def summarize(output: Path, manifest: dict, python_records: list) -> dict:
    import numpy as np
    cases = {c["id"]:c for c in manifest["cases"]}
    if not cases or len(cases) != len(manifest["cases"]):
        raise ValueError("Manifest must contain unique nonempty cases")
    expected = {(c,b) for c in cases for b in manifest["backends"]}
    index_records(python_records,expected,"Python")
    expected_r = {(c,b) for c in cases for b in manifest["r_backends"]}
    r_records = index_records(read_lines(output/"r-results.jsonl"),expected_r,"R")
    expected_cross = {(r["id"],r["backend"]) for r in python_records if r["status"] == "ok"}
    cross_scores = index_records(read_lines(output/"r-python-scores.jsonl"),expected_cross,"R rescore")
    rows = []
    for actual in python_records:
        case = cases[actual["id"]]
        backend = actual["backend"]
        rb = backend if manifest["r_backend"] == "same" else "cpu"
        ref = r_records[(case["id"],rb)]
        row = {k:case.get(k) for k in ["id","function","p","l","structure","m","n_aug","N","seed","initialization","supplied","tag"]}
        row.update(backend=backend,r_backend=rb,status="passed",error=None,
                   validation_status="not_run",agreement_status="not_applicable")
        try:
            if case.get("expect_error"):
                assert actual["status"] == ref["status"] == "expected_error", str((actual,ref))
                row["validation_status"] = "passed"
            else:
                assert actual["status"] == ref["status"] == "ok", str((actual.get("error"),ref.get("error")))
                validate_times(actual["seconds"],manifest["repeats"],"Python")
                validate_times(ref["seconds"],manifest["repeats"],"R workflow")
                validate_times(ref["scored_seconds"],manifest["repeats"],"R scored")
                cross = cross_scores[(case["id"],backend)]
                assert cross["status"] == "ok", cross.get("error")
                row.update(compare_scores(actual["cpu_score"],cross["score"],"cpu"))
                row.update({"python_returned_"+key:value for key,value in
                            compare_scores(actual,cross["score"],backend).items()})
                row.update({"r_returned_"+key:value for key,value in
                            compare_scores(ref,ref["cpu_score"],rb).items()})
                # Check both output designs against the same structural/coordinate rules.
                validate_design(case,np.asarray(ref["design"]),evaluation(case),transported=True)
                if case["function"] == "soft":
                    assert ref["qmc_size"] == len(evaluation(case)), "incorrect R qmc_size"
                same = bool(np.allclose(actual["design"],ref["design"],rtol=0,atol=2e-14))
                a,b = actual["cpu_score"]["normalized_Q"],ref["cpu_score"]["normalized_Q"]
                quality = (a-b)/b if b else None
                row.update(design_equal=same,python_Q=actual["Q"],r_Q=ref["Q"],
                           python_normalized_Q=actual["normalized_Q"],r_normalized_Q=ref["normalized_Q"],
                           python_R_rescored_Q=cross["score"]["Q"],
                           normalized_quality_relative_difference=quality,
                           python_seconds=float(np.median(actual["seconds"])),
                           r_workflow_seconds=float(np.median(ref["seconds"])),
                           r_scored_seconds=float(np.median(ref["scored_seconds"])))
                row["r_over_python_scored_time"] = row["r_scored_seconds"]/row["python_seconds"]
                row["r_over_python_workflow_time"] = row["r_workflow_seconds"]/row["python_seconds"]
                row["validation_status"] = "passed"
                if case["initialization"] == "shared" and rb == backend:
                    compare_scores(actual["cpu_score"],ref["cpu_score"],backend)
                    row["agreement_status"] = "passed"
                    row["comparison"] = "same_design" if same else "equivalent_score_different_design"
                else:
                    row["comparison"] = "independent_initializers" if case["initialization"] == "native" else "different_backends"
        except Exception as error:
            row.update(status="failed",error=f"{type(error).__name__}: {error}")
            if row["validation_status"] == "passed":
                row.update(agreement_status="failed",comparison="final_score_divergence")
            else:
                row["validation_status"] = "failed"
        rows.append(row)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (output/"results.csv").open("w",newline="",encoding="utf-8") as stream:
        writer = csv.DictWriter(stream,fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)
    summary = dict(status="passed" if all(r["status"] == "passed" for r in rows) else "failed",
                   case_counts=case_counts(manifest["cases"]),executions=len(rows),
                   passed=sum(r["status"]=="passed" for r in rows),
                   failed=sum(r["status"]=="failed" for r in rows),
                   validation_passed=sum(r["validation_status"]=="passed" for r in rows),
                   validation_failed=sum(r["validation_status"]=="failed" for r in rows),
                   agreement_failed=sum(r["agreement_status"]=="failed" for r in rows),
                   comparisons=dict(Counter(r.get("comparison","invalid_or_failed") for r in rows)),
                   finished=datetime.now(timezone.utc).isoformat(),
                   analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   tolerances=dict(coordinate_atol=2e-14,score_rtol_cpu_cuda=1e-8,score_rtol_metal=3e-3),
                   results=rows)
    write_json(output/"summary.json",summary)
    return summary


def main() -> int:
    if not __debug__:
        raise RuntimeError("Do not use python -O or PYTHONOPTIMIZE: experiment checks require assertions")
    args, cases = parse_args()
    if args.list:
        print(json.dumps(dict(counts=case_counts(cases),cases=cases),indent=2))
        return 0
    for name in ["OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","VECLIB_MAXIMUM_THREADS","MKL_NUM_THREADS"]:
        os.environ[name] = str(args.threads)
    import softDOE
    if not shutil.which(args.rscript):
        raise RuntimeError("Rscript is missing; add R's bin directory to PATH or use --rscript")
    info = softDOE.backend_info(refresh=True)
    for backend in args.backends:
        if not info[backend]["available"]:
            raise RuntimeError(f"Requested Python {backend} is unavailable: {info[backend]['reason']}")
    if "cuda" in args.backends:
        import torch
        torch.set_num_threads(args.threads)
    output = (args.output or Path("experiment-results")/datetime.now().strftime("%Y%m%d-%H%M%S-%f")).resolve()
    output.mkdir(parents=True,exist_ok=False)
    metadata = dict(started=datetime.now(timezone.utc).isoformat(),command=sys.argv,
                    python=sys.version,platform=platform.platform(),machine=platform.machine(),
                    softDOE=softDOE.__version__,import_path=softDOE.__file__,backends=info,
                    packages={name:importlib.metadata.version(name) for name in ["numpy","scipy","pydoe"]},
                    threads=args.threads,source_sha256={})
    source = Path(softDOE.__file__).parent
    for path in sorted(source.iterdir()):
        if path.is_file():
            metadata["source_sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata["script_sha256"] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [Path(__file__),Path(__file__).with_name("experiment_cases.py"),Path(__file__).with_name("reference_experiments.R")]}
    write_json(output/"python-environment.json",metadata)
    manifest = dict(profile=args.profile,repeats=args.repeats,mofat_version=args.mofat_version,
                    r_backend=args.r_backend,backends=args.backends,
                    r_backends=args.backends if args.r_backend == "same" else ["cpu"],
                    cases=prepare_cases(cases))
    write_json(output/"manifest.json",manifest)
    print(f"{case_counts(cases)}; backends={args.backends}; output={output}",flush=True)
    run_r(args,output,"baseline")
    records = []
    (output/"python-designs.jsonl").touch()
    for index,case in enumerate(cases,1):
        for backend in args.backends:
            try:
                record = run_python_case(case,backend,args.repeats)
            except Exception as error:
                record = dict(status="error",error=f"{type(error).__name__}: {error}")
            record.update(id=case["id"],backend=backend)
            records.append(record)
            append_json(output/"python-results.jsonl",record)
            if record["status"] == "ok":
                append_json(output/"python-designs.jsonl",{k:record[k] for k in ["id","backend","design"]})
            print(f"Python {index}/{len(cases)} {case['id']} {backend}: {record['status']}",flush=True)
    run_r(args,output,"score")
    # Empty file is legitimate if a filtered run includes only invalid cases.
    (output/"r-python-scores.jsonl").touch(exist_ok=True)
    summary = summarize(output,manifest,records)
    print(f"{summary['status']}: {summary['passed']} passed, {summary['failed']} failed; {output/'results.csv'}")
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
