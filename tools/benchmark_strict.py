"""Benchmark the installed package and export shared inputs for the R check.

Run from another directory to measure an installed wheel/source distribution:
python /path/to/softDOE/tools/benchmark_strict.py --backend metal --output /tmp/softdoe-bench
"""
import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np
import softDOE
from softDOE._backend import _run_with_fallback
from softDOE._soft import _prepare, _optimize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["cpu", "metal", "cuda"], default="metal")
    parser.add_argument("--p", type=int, default=20)
    parser.add_argument("--l", type=int, default=6)
    parser.add_argument("--m", type=int, default=500)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    # Separate discovery/compilation and PyDOE preparation from optimizer timing.
    print(softDOE.backend_info(), flush=True)
    started = time.perf_counter()
    prepared = _prepare(args.p, args.l, args.m, args.seed, 2 * args.l)
    preparation = time.perf_counter() - started
    times = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        result, backend, device, reason = _run_with_fallback(
            args.backend, lambda ops: _optimize(prepared, "strict", ops))
        elapsed = time.perf_counter() - started
        if backend != args.backend:
            raise RuntimeError(f"Requested {args.backend}, used {backend}: {reason}")
        times.append(elapsed)
        print(f"{backend}: {elapsed:.3f} s; log(Q)={result[1]:.12g}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output / "A.csv", prepared["A"], delimiter=",")
    np.savetxt(args.output / "qmc.csv", prepared["qmc"], delimiter=",")
    np.savetxt(args.output / "design.csv", result[0], delimiter=",")
    metadata = dict(p=args.p, l=args.l, m=args.m, seed=args.seed,
                    version=softDOE.__version__, package_path=softDOE.__file__,
                    backend=backend, device=device, seconds=times,
                    median_seconds=statistics.median(times),
                    preparation_seconds=preparation, log_Q=result[1])
    (args.output / "timing.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
