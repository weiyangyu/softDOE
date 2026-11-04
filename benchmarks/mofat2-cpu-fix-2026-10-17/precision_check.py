"""Recompute first-divergent scores independently at 80 decimal digits.

Requires the current softDOE checkout and mpmath. Run from any directory:
    python benchmarks/mofat2-cpu-fix-2026-10-17/precision_check.py
Prints results only; does not change saved evidence or reference data.
"""

import json
from pathlib import Path
import mpmath as mp
import softDOE

mp.mp.dps = 80
root = Path(__file__).resolve().parent
reports = {}
inputs = (
    root.parent / "mofat2-mac-2026-10-17/0066-diagnostic.json",
    root.parent / "mofat2-mac-2026-10-17/0069-diagnostic.json",
    root / "0118-before-diagnostic.json",
)
for path in inputs:
    filename = str(path.relative_to(root.parent))
    data = json.loads(path.read_text())
    c = data["case"]
    idx = data["first_decision_divergence"]["trace_index"]
    t = data["python"]["trace"]
    initial = data["python"]["initial"]
    inc = initial["design"]
    for step in t[:idx]:
        if step["accepted"]:
            inc = step["design"]
    cand = t[idx]["design"]
    points = softDOE.qmc_generate(c["m"], c["p"])
    beta = mp.mpf(1) / (2 * c["l"]) ** 2

    def score(d):
        x = [[mp.mpf(float(v)) for v in row] for row in d]
        n = len(x)
        q = [[mp.mpf(float(v)) for v in row] for row in points]

        def k(a, b):
            return mp.fprod(1 / (1 + (u - v) ** 2 / beta) for u, v in zip(a, b))

        R = mp.matrix(
            [
                [
                    k(a, b) + (mp.mpf(float(1e-6)) if i == j else 0)
                    for j, b in enumerate(x)
                ]
                for i, a in enumerate(x)
            ]
        )
        L = mp.cholesky(R)
        logs = []
        for pt in q:
            rhs = [k(a, pt) for a in x]
            y = []
            for i in range(n):
                y.append((rhs[i] - mp.fsum(L[i, j] * y[j] for j in range(i))) / L[i, i])
            logs.append(mp.log(mp.fsum(v * v for v in y)))
        i = min(range(len(logs)), key=logs.__getitem__)
        return logs[i], i

    a, ia = score(inc)
    b, ib = score(cand)
    reports[c["id"]] = {
        "source": filename,
        "step": t[idx]["step"],
        "precision_digits": 80,
        "incumbent_argmin": ia,
        "candidate_argmin": ib,
        "incumbent_log_Q": str(a),
        "candidate_log_Q": str(b),
        "delta": str(b - a),
        "python_delta": data["first_decision_divergence"]["python_delta"],
        "r_delta": data["first_decision_divergence"]["r_delta"],
    }
    print(c["id"], reports[c["id"]], flush=True)
print(json.dumps(reports, indent=2))
