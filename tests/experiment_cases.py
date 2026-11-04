"""Portable, deterministic SOFT/A-SOFT experiment specifications (no execution)."""
from itertools import product


def experiment_cases(profile: str = "full") -> list:
    """Return explicit cases; smoke is a setup check, stress adds costly sizes."""
    if profile not in {"smoke", "full", "stress"}:
        raise ValueError("profile must be smoke, full, or stress")
    cases = []

    def add(function="soft", p=3, l=4, structure="standard", m=64,
            seed=1, n_aug=4, N=128, supplied=None, initialization="shared",
            tag="grid", invalid=None):
        case = dict(function=function, p=p, l=l, structure=structure, m=m,
                    seed=seed, initialization=initialization, tag=tag)
        if function == "a_soft":
            case.update(n_aug=n_aug, N=N, supplied=supplied)
        if invalid:
            case.update(invalid)
            case["expect_error"] = True
        case["id"] = f"{len(cases):04d}_{function}_{tag}"
        cases.append(case)

    dimensions = [(2, 2), (3, 4)] if profile == "smoke" else list(
        product([2, 3, 5, 10, 11, 20], [2, 4, 6]))
    seeds = [1] if profile == "smoke" else [1, 42]
    for (p, l), structure, seed in product(dimensions, ["standard", "strict"], seeds):
        add(p=p, l=l, structure=structure, seed=seed)
        for n_aug in ([1] if profile == "smoke" else [1, 4, 12]):
            add("a_soft", p, l, structure, seed=seed, n_aug=n_aug)
    supplied_dimensions = [(2, 2)] if profile == "smoke" else list(
        product([2, 5, 11, 20], [2, 4, 6]))
    for (p, l), supplied, n_aug in product(
            supplied_dimensions, ["ofat", "boundary", "duplicate"],
            [1] if profile == "smoke" else [1, 12]):
        add("a_soft", p, l, n_aug=n_aug, supplied=supplied, tag="supplied")

    for structure in ["standard", "strict"]:
        add(p=2, l=2, structure=structure, m=1, tag="minimum")
        for count in [1, 8]:
            add("a_soft", 2, 2, structure, m=1, n_aug=count, N=count,
                tag="exhaust_pool")
    if profile != "smoke":
        # l! boundary: below, equal to, and above 4!; l=2 branches are in grid.
        for p, structure in product([23, 24, 25], ["standard", "strict"]):
            add(p=p, l=4, structure=structure, m=32, tag="factorial")
            add("a_soft", p, 4, structure, m=32, n_aug=3, N=33, tag="factorial")
        for m, structure in product([1, 17, 500], ["standard", "strict"]):
            add(p=5, l=4, structure=structure, m=m, tag="evaluation_size")
            add("a_soft", 5, 4, structure, m=m, n_aug=5, N=17,
                tag="evaluation_size")
        # Public default workloads and genuine initializer-inclusive timing.
        for p, l, structure, initialization in product(
                [5, 20], [4, 6], ["standard", "strict"], ["shared", "native"]):
            add(p=p, l=l, structure=structure, m=500,
                initialization=initialization, tag="default_workload")
            add("a_soft", p, l, structure, m=500, n_aug=10, N=10000,
                initialization=initialization, tag="default_workload")
        for structure in ["standard", "strict"]:
            add(p=3, l=8, structure=structure, tag="larger_l")
            add("a_soft", 3, 8, structure, n_aug=24, N=257, tag="larger_l")
        add(structure="invalid", tag="structure_fallback")
        add("a_soft", structure="invalid", tag="structure_fallback")
        # Supplied input must override otherwise invalid generation arguments.
        add("a_soft", supplied="ofat", structure="invalid", m=0,
            tag="supplied_override")
    if profile == "stress":
        for p, structure in product([50, 100, 256, 1024], ["standard", "strict"]):
            add(p=p, l=2, structure=structure, m=32, tag="stress")
            add("a_soft", p, 2, structure, m=32, n_aug=16, N=512, tag="stress")

    invalids = [
        ("soft", {"p": 1}), ("soft", {"l": 3}),
        ("soft", {"l": 0}), ("soft", {"m": 0}),
        ("soft", {"p": 2.5}), ("soft", {"m": 1.5}),
        ("a_soft", {"n_aug": 0}), ("a_soft", {"n_aug": -1}),
        ("a_soft", {"n_aug": 129, "N": 128}),
        ("a_soft", {"N": 0}), ("a_soft", {"n_aug": 1.5}),
        ("a_soft", {"l": 3}), ("a_soft", {"p": 1}),
        ("a_soft", {"supplied": "bad_rows"}),
        ("a_soft", {"supplied": "out_of_bounds"}),
    ]
    for function, invalid in invalids:
        add(function, invalid=invalid, tag="invalid")
    return cases


def case_counts(cases: list) -> dict:
    return {"total": len(cases),
            "valid": sum(not c.get("expect_error", False) for c in cases),
            "invalid": sum(c.get("expect_error", False) for c in cases),
            "soft": sum(c["function"] == "soft" and not c.get("expect_error") for c in cases),
            "a_soft": sum(c["function"] == "a_soft" and not c.get("expect_error") for c in cases),
            "supplied": sum(bool(c.get("supplied")) and not c.get("expect_error") for c in cases)}
