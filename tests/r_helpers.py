from pathlib import Path


def project_root():
    return Path(__file__).resolve().parents[1]


def local_soft_prelude():
    root = project_root()
    return "\n".join(
        [
            "library(Rcpp)",
            f'sourceCpp("{root / "R" / "Q.cpp"}")',
            f'source("{root / "R" / "soft_measure_lib.R"}")',
            f'source("{root / "R" / "soft.R"}")',
        ]
    )


def local_measure_prelude():
    root = project_root()
    return "\n".join(
        [
            local_soft_prelude(),
            f'source("{root / "R" / "measure.R"}")',
        ]
    )
