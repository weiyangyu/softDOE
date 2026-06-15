import shutil
import subprocess
import textwrap

import numpy as np
import pytest

from softDOE import measure
from r_helpers import local_measure_prelude


def _require_r_package(package):
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript is not available.")

    check = subprocess.run(
        [
            "Rscript",
            "-e",
            f"quit(status = ifelse(requireNamespace('{package}', quietly = TRUE), 0, 1))",
        ],
        capture_output=True,
        text=True,
    )
    if check.returncode != 0:
        pytest.skip(f"R package {package} is not available.")


def _read_vector(path):
    return np.loadtxt(path, delimiter=",", skiprows=1)


@pytest.mark.parametrize("structure", ["standard", "strict"])
def test_measure_matches_r_measure_on_same_design(tmp_path, structure, capsys):
    _require_r_package("Rcpp")
    _require_r_package("RcppArmadillo")
    _require_r_package("SFDesign")
    _require_r_package("gtools")
    _require_r_package("purrr")

    script_path = tmp_path / "measure_reference.R"
    script_path.write_text(
        textwrap.dedent(
            """
            __LOCAL_MEASURE_PRELUDE__

            fun <- function(X) {
              Y <- 10 * sin(pi * X[1] * X[2]) +
                20 * (X[3] - 0.5)^2 +
                10 * X[4] +
                5 * X[5]
              return(Y)
            }

            set.seed(1)
            D <- soft(p = 10, l = 4, structure = "__STRUCTURE__")
            y <- apply(D, 1, fun)
            result <- measure(D, y)

            write.csv(D, "design.csv", row.names = FALSE)
            write.csv(y, "response.csv", row.names = FALSE)
            write.csv(result$mustar, "mustar.csv", row.names = FALSE)
            write.csv(result$t, "t.csv", row.names = FALSE)
            """
        )
        .replace("__LOCAL_MEASURE_PRELUDE__", local_measure_prelude())
        .replace("__STRUCTURE__", structure)
    )

    subprocess.run(["Rscript", str(script_path)], cwd=tmp_path, check=True)

    design = np.loadtxt(tmp_path / "design.csv", delimiter=",", skiprows=1)
    response = _read_vector(tmp_path / "response.csv")
    r_mustar = _read_vector(tmp_path / "mustar.csv")
    r_t = _read_vector(tmp_path / "t.csv")

    py_result = measure(design, response)

    with capsys.disabled():
        print(f"\n{structure} R measure mustar:")
        print(r_mustar)
        print(f"{structure} Python measure mustar:")
        print(py_result["mustar"])
        print(f"\n{structure} R measure t:")
        print(r_t)
        print(f"{structure} Python measure t:")
        print(py_result["t"])

    np.testing.assert_allclose(py_result["mustar"], r_mustar, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(py_result["t"], r_t, rtol=1e-12, atol=1e-12)
