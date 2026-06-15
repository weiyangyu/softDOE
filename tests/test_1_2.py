import shutil
import subprocess
import textwrap

import numpy as np
import pytest

from softDOE import _core
from r_helpers import local_soft_prelude


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


def test_corr_matrix_matches_r_fun_rcpp(tmp_path, capsys):
    _require_r_package("Rcpp")
    _require_r_package("RcppArmadillo")
    _require_r_package("SFDesign")
    _require_r_package("gtools")
    _require_r_package("purrr")

    script_path = tmp_path / "r_fun_reference.R"
    script_path.write_text(
        textwrap.dedent(
            """
            __LOCAL_SOFT_PRELUDE__

            set.seed(1)
            D <- soft(2, 4)
            A <- D[1:4, ]
            l <- 4
            delta <- D[5, 1] - D[1, 1]
            beta <- 1 / (2 * l)^2

            design_ls <- list(A)

            a <- design_ls[[1]]
            a[, 1] <- ifelse(a[, 1] < 0.5, a[, 1] + delta, a[, 1] - delta)
            design_ls <- append(design_ls, list(a))

            old_X <- Reduce(rbind, design_ls)
            R_info <- R_fun_rcpp(NULL, old_X, NULL, 1, beta)

            q <- c(1, 2)
            temp <- lapply(1:2, function(j) {
              b <- design_ls[[q[j]]]
              b[, 2] <- ifelse(b[, 2] < 0.5, b[, 2] + delta, b[, 2] - delta)
              return(b)
            })

            opt <- lapply(1:2, function(j) {
              R_fun_rcpp(R_info$R, old_X, temp[[j]], 1, beta)
            })

            write.csv(old_X, "old_X.csv", row.names = FALSE)
            write.csv(temp[[1]], "temp_1.csv", row.names = FALSE)
            write.csv(temp[[2]], "temp_2.csv", row.names = FALSE)
            write.csv(R_info$R, "R_full.csv", row.names = FALSE)
            write.csv(opt[[1]]$R, "R_incremental_1.csv", row.names = FALSE)
            write.csv(opt[[2]]$R, "R_incremental_2.csv", row.names = FALSE)
            write.csv(matrix(beta), "beta.csv", row.names = FALSE)
            """
        ).replace("__LOCAL_SOFT_PRELUDE__", local_soft_prelude())
    )

    subprocess.run(
        ["Rscript", str(script_path)],
        cwd=tmp_path,
        check=True,
    )

    old_X = np.loadtxt(tmp_path / "old_X.csv", delimiter=",", skiprows=1)
    temp_1 = np.loadtxt(tmp_path / "temp_1.csv", delimiter=",", skiprows=1)
    temp_2 = np.loadtxt(tmp_path / "temp_2.csv", delimiter=",", skiprows=1)
    r_full = np.loadtxt(tmp_path / "R_full.csv", delimiter=",", skiprows=1)
    r_incremental_1 = np.loadtxt(
        tmp_path / "R_incremental_1.csv", delimiter=",", skiprows=1
    )
    r_incremental_2 = np.loadtxt(
        tmp_path / "R_incremental_2.csv", delimiter=",", skiprows=1
    )
    beta = float(np.loadtxt(tmp_path / "beta.csv", delimiter=",", skiprows=1))

    py_full = _core.corr_matrix(old_X, None, None, 1.0, beta)
    py_incremental_1 = _core.corr_matrix(old_X, temp_1, py_full, 1.0, beta)
    py_incremental_2 = _core.corr_matrix(old_X, temp_2, py_full, 1.0, beta)

    with capsys.disabled():
        print("\nSetting 1 R R_fun_rcpp full R:")
        print(r_full)
        print("Setting 1 Python _core.corr_matrix full R:")
        print(py_full)
        print("\nSetting 2 R R_fun_rcpp incremental R, candidate 1:")
        print(r_incremental_1)
        print("Setting 2 Python _core.corr_matrix incremental R, candidate 1:")
        print(py_incremental_1)
        print("\nSetting 2 R R_fun_rcpp incremental R, candidate 2:")
        print(r_incremental_2)
        print("Setting 2 Python _core.corr_matrix incremental R, candidate 2:")
        print(py_incremental_2)

    np.testing.assert_allclose(py_full, r_full, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(
        py_incremental_1, r_incremental_1, rtol=1e-12, atol=1e-14
    )
    np.testing.assert_allclose(
        py_incremental_2, r_incremental_2, rtol=1e-12, atol=1e-14
    )
