import shutil
import subprocess
import textwrap

import numpy as np
import pytest
from scipy.linalg import cholesky

from softDOE import _core
from softDOE._soft import _q_min_value
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


def test_q_matrix_and_q_value_match_q_fun_rcpp(tmp_path, capsys):
    _require_r_package("Rcpp")
    _require_r_package("RcppArmadillo")
    _require_r_package("SFDesign")
    _require_r_package("gtools")
    _require_r_package("purrr")

    script_path = tmp_path / "q_fun_reference.R"
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
            qmc <- qmc_generate(500, 2)

            design_ls <- list(A)

            a <- design_ls[[1]]
            a[, 1] <- ifelse(a[, 1] < 0.5, a[, 1] + delta, a[, 1] - delta)
            design_ls <- append(design_ls, list(a))

            old_X <- Reduce(rbind, design_ls)
            R_info <- R_fun_rcpp(NULL, old_X, NULL, 1, beta)
            old_R <- R_info$R

            MMSE_info <- Q_fun_rcpp(qmc, old_X, NULL, NULL, R_info$L, 1, beta)
            old_R_qmc <- MMSE_info$new_R
            old_mmse <- min(MMSE_info$val)

            q <- c(1, 2)
            temp <- lapply(1:2, function(j) {
              b <- design_ls[[q[j]]]
              b[, 2] <- ifelse(b[, 2] < 0.5, b[, 2] + delta, b[, 2] - delta)
              return(b)
            })

            opt <- lapply(1:2, function(j) {
              R_info <- R_fun_rcpp(old_R, old_X, temp[[j]], 1, beta)
              MMSE_info <- Q_fun_rcpp(qmc, old_X, old_R_qmc, temp[[j]], R_info$L, 1, beta)
              ans <- list(
                mmse = min(MMSE_info$val),
                old_R = R_info$R,
                old_R_qmc = MMSE_info$new_R
              )
              return(ans)
            })

            write.csv(qmc, "qmc.csv", row.names = FALSE)
            write.csv(old_X, "old_X.csv", row.names = FALSE)
            write.csv(old_R, "old_R.csv", row.names = FALSE)
            write.csv(temp[[1]], "temp_1.csv", row.names = FALSE)
            write.csv(temp[[2]], "temp_2.csv", row.names = FALSE)
            write.csv(old_R_qmc, "q_full.csv", row.names = FALSE)
            write.csv(matrix(old_mmse), "mmse_full.csv", row.names = FALSE)
            write.csv(opt[[1]]$old_R, "R_incremental_1.csv", row.names = FALSE)
            write.csv(opt[[2]]$old_R, "R_incremental_2.csv", row.names = FALSE)
            write.csv(opt[[1]]$old_R_qmc, "q_incremental_1.csv", row.names = FALSE)
            write.csv(opt[[2]]$old_R_qmc, "q_incremental_2.csv", row.names = FALSE)
            write.csv(matrix(opt[[1]]$mmse), "mmse_incremental_1.csv", row.names = FALSE)
            write.csv(matrix(opt[[2]]$mmse), "mmse_incremental_2.csv", row.names = FALSE)
            write.csv(matrix(beta), "beta.csv", row.names = FALSE)
            """
        ).replace("__LOCAL_SOFT_PRELUDE__", local_soft_prelude())
    )

    subprocess.run(["Rscript", str(script_path)], cwd=tmp_path, check=True)

    qmc = np.loadtxt(tmp_path / "qmc.csv", delimiter=",", skiprows=1)
    old_X = np.loadtxt(tmp_path / "old_X.csv", delimiter=",", skiprows=1)
    old_R = np.loadtxt(tmp_path / "old_R.csv", delimiter=",", skiprows=1)
    temp_1 = np.loadtxt(tmp_path / "temp_1.csv", delimiter=",", skiprows=1)
    temp_2 = np.loadtxt(tmp_path / "temp_2.csv", delimiter=",", skiprows=1)
    r_q_full = np.loadtxt(tmp_path / "q_full.csv", delimiter=",", skiprows=1)
    r_mmse_full = float(
        np.loadtxt(tmp_path / "mmse_full.csv", delimiter=",", skiprows=1)
    )
    r_incremental_1 = np.loadtxt(
        tmp_path / "q_incremental_1.csv", delimiter=",", skiprows=1
    )
    r_incremental_2 = np.loadtxt(
        tmp_path / "q_incremental_2.csv", delimiter=",", skiprows=1
    )
    r_mmse_incremental_1 = float(
        np.loadtxt(tmp_path / "mmse_incremental_1.csv", delimiter=",", skiprows=1)
    )
    r_mmse_incremental_2 = float(
        np.loadtxt(tmp_path / "mmse_incremental_2.csv", delimiter=",", skiprows=1)
    )
    beta = float(np.loadtxt(tmp_path / "beta.csv", delimiter=",", skiprows=1))

    py_q_full = _core.q_matrix(qmc, old_X, None, None, 1.0, beta)
    py_L_full = cholesky(
        old_R + np.eye(old_R.shape[0]) * 1e-6,
        lower=True,
        check_finite=False,
    )
    py_mmse_full = _q_min_value(py_q_full, py_L_full)

    py_R_incremental_1 = _core.corr_matrix(old_X, temp_1, old_R, 1.0, beta)
    py_R_incremental_2 = _core.corr_matrix(old_X, temp_2, old_R, 1.0, beta)
    py_q_incremental_1 = _core.q_matrix(qmc, old_X, py_q_full, temp_1, 1.0, beta)
    py_q_incremental_2 = _core.q_matrix(qmc, old_X, py_q_full, temp_2, 1.0, beta)
    py_L_incremental_1 = cholesky(
        py_R_incremental_1 + np.eye(py_R_incremental_1.shape[0]) * 1e-6,
        lower=True,
        check_finite=False,
    )
    py_L_incremental_2 = cholesky(
        py_R_incremental_2 + np.eye(py_R_incremental_2.shape[0]) * 1e-6,
        lower=True,
        check_finite=False,
    )
    py_mmse_incremental_1 = _q_min_value(py_q_incremental_1, py_L_incremental_1)
    py_mmse_incremental_2 = _q_min_value(py_q_incremental_2, py_L_incremental_2)

    with capsys.disabled():
        print("\nSetting 1 R Q_fun_rcpp new_R:")
        print(r_q_full)
        print("Setting 1 Python _core.q_matrix new_R:")
        print(py_q_full)
        print("Setting 1 R min(Q_fun_rcpp val):")
        print(r_mmse_full)
        print("Setting 1 Python _q_min_value:")
        print(py_mmse_full)
        print("\nSetting 2 R Q_fun_rcpp incremental new_R, candidate 1:")
        print(r_incremental_1)
        print("Setting 2 Python _core.q_matrix incremental new_R, candidate 1:")
        print(py_q_incremental_1)
        print("Setting 2 R mmse, candidate 1:")
        print(r_mmse_incremental_1)
        print("Setting 2 Python _q_min_value, candidate 1:")
        print(py_mmse_incremental_1)
        print("\nSetting 2 R Q_fun_rcpp incremental new_R, candidate 2:")
        print(r_incremental_2)
        print("Setting 2 Python _core.q_matrix incremental new_R, candidate 2:")
        print(py_q_incremental_2)
        print("Setting 2 R mmse, candidate 2:")
        print(r_mmse_incremental_2)
        print("Setting 2 Python _q_min_value, candidate 2:")
        print(py_mmse_incremental_2)

    np.testing.assert_allclose(py_q_full, r_q_full, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(py_mmse_full, r_mmse_full, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        py_q_incremental_1, r_incremental_1, rtol=1e-12, atol=1e-14
    )
    np.testing.assert_allclose(
        py_q_incremental_2, r_incremental_2, rtol=1e-12, atol=1e-14
    )
    np.testing.assert_allclose(
        py_mmse_incremental_1, r_mmse_incremental_1, rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(
        py_mmse_incremental_2, r_mmse_incremental_2, rtol=1e-10, atol=1e-12
    )
