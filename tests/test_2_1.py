import shutil
import subprocess
import textwrap
from pathlib import Path

import numpy as np
import pytest

from softDOE._soft import _a_fun


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


def test_a_fun_matches_r_a_fun(tmp_path, capsys):
    _require_r_package("SFDesign")
    _require_r_package("gtools")

    py_lhd = _a_fun(2, 4, seed=1)
    py_perm = _a_fun(30, 4, seed=1)

    np.savetxt(tmp_path / "py_lhd_A.csv", py_lhd["A"], delimiter=",")

    project_root = Path(__file__).resolve().parents[1]
    soft_lib = project_root / "R" / "soft_measure_lib.R"

    script_path = tmp_path / "a_fun_reference.R"
    script_path.write_text(
        textwrap.dedent(
            """
            source("__SOFT_LIB__")

            set.seed(1)
            r_lhd <- A_fun(2, 4)
            r_perm <- A_fun(30, 4)

            py_lhd_A <- as.matrix(read.csv("py_lhd_A.csv", header = FALSE))

            r_lhd_crit <- SFDesign::maximin.crit(r_lhd$A)
            py_lhd_crit <- SFDesign::maximin.crit(py_lhd_A)

            write.csv(r_lhd$A, "r_lhd_A.csv", row.names = FALSE)
            write.csv(matrix(r_lhd$L), "r_lhd_L.csv", row.names = FALSE)
            write.csv(matrix(ifelse(is.null(r_lhd$quotient), NA, r_lhd$quotient)), "r_lhd_quotient.csv", row.names = FALSE)
            write.csv(matrix(ifelse(is.null(r_lhd$mod), NA, r_lhd$mod)), "r_lhd_mod.csv", row.names = FALSE)
            write.csv(matrix(r_lhd_crit), "r_lhd_crit.csv", row.names = FALSE)
            write.csv(matrix(py_lhd_crit), "py_lhd_crit.csv", row.names = FALSE)

            write.csv(r_perm$A, "r_perm_A.csv", row.names = FALSE)
            write.csv(matrix(r_perm$L), "r_perm_L.csv", row.names = FALSE)
            write.csv(matrix(r_perm$quotient), "r_perm_quotient.csv", row.names = FALSE)
            write.csv(matrix(r_perm$mod), "r_perm_mod.csv", row.names = FALSE)
            """
        ).replace("__SOFT_LIB__", str(soft_lib))
    )

    subprocess.run(["Rscript", str(script_path)], cwd=tmp_path, check=True)

    r_lhd_A = np.loadtxt(tmp_path / "r_lhd_A.csv", delimiter=",", skiprows=1)
    r_lhd_L = int(np.loadtxt(tmp_path / "r_lhd_L.csv", delimiter=",", skiprows=1))
    r_lhd_quotient = np.genfromtxt(
        tmp_path / "r_lhd_quotient.csv", delimiter=",", skip_header=1
    )
    r_lhd_mod = np.genfromtxt(tmp_path / "r_lhd_mod.csv", delimiter=",", skip_header=1)
    r_lhd_crit = float(
        np.loadtxt(tmp_path / "r_lhd_crit.csv", delimiter=",", skiprows=1)
    )
    py_lhd_crit = float(
        np.loadtxt(tmp_path / "py_lhd_crit.csv", delimiter=",", skiprows=1)
    )

    r_perm_A = np.loadtxt(tmp_path / "r_perm_A.csv", delimiter=",", skiprows=1)
    r_perm_L = int(np.loadtxt(tmp_path / "r_perm_L.csv", delimiter=",", skiprows=1))
    r_perm_quotient = int(
        np.loadtxt(tmp_path / "r_perm_quotient.csv", delimiter=",", skiprows=1)
    )
    r_perm_mod = int(np.loadtxt(tmp_path / "r_perm_mod.csv", delimiter=",", skiprows=1))

    with capsys.disabled():
        print("\nSetting 1 R A_fun(2, 4)$A:")
        print(r_lhd_A)
        print("Setting 1 Python _a_fun(2, 4)['A']:")
        print(py_lhd["A"])
        print("Setting 1 R maximin.crit:")
        print(r_lhd_crit)
        print("Setting 1 Python maximin.crit:")
        print(py_lhd_crit)
        print("\nSetting 2 R A_fun(30, 4)$A:")
        print(r_perm_A)
        print("Setting 2 Python _a_fun(30, 4)['A']:")
        print(py_perm["A"])

    assert py_lhd["A"].shape == r_lhd_A.shape == (4, 2)
    assert py_lhd["L"] == r_lhd_L == 24
    assert py_lhd["quotient"] is None
    assert py_lhd["mod"] is None
    assert np.isnan(r_lhd_quotient)
    assert np.isnan(r_lhd_mod)
    assert np.all(py_lhd["A"] >= 0.5)
    assert np.all(py_lhd["A"] <= 4.5)
    assert py_lhd_crit >= r_lhd_crit

    np.testing.assert_allclose(py_perm["A"], r_perm_A, rtol=0.0, atol=0.0)
    assert py_perm["L"] == r_perm_L == 24
    assert py_perm["quotient"] == r_perm_quotient == 1
    assert py_perm["mod"] == r_perm_mod == 6
