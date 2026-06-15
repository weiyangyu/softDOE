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


def test_corr_idx_matches_r_corr_idx_fast(tmp_path, capsys):
    _require_r_package("Rcpp")
    _require_r_package("RcppArmadillo")
    _require_r_package("SFDesign")
    _require_r_package("gtools")
    _require_r_package("purrr")

    script_path = tmp_path / "corr_idx_reference.R"
    script_path.write_text(
        textwrap.dedent(
            """
            __LOCAL_SOFT_PRELUDE__

            cppFunction(
              depends = "RcppArmadillo",
              code = '
              #include <RcppArmadillo.h>
              using namespace Rcpp;

              inline double corr_idx_fast(const arma::mat& A, arma::uword ia,
                                          const arma::mat& B, arma::uword ib,
                                          const double alpha, const double beta) {
                const arma::uword p = A.n_cols;
                const double invbeta = 1.0 / beta;

                if (alpha == 1.0) {
                  double c = 1.0;
                  for (arma::uword k = 0; k < p; ++k) {
                    const double d = A(ia, k) - B(ib, k);
                    c *= 1.0 / (1.0 + (d * d) * invbeta);
                  }
                  return c;
                }

                const double negalpha = -alpha;
                double logc = 0.0;
                for (arma::uword k = 0; k < p; ++k) {
                  const double d = A(ia, k) - B(ib, k);
                  logc += negalpha * std::log1p((d * d) * invbeta);
                }
                return std::exp(logc);
              }

              // [[Rcpp::export]]
              NumericVector corr_idx_many(const arma::mat& D,
                                          const arma::uword ia,
                                          const IntegerVector ib,
                                          const double alpha,
                                          const double beta) {
                const int n = ib.size();
                NumericVector out(n);
                for (int i = 0; i < n; ++i) {
                  out[i] = corr_idx_fast(D, ia, D, ib[i], alpha, beta);
                }
                return out;
              }
              '
            )

            set.seed(1)
            D <- soft(2, 4)

            setting1 <- corr_idx_many(D, 0, 1:11, 1, 1/64)
            setting2 <- corr_idx_many(D, 2, 0:11, 0.79, 0.41)

            write.csv(D, "D.csv", row.names = FALSE)
            write.csv(setting1, "setting1.csv", row.names = FALSE)
            write.csv(setting2, "setting2.csv", row.names = FALSE)
            """
        ).replace("__LOCAL_SOFT_PRELUDE__", local_soft_prelude())
    )

    subprocess.run(["Rscript", str(script_path)], cwd=tmp_path, check=True)

    design = np.loadtxt(tmp_path / "D.csv", delimiter=",", skiprows=1)
    r_setting1 = np.loadtxt(tmp_path / "setting1.csv", delimiter=",", skiprows=1)
    r_setting2 = np.loadtxt(tmp_path / "setting2.csv", delimiter=",", skiprows=1)

    py_setting1 = np.array(
        [_core._corr_idx(design, 0, design, j, 1.0, 1.0 / 64.0) for j in range(1, 12)]
    )
    py_setting2 = np.array(
        [_core._corr_idx(design, 2, design, j, 0.79, 0.41) for j in range(12)]
    )

    with capsys.disabled():
        print("\nSetting 1 R corr_idx_fast:")
        print(r_setting1)
        print("Setting 1 Python _core._corr_idx:")
        print(py_setting1)
        print("\nSetting 2 R corr_idx_fast:")
        print(r_setting2)
        print("Setting 2 Python _core._corr_idx:")
        print(py_setting2)

    np.testing.assert_allclose(py_setting1, r_setting1, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(py_setting2, r_setting2, rtol=1e-12, atol=1e-14)
