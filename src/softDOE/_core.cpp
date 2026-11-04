#include <cmath>
#include <stdexcept>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include "_kernel.hpp"

namespace py = pybind11;

namespace {

using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

// Evaluate the MIM correlation kernel between row ia of A and row ib of B.
double corr_idx(const double* A, py::ssize_t ia, const double* B, py::ssize_t ib,
                py::ssize_t p, double alpha, double beta) {
    return softdoe::kernel_correlation(A + ia * p, B + ib * p,
                                       static_cast<int>(p), alpha, beta);
}

// Validate that a Python/NumPy input is a two-dimensional matrix.
void require_matrix(const py::buffer_info& info, const char* name) {
    if (info.ndim != 2) {
        throw std::invalid_argument(std::string(name) + " must be two-dimensional.");
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// Testing helpers
// ---------------------------------------------------------------------------

// Expose corr_idx to Python for parity tests against R's internal corr_idx_fast.
// This is intentionally private (_corr_idx) and is not part of the public API.
double corr_idx_binding(Array A, py::ssize_t ia, Array B, py::ssize_t ib,
                        double alpha, double beta) {
    const py::buffer_info a_info = A.request();
    const py::buffer_info b_info = B.request();
    require_matrix(a_info, "A");
    require_matrix(b_info, "B");

    if (a_info.shape[1] != b_info.shape[1]) {
        throw std::invalid_argument("A and B must have the same number of columns.");
    }
    if (ia < 0 || ia >= a_info.shape[0]) {
        throw std::out_of_range("ia is out of range for A.");
    }
    if (ib < 0 || ib >= b_info.shape[0]) {
        throw std::out_of_range("ib is out of range for B.");
    }

    const auto* a = static_cast<const double*>(a_info.ptr);
    const auto* b = static_cast<const double*>(b_info.ptr);
    return corr_idx(a, ia, b, ib, a_info.shape[1], alpha, beta);
}

// Build the MIM correlation matrix for X, or append new_X to a precomputed
// old_R block when incrementally growing a design.
Array corr_matrix(Array X, py::object new_X_obj, py::object old_R_obj,
                  double alpha, double beta) {
    const py::buffer_info x_info = X.request();
    require_matrix(x_info, "X");
    const auto n = x_info.shape[0];
    const auto p = x_info.shape[1];
    const auto* x = static_cast<const double*>(x_info.ptr);

    if (new_X_obj.is_none()) {
        Array out({n, n});
        auto out_info = out.mutable_unchecked<2>();
        for (py::ssize_t i = 0; i < n; ++i) {
            out_info(i, i) = 1.0;
            for (py::ssize_t j = i + 1; j < n; ++j) {
                const double c = corr_idx(x, i, x, j, p, alpha, beta);
                out_info(i, j) = c;
                out_info(j, i) = c;
            }
        }
        return out;
    }

    if (old_R_obj.is_none()) {
        throw std::invalid_argument("old_R must be provided when new_X is not None.");
    }

    Array new_X = py::cast<Array>(new_X_obj);
    Array old_R = py::cast<Array>(old_R_obj);
    const py::buffer_info x2_info = new_X.request();
    const py::buffer_info r0_info = old_R.request();
    require_matrix(x2_info, "new_X");
    require_matrix(r0_info, "old_R");

    const auto k = x2_info.shape[0];
    if (x2_info.shape[1] != p) {
        throw std::invalid_argument("new_X must have the same number of columns as X.");
    }
    if (r0_info.shape[0] != n || r0_info.shape[1] != n) {
        throw std::invalid_argument("old_R must have shape (X rows, X rows).");
    }

    const auto* x2 = static_cast<const double*>(x2_info.ptr);
    const auto* r0 = static_cast<const double*>(r0_info.ptr);
    const auto new_n = n + k;
    Array out({new_n, new_n});
    auto out_info = out.mutable_unchecked<2>();

    for (py::ssize_t i = 0; i < new_n; ++i) {
        for (py::ssize_t j = 0; j < new_n; ++j) {
            out_info(i, j) = 0.0;
        }
        out_info(i, i) = 1.0;
    }

    for (py::ssize_t i = 0; i < n; ++i) {
        for (py::ssize_t j = 0; j < n; ++j) {
            out_info(i, j) = r0[i * n + j];
        }
    }

    for (py::ssize_t i = 0; i < k; ++i) {
        const py::ssize_t ii = n + i;
        for (py::ssize_t j = 0; j < n; ++j) {
            const double c = corr_idx(x2, i, x, j, p, alpha, beta);
            out_info(ii, j) = c;
            out_info(j, ii) = c;
        }
    }

    for (py::ssize_t i = 0; i < k; ++i) {
        const py::ssize_t ii = n + i;
        for (py::ssize_t j = i + 1; j < k; ++j) {
            const py::ssize_t jj = n + j;
            const double c = corr_idx(x2, i, x2, j, p, alpha, beta);
            out_info(ii, jj) = c;
            out_info(jj, ii) = c;
        }
    }

    return out;
}

// Build correlations between QMC/evaluation points and design points X, or
// append rows for new_X to a precomputed design-vs-QMC correlation block.
Array q_matrix(Array qmc, Array X, py::object old_R_obj, py::object new_X_obj,
               double alpha, double beta) {
    const py::buffer_info q_info = qmc.request();
    const py::buffer_info x_info = X.request();
    require_matrix(q_info, "qmc");
    require_matrix(x_info, "X");

    const auto m = q_info.shape[0];
    const auto p = q_info.shape[1];
    const auto n = x_info.shape[0];
    if (x_info.shape[1] != p) {
        throw std::invalid_argument("qmc and X must have the same number of columns.");
    }

    const auto* q = static_cast<const double*>(q_info.ptr);
    const auto* x = static_cast<const double*>(x_info.ptr);

    if (new_X_obj.is_none()) {
        Array out({n, m});
        auto out_info = out.mutable_unchecked<2>();
        for (py::ssize_t j = 0; j < m; ++j) {
            for (py::ssize_t i = 0; i < n; ++i) {
                out_info(i, j) = corr_idx(q, j, x, i, p, alpha, beta);
            }
        }
        return out;
    }

    if (old_R_obj.is_none()) {
        throw std::invalid_argument("old_R must be provided when new_X is not None.");
    }

    Array old_R = py::cast<Array>(old_R_obj);
    Array new_X = py::cast<Array>(new_X_obj);
    const py::buffer_info r0_info = old_R.request();
    const py::buffer_info x2_info = new_X.request();
    require_matrix(r0_info, "old_R");
    require_matrix(x2_info, "new_X");

    const auto k = x2_info.shape[0];
    if (x2_info.shape[1] != p) {
        throw std::invalid_argument("new_X must have the same number of columns as X.");
    }
    if (r0_info.shape[0] != n || r0_info.shape[1] != m) {
        throw std::invalid_argument("old_R must have shape (X rows, qmc rows).");
    }

    const auto* r0 = static_cast<const double*>(r0_info.ptr);
    const auto* x2 = static_cast<const double*>(x2_info.ptr);
    Array out({n + k, m});
    auto out_info = out.mutable_unchecked<2>();

    for (py::ssize_t i = 0; i < n; ++i) {
        for (py::ssize_t j = 0; j < m; ++j) {
            out_info(i, j) = r0[i * m + j];
        }
    }

    for (py::ssize_t j = 0; j < m; ++j) {
        for (py::ssize_t i = 0; i < k; ++i) {
            out_info(n + i, j) = corr_idx(q, j, x2, i, p, alpha, beta);
        }
    }

    return out;
}

// Log correlations for the Python incremental evaluator when products underflow.
Array log_q_matrix(Array qmc, Array X, double alpha, double beta) {
    const auto q_info = qmc.request();
    const auto x_info = X.request();
    require_matrix(q_info, "qmc");
    require_matrix(x_info, "X");
    if (q_info.shape[1] != x_info.shape[1])
        throw std::invalid_argument("qmc and X must have the same number of columns.");
    const auto m = q_info.shape[0], n = x_info.shape[0], p = x_info.shape[1];
    const auto* q = static_cast<const double*>(q_info.ptr);
    const auto* x = static_cast<const double*>(x_info.ptr);
    Array out({n, m});
    auto values = out.mutable_unchecked<2>();
    for (py::ssize_t j = 0; j < m; ++j)
        for (py::ssize_t i = 0; i < n; ++i)
            values(i, j) = softdoe::kernel_log_correlation(
                x + i * p, q + j * p, static_cast<int>(p), alpha, beta);
    return out;
}

// Register the compiled backend functions exposed to the Python module.
PYBIND11_MODULE(_core, m) {
    m.doc() = "Compiled correlation kernels for softDOE.";
    m.def("_corr_idx", &corr_idx_binding, py::arg("A"), py::arg("ia"),
          py::arg("B"), py::arg("ib"), py::arg("alpha") = 1.0,
          py::arg("beta") = 1.0,
          "Evaluate the MIM kernel between row ia of matrix A and row ib of matrix B.");
    m.def("corr_matrix", &corr_matrix, py::arg("X"), py::arg("new_X") = py::none(),
          py::arg("old_R") = py::none(), py::arg("alpha") = 1.0,
          py::arg("beta") = 1.0);
    m.def("q_matrix", &q_matrix, py::arg("qmc"), py::arg("X"),
          py::arg("old_R") = py::none(), py::arg("new_X") = py::none(),
          py::arg("alpha") = 1.0, py::arg("beta") = 1.0);
    m.def("_log_q_matrix", &log_q_matrix, py::arg("qmc"), py::arg("X"),
          py::arg("alpha") = 1.0, py::arg("beta") = 1.0);
}
