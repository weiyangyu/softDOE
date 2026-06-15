# Test Plan

## 1. Test `_core.cpp` Against `Q.cpp`

The goal of this group of tests is to check whether the compiled Python
implementation in `src/softDOE/_core.cpp` produces the same numerical results
as the original R/Rcpp implementation in `R/Q.cpp`.

### 1.1 Test `corr_idx_fast`

This test checks whether the Python implementation of the MIM kernel matches
`corr_idx_fast()` from `Q.cpp`.

1. Generate a design matrix in R:

   ```r
   Rcpp::sourceCpp("R/Q.cpp")
   source("R/soft_measure_lib.R")
   source("R/soft.R")
   D <- soft(2, 4)
   ```

2. Compute the following reference values in R:

   - Setting 1:

     ```r
     corr_idx_fast(D, 1, D, 2, 1, 1/64), ..., corr_idx_fast(D, 1, D, 12, 1, 1/64)
     ```

   - Setting 2:

     ```r
     corr_idx_fast(D, 3, D, 1, 0.79, 0.41), ..., corr_idx_fast(D, 3, D, 12, 0.79, 0.41)
     ```

3. Compute the same values in Python using `_core._corr_idx()`.

4. Compare the R and Python results with a tight numerical tolerance.

Notes:

- The settings above use R's 1-based row numbers.
- The Python test converts those row numbers to Python/C++ 0-based indices.
- Setting 1 tests the `alpha == 1` branch.
- Setting 2 tests the general `alpha != 1` branch.
- The corresponding pytest file is `test_1_1.py`.


### 1.2 Test `R_fun_rcpp`

This test checks whether Python's `_core.corr_matrix()` matches
`R_fun_rcpp()` from `Q.cpp`.

The test should cover both ways that `R_fun_rcpp()` is used:

- building a full correlation matrix from scratch;
- incrementally appending new design points to an existing correlation matrix.

1. Generate the same base design in R:

   ```r
   Rcpp::sourceCpp("R/Q.cpp")
   source("R/soft_measure_lib.R")
   source("R/soft.R")
   D <- soft(2, 4)
   A <- D[1:4, ]
   l <- 4
   delta <- D[5, 1] - D[1, 1]
   beta <- 1 / (2 * l)^2
   ```

2. Build the first two OFAT blocks:

   ```r
   design_ls <- list(A)

   a <- design_ls[[1]]
   a[, 1] <- ifelse(a[, 1] < 0.5, a[, 1] + delta, a[, 1] - delta)
   design_ls <- append(design_ls, list(a))

   old_X <- Reduce(rbind, design_ls)
   ```

3. Setting 1: test the full matrix build.

   In R:

   ```r
   R_info <- R_fun_rcpp(NULL, old_X, NULL, 1, beta)
   ```

   In Python:

   ```python
   py_R = _core.corr_matrix(old_X, None, None, 1.0, beta)
   ```

   Compare `py_R` with `R_info$R`.

4. Setting 2: test the incremental matrix build.

   In R, create two candidate `new_X` matrices by shifting the second factor:

   ```r
   q <- c(1, 2)

   temp <- lapply(1:2, function(j) {
     b <- design_ls[[q[j]]]
     b[, 2] <- ifelse(b[, 2] < 0.5, b[, 2] + delta, b[, 2] - delta)
     return(b)
   })
   ```

   Then evaluate both candidates:

   ```r
   opt <- lapply(1:2, function(j) {
     R_fun_rcpp(R_info$R, old_X, temp[[j]], 1, beta)
   })
   ```

   In Python, evaluate the same candidates:

   ```python
   py_opt = [
       _core.corr_matrix(old_X, temp_j, py_R, 1.0, beta)
       for temp_j in temp
   ]
   ```

   Compare each Python matrix with the corresponding `opt[[j]]$R`.

Notes:

- `R_fun_rcpp()` returns both `R` and `L`; `_core.corr_matrix()` only returns
  `R`.
- The Cholesky factor `L` can be tested separately using SciPy if needed.
- The corresponding pytest file should be `test_1_2.py`.

### 1.3 Test `Q_fun_rcpp`

This test checks whether Python's `_core.q_matrix()` matches
`Q_fun_rcpp()` from `Q.cpp`.

The test should cover both ways that `Q_fun_rcpp()` is used:

- building the full design-vs-QMC correlation matrix from scratch;
- incrementally appending rows for new design points to an existing
  design-vs-QMC correlation matrix.

1. Generate the same base design in R:

   ```r
   Rcpp::sourceCpp("R/Q.cpp")
   source("R/soft_measure_lib.R")
   source("R/soft.R")
   D <- soft(2, 4)
   A <- D[1:4, ]
   l <- 4
   delta <- D[5, 1] - D[1, 1]
   beta <- 1 / (2 * l)^2
   qmc <- qmc_generate(500, 2)
   ```

2. Build the first two OFAT blocks:

   ```r
   design_ls <- list(A)

   a <- design_ls[[1]]
   a[, 1] <- ifelse(a[, 1] < 0.5, a[, 1] + delta, a[, 1] - delta)
   design_ls <- append(design_ls, list(a))

   old_X <- Reduce(rbind, design_ls)
   R_info <- R_fun_rcpp(NULL, old_X, NULL, 1, beta)
   ```

3. Setting 1: test the full design-vs-QMC matrix build.

   In R:

   ```r
   MMSE_info <- Q_fun_rcpp(qmc, old_X, NULL, NULL, R_info$L, 1, beta)
   old_R_qmc <- MMSE_info$new_R
   ```

   In Python:

   ```python
   py_old_R_qmc = _core.q_matrix(qmc, old_X, None, None, 1.0, beta)
   ```

   Compare `py_old_R_qmc` with `MMSE_info$new_R`.

4. Setting 2: test the incremental design-vs-QMC matrix build.

   In R, create two candidate `new_X` matrices by shifting the second factor:

   ```r
   q <- c(1, 2)

   temp <- lapply(1:2, function(j) {
     b <- design_ls[[q[j]]]
     b[, 2] <- ifelse(b[, 2] < 0.5, b[, 2] + delta, b[, 2] - delta)
     return(b)
   })
   ```

   Then evaluate both candidates:

   ```r
   opt <- lapply(1:2, function(j) {
     R_info <- R_fun_rcpp(R_info$R, old_X, temp[[j]], 1, beta)
     MMSE_info <- Q_fun_rcpp(qmc, old_X, old_R_qmc, temp[[j]], R_info$L, 1, beta)
     ans <- list(
       mmse = min(MMSE_info$val),
       old_R = R_info$R,
       old_R_qmc = MMSE_info$new_R
     )
     return(ans)
   })
   ```

   In Python, evaluate the same candidates:

   ```python
   py_opt = []

   for temp_j in temp:
       py_R = _core.corr_matrix(old_X, temp_j, old_R, 1.0, beta)
       py_L = cholesky(py_R + np.eye(py_R.shape[0]) * 1e-6, lower=True)
       py_R_qmc = _core.q_matrix(qmc, old_X, py_old_R_qmc, temp_j, 1.0, beta)
       py_mmse = _q_min_value(py_R_qmc, py_L)
       py_opt.append((py_mmse, py_R, py_R_qmc))
   ```

   Compare each Python `py_R_qmc` matrix with the corresponding
   `opt[[j]]$old_R_qmc`.

   Compare each Python `py_mmse` value with the corresponding `opt[[j]]$mmse`.

Notes:

- `Q_fun_rcpp()` returns both `new_R` and `val`.
- `_core.q_matrix()` only returns the `new_R` matrix.
- The scalar Q objective used by `soft()` is computed in Python by
  `_q_min_value(new_R, L)`, which matches `min(Q_fun_rcpp(...)$val)`.
- This test depends on `R_fun_rcpp()` for the Cholesky factor `L`, so it should
  be implemented after section 1.2 is passing.
- The corresponding pytest file should be `test_1_3.py`.

## 2. Test `_soft.py` Against `soft.R` and `soft_measure_lib.R`

The goal of this group of tests is to check whether the Python design-building
helpers in `src/softDOE/_soft.py` match the corresponding R helpers in
`R/soft.R` and `R/soft_measure_lib.R`.

### 2.1 Test `_a_fun`

This test checks whether Python's `_a_fun()` matches R's `A_fun()`.

`A_fun()` has two branches, so the test should cover both:

- a maximin Latin hypercube branch when `p < factorial(l)`;
- a permutation-based branch when `p >= factorial(l)`.

1. Setting 1: test the maximin Latin hypercube branch.

   In R:

   ```r
   A_info <- A_fun(2, 4)
   ```

   In Python:

   ```python
   py_A_info = _a_fun(2, 4, seed=...)
   ```

   Compare:

   - the shape of `A_info$A` and `py_A_info["A"]`;
   - `A_info$L` and `py_A_info["L"]`;
   - `A_info$quotient` and `py_A_info["quotient"]`;
   - `A_info$mod` and `py_A_info["mod"]`.

   Because this branch uses different LHD generators in R and Python, exact
   numerical equality of `A` is not required. Instead, check that Python
   returns an `l` by `p` matrix with values in the expected level-index range.

   After generating both matrices, store the R and Python designs and compare
   their maximin distance quality using `SFDesign::maximin.crit`.

   In R:

   ```r
   r_crit <- SFDesign::maximin.crit(A_info$A)
   py_crit <- SFDesign::maximin.crit(py_A)
   ```

   Compare `r_crit` and `py_crit` to check whether the Python-generated LHD has
   comparable maximin distance quality to the R-generated LHD.

2. Setting 2: test the permutation-based branch.

   In R:

   ```r
   A_info <- A_fun(30, 4)
   ```

   In Python:

   ```python
   py_A_info = _a_fun(30, 4, seed=...)
   ```

   Compare:

   - `A_info$A` and `py_A_info["A"]`;
   - `A_info$L` and `py_A_info["L"]`;
   - `A_info$quotient` and `py_A_info["quotient"]`;
   - `A_info$mod` and `py_A_info["mod"]`.

   Exact equality is expected in this branch because both implementations
   should generate the same permutation matrix.

Notes:

- For `l = 4`, `factorial(l) = 24`.
- `A_fun(2, 4)` uses the LHD branch because `2 < 24`.
- `A_fun(30, 4)` uses the permutation branch because `30 >= 24`.
- The corresponding pytest file should be `test_2_1.py`.

### 2.2 Test `soft`

This test checks whether Python's public `soft()` function produces SOFT
designs with behavior comparable to R's `soft()` function.

Because the R and Python implementations use different random/LHD generators,
the designs are not expected to be identical point by point. Instead, this
test should compare design structure, visual behavior, and the distribution of
the Q objective values.

1. Setting 1: visually compare small standard SOFT designs.

   In Python:

   ```python
   py_design = soft(2, 4, structure="standard", seed=...)
   ```

   In R:

   ```r
   r_design <- soft(2, 4, structure = "standard")
   ```

   Plot both designs and save the figures. The plots should show comparable
   standard OFAT structure and space-filling behavior.

2. Setting 2: visually compare small strict SOFT designs.

   In Python:

   ```python
   py_design = soft(2, 4, structure="strict", seed=...)
   ```

   In R:

   ```r
   r_design <- soft(2, 4, structure = "strict")
   ```

   Plot both designs and save the figures. The plots should show comparable
   strict OFAT structure and space-filling behavior.

3. Setting 3: compare the distribution of Q objective values for standard SOFT
   designs.

   Repeat the following simulation 30 times:

   - generate a Python design with `soft(10, 4, structure="standard")`;
   - generate an R design with `soft(10, 4, structure = "standard")`;
   - compute one Q objective value for each design using the same QMC sample
     and the Python Q calculation;
   - take `log(Q)` for each Q value;
   - plot the distribution of the 30 R `log(Q)` values and overlay the
     distribution of the 30 Python `log(Q)` values on the same graph.

   The goal is not to compare the Q values pair by pair. The goal is to check
   whether the Python implementation produces a comparable distribution of Q
   values across repeated design generations.

4. Setting 4: compare the distribution of Q objective values for strict SOFT
   designs.

   Repeat the same 30-run simulation as Setting 3, but use:

   ```python
   soft(10, 4, structure="strict")
   ```

   and:

   ```r
   soft(10, 4, structure = "strict")
   ```

   Plot the R and Python `log(Q)` distributions on the same graph and compare
   their overall behavior.

5. Setting 5: repeat the Q-distribution comparison with a larger design-search
   Monte Carlo size.

   The current simulation uses:

   ```python
   m = 500
   ```

   Repeat Settings 3 and 4 with:

   ```python
   m = 1000
   ```

   and the corresponding R call:

   ```r
   m = 1000
   ```

   Compare the R and Python `log(Q)` distributions again. The purpose is to
   check whether increasing `m` makes the discrepancy between the R and Python
   implementations smaller.

Notes:

- The visual checks are diagnostic and should save plots for manual inspection.
- The Q distribution checks should be automated and should also save overlaid
  distribution plots for manual inspection.
- The Q values are very small, so the distribution plots should use `log(Q)`
  instead of the original Q values.
- The same QMC sample should be used when evaluating Python and R designs in
  each simulation run.
- Exact equality of designs is not expected.
- Exact equality of individual Q values is not expected.
- The test should save separate overlaid Q-distribution plots for standard and
  strict designs.
- The corresponding pytest file should be `test_2_2.py`.

## 3. Test `_measure.py` Against `measure.R`

The goal of this group of tests is to check whether Python's `measure()`
function produces the same screening measures as R's `measure()` function.

The main parity test should use the same design matrix and the same response
vector in both languages. This avoids confounding the `measure()` comparison
with differences between R and Python design generation.

Use the Friedman-style test function from the R examples:

```r
fun <- function(X) {
  Y <- 10 * sin(pi * X[1] * X[2]) +
    20 * (X[3] - 0.5)^2 +
    10 * X[4] +
    5 * X[5]
  return(Y)
}
```

### 3.1 Test `measure` on a Standard SOFT Design

1. Generate a standard SOFT design in R:

   ```r
   Rcpp::sourceCpp("R/Q.cpp")
   source("R/soft_measure_lib.R")
   source("R/soft.R")
   source("R/measure.R")
   D <- soft(p = 10, l = 4, structure = "standard")
   y <- apply(D, 1, fun)
   r_result <- measure(D, y)
   ```

2. Use the same `D` and `y` in Python:

   ```python
   py_result = measure(D, y)
   ```

3. Compare:

   - `py_result["mustar"]` with `r_result$mustar`;
   - `py_result["t"]` with `r_result$t`.

Exact numerical equality up to floating-point tolerance is expected.

### 3.2 Test `measure` on a Strict SOFT Design

1. Generate a strict SOFT design in R:

   ```r
   Rcpp::sourceCpp("R/Q.cpp")
   source("R/soft_measure_lib.R")
   source("R/soft.R")
   source("R/measure.R")
   D <- soft(p = 10, l = 4, structure = "strict")
   y <- apply(D, 1, fun)
   r_result <- measure(D, y)
   ```

2. Use the same `D` and `y` in Python:

   ```python
   py_result = measure(D, y)
   ```

3. Compare:

   - `py_result["mustar"]` with `r_result$mustar`;
   - `py_result["t"]` with `r_result$t`.

Exact numerical equality up to floating-point tolerance is expected.

Notes:

- Exact R-vs-Python equality should be tested only when both implementations
  use the same `D` and `y`.
- The corresponding pytest file should be `test_3_measure.py`.
