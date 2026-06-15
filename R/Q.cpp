// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
using namespace Rcpp;

// Calculate MIM kernel between two points A[ia,] and B[ib,]: R(A[ia,], B[ib,]).
// beta = theta^2
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

// Calculate the correlation matrix for the new design points (new_X) and the existing points (X).
// old_R is the preevaluated correlation matrix for X.
// [[Rcpp::export]]
Rcpp::List R_fun_rcpp(Nullable<arma::mat> old_R, const arma::mat& X,
                      Nullable<arma::mat> new_X,
                      const double alpha, const double beta,
                      const double nugget = 1e-6)
{
  const arma::uword n = X.n_rows;
  arma::mat R;
  
  if (new_X.isNull()) {
    // Full build: n x n
    R.eye(n, n);
    
    for (arma::uword i = 0; i < n; ++i) {
      for (arma::uword j = i + 1; j < n; ++j) {
        const double c = corr_idx_fast(X, i, X, j, alpha, beta);
        R(i, j) = c;
        R(j, i) = c;
      }
    }
    
  } else {
    if (old_R.isNull()) stop("old_R must be provided when new_X is not NULL.");
    
    const arma::mat X2 = as<arma::mat>(new_X);
    const arma::mat R0 = as<arma::mat>(old_R);
    
    const arma::uword k = X2.n_rows;
    const arma::uword new_n = n + k;
    
    R.eye(new_n, new_n);
    R.submat(0, 0, n - 1, n - 1) = R0;
    
    // Old-New block
    for (arma::uword i = 0; i < k; ++i) {
      const arma::uword ii = n + i;
      for (arma::uword j = 0; j < n; ++j) {
        const double c = corr_idx_fast(X2, i, X, j, alpha, beta);
        R(ii, j) = c;
        R(j, ii) = c;
      }
    }
    
    // New-New block
    for (arma::uword i = 0; i < k; ++i) {
      const arma::uword ii = n + i;
      for (arma::uword j = i + 1; j < k; ++j) {
        const arma::uword jj = n + j;
        const double c = corr_idx_fast(X2, i, X2, j, alpha, beta);
        R(ii, jj) = c;
        R(jj, ii) = c;
      }
    }
  }
  
  // Nugget + Cholesky
  arma::mat M = R;
  M.diag() += nugget;
  
  arma::mat U;
  const bool ok = arma::chol(U, M); // upper
  if (!ok) stop("Cholesky failed: matrix not positive definite (even with nugget).");
  
  arma::mat L = U.t(); // lower
  return List::create(_["R"] = R, _["L"] = L);
}

// Evaluate Q for each test point (qmc) based on the design points (X + new_X).
// old_R is the preevaluated correlation matrix for X.
// [[Rcpp::export]]
Rcpp::List Q_fun_rcpp(const arma::mat& qmc, const arma::mat& X,
                      Rcpp::Nullable<arma::mat> old_R,
                      Rcpp::Nullable<arma::mat> new_X,
                      const arma::mat& L,
                      const double alpha, const double beta)
{
  const arma::uword m = qmc.n_rows;   // # test points
  const arma::uword n = X.n_rows;     // # design points
  
  arma::mat new_R;
  
  // ------------------ build new_R ------------------
  if (new_X.isNull()) {
    new_R.set_size(n, m);
    
    for (arma::uword j = 0; j < m; ++j) {
      for (arma::uword i = 0; i < n; ++i) {
        new_R(i, j) = corr_idx_fast(qmc, j, X, i, alpha, beta);
      }
    }
    
  } else {
    if (old_R.isNull())
      Rcpp::stop("old_R (old_R_qmc) must be provided when new_X is not NULL.");
    
    const arma::mat X2 = Rcpp::as<arma::mat>(new_X);
    const arma::mat R0 = Rcpp::as<arma::mat>(old_R); // n x m
    
    const arma::uword k = X2.n_rows;
    const arma::uword new_n = n + k;
    
    new_R.set_size(new_n, m);
    
    // copy existing block
    new_R.submat(0, 0, n - 1, m - 1) = R0;
    
    // fill new rows only
    for (arma::uword j = 0; j < m; ++j) {
      for (arma::uword i = 0; i < k; ++i) {
        new_R(n + i, j) = corr_idx_fast(qmc, j, X2, i, alpha, beta);
      }
    }
  }
  
  // ------------------ solve + MMSE ------------------
  arma::mat A = arma::solve(arma::trimatl(L), new_R, arma::solve_opts::fast);
  
  arma::rowvec val_row = arma::sum(arma::square(A), 0);
  
  return Rcpp::List::create(
    Rcpp::Named("new_R")     = new_R,
    Rcpp::Named("val")       = val_row.t()
  );
}
