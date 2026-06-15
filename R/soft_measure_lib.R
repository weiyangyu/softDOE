############################################ simple functions for soft.R
# Transform the levels of "design" to "levels". Here, "design" means the A0 matrix (matrix of the l base runs).
#' @noRd
level_trans <- function(design, l, levels) {
  b <- design
  b[design < ((l+1)/2)] <- 2*design[design < ((l+1)/2)]-1
  b[design > ((l+1)/2)] <- 2*design[design > ((l+1)/2)]
  design <- matrix(levels[b], nrow = l)
  return(design)
}

# Generate representative vertex points. Here, "design" means the Monte Carlo samples (qmc).
#' @noRd
approx_corner <- function(design) {
  a <- design
  a[design < 0.5] <- 1
  a[design > 0.5] <- 0
  return(a)
}

# Generate Monte Carlo samples as well as the representative vertex points for evaluating Q.
#' @noRd
qmc_generate <- function(n,p,threshold = 10) {
  qmc <- matrix(runif(n*p),nrow=n)
  if(p <= threshold) {
    qmc <- rbind(qmc, SFDesign::full.factorial(p,2))
  } else {
    qmc <- rbind(qmc, approx_corner(qmc))
  }
  qmc <- unique(qmc)
  return(qmc)
}

# Generate A0 matrix. 
#' @noRd
A_fun <- function(p,l) {
  L <- factorial(l)
  if(p < L) {
    A <- matrix(SFDesign::maximinLHD(l,p)$design, ncol = p)
    A <- A*l + 0.5
    a <- NULL
    b <- NULL
  } else {
    a <- p%/%(L)
    b <- p%%L
    P_l <- t(gtools::permutations(l,l,1:l))
    A1 <- do.call(cbind, replicate(a, P_l, simplify = FALSE))
    if(b > 0) {
      A2 <- P_l[,1:b]
    } else {
      A2 = NULL
    }
    A <- cbind(A1, A2)
  }
  ans <- list(A = A, L = L, quotient = a, mod = b)
  return(ans)
}

# Construct a standard OFAT design given an A0 matrix.
#' @noRd
stand_ofat_constr <- function(A, p, delta) {
  design_ls <- list(A)
  for(i in 1:p) {
    a <- design_ls[[1]]
    a[,i] <- ifelse(a[,i] < 0.5, a[,i] + delta, a[,i] - delta)
    design_ls <- append(design_ls, list(a))
  }
  return(Reduce(rbind, design_ls))
}

# Construct a strict OFAT design given an A0 matrix.
#' @noRd
strict_ofat_constr <- function(A, p, delta, qmc) {
  l <- nrow(A)
  beta <- 1/(2*l)^2
  design_ls <- list(A)
  a <- design_ls[[1]]
  a[,1] <- ifelse(a[,1] < 0.5, a[,1] + delta, a[,1] - delta)
  design_ls <- append(design_ls, list(a))
  old_X <- Reduce(rbind, design_ls)
  R_info <- R_fun_rcpp(NULL, old_X, NULL, 1, beta)
  old_R <- R_info$R
  MMSE_info <- Q_fun_rcpp(qmc, old_X, NULL, NULL, R_info$L, 1, beta)
  old_R_qmc <- MMSE_info$new_R
  
  for(i in 2:p) {
    q <- c(1,i)
    temp <- lapply(1:2, function(j) {
      b <- design_ls[[q[j]]]
      b[,i] <- ifelse(b[,i] < 0.5, b[,i] + delta, b[,i] - delta)
      return(b)
    })
    opt <- lapply(1:2, function(j) {
      R_info <- R_fun_rcpp(old_R, old_X, temp[[j]], 1, beta)
      MMSE_info <- Q_fun_rcpp(qmc, old_X, old_R_qmc, temp[[j]], R_info$L, 1, beta)
      ans <- list(mmse = min(MMSE_info$val), old_R = R_info$R, old_R_qmc = MMSE_info$new_R)
      return(ans)
    })
    val <- unlist(purrr::map(opt, "mmse"))
    k <- which.max(val)
    if (k == 1) {
      design_ls <- append(list(temp[[k]]), design_ls)
    } else {
      design_ls <- append(design_ls, list(temp[[k]]))
    }
    
    old_X <- Reduce(rbind, design_ls)
    old_R <- opt[[k]]$old_R
    old_R_qmc <- opt[[k]]$old_R_qmc
    
  }
  
  ans <- list(design = Reduce(rbind, design_ls), val = val[k])
  return(ans)
}


# Optimize for a space-filling OFAT design using deterministic local search algorithm.
#' @noRd
swap <- function(qmc, A, l, p, ofat_type, ofat_constr, levels, delta,
                 L, mod, quotient) {
  #################### initial
  alpha = 1
  beta = 1/(2*l)^2
  A = level_trans(A, l, levels)
  if(ofat_type == "stand") {
    design = ofat_constr(A, p, delta)
    R_info = R_fun_rcpp(NULL, design, NULL, alpha, beta)
    old_val = min(Q_fun_rcpp(qmc, design, NULL, NULL, R_info$L, alpha, beta)$val)
  } else {
    design_info = ofat_constr(A, p, delta, qmc)
    design = design_info$design
    old_val = design_info$val
  }
  ####################
  
  #################### update
  p_0 = ifelse(is.null(mod),p,mod)
  p_1 = ifelse(is.null(quotient),0,L*quotient)

  if(p_0 > 1) { # number of factors that require deterministic swap
    for(i in 1:(l-1)) {
      for(j in (i+1):l) {
        for(k in 2:p_0) {
          x <- A[,(p_1+k)]
          x[c(i, j)] <- x[c(j, i)]
          A_new <- A
          A_new[,(p_1+k)] <- x
          if(ofat_type == "stand") {
            design_new <- ofat_constr(A_new, p, delta)
            new_R_info <- R_fun_rcpp(NULL, design_new, NULL, alpha, beta)
            new_val <- min(Q_fun_rcpp(qmc, design_new, NULL, NULL, new_R_info$L, alpha, beta)$val)
          } else {
            design_info = ofat_constr(A_new, p, delta, qmc)
            design_new = design_info$design
            new_val = design_info$val
          }

          if(new_val > old_val) {
            A <- A_new
            design <- design_new
            old_val <- new_val
          }
        }
      }
    }
  }
  
  ans <- list(design = design, opt_val = old_val)
  return(ans)
}

############################################ simple functions for measure.R
# Given A(i), find A(oi). 
#' @noRd
find_A <- function(X_ls, idx, p) {
  B <- X_ls[[idx]]
  for(j in seq_len(idx-1)) {
    A <- X_ls[[j]]
    col_same <- vapply(seq_len(p), function(k) { # check each column one by one and return a vector of TRUE and FALSE.
      isTRUE(all.equal(A[, k], B[, k], check.attributes = FALSE))
    }, logical(1))
    if(sum(!col_same) == 1) { # if exactly one column differs, return.
      return(list(which_matrix = j, which_factor = which(!col_same))) # Compared to X_ls[[idx]], which factor is changed from which matrix.
    }
  }
}
