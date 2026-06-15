#' @title
#' SOFT Designs
#'
#' @description
#' This function generates a SOFT design under either the standard or strict OFAT structure.
#'
#' @details
#' The \code{soft} function generates the SOFT design for a given number of factors (\eqn{p \geq 2}) and an even number of base runs (\eqn{l}) under an OFAT structure.
#' The total number of runs will be \eqn{l(p+1)}. 
#' Two OFAT structures are available: \code{"standard"} or \code{"strict"}, controlled by the \code{structure} argument.
#' See Yu and Joseph (2026) for further details.
#' 
#' @author Wei-Yang Yu and V. Roshan Joseph
#' 
#' @importFrom stats runif
#' 
#' @param p The number of factors.
#' @param l The number of base runs; must be even.
#' @param structure Indicating the OFAT structure. Choose from
#'   \code{"standard"} or \code{"strict"}. The default is \code{"standard"}.
#' @param m Optional. The number of uniform Monte Carlo
#'   samples used to evaluate \eqn{Q}. The default is \code{500}.
#'
#' @return
#' A SOFT design.
#' 
#' @references 
#' Yu, W. Y. and Joseph, V. R. (2026). “Space-Filling One-Factor-At-A-Time Designs”.
#' 
#' @seealso \code{\link{measure}}
#' 
#' @export
#' 
#' @examples
#' #SOFT with four base runs under the standard structure
#' soft(p=10, l=4)
#' 
#' #SOFT with four base runs under the strict structure
#' soft(p=10, l=4, structure = "strict")

soft <- function(p, l, structure = "standard", m = 500) {
  if (l %% 2 != 0) {
    stop("l must be even.")
  }
  qmc <- qmc_generate(m, p)
  A_info <- A_fun(p,l)
  A <- A_info$A # before level_trans
  L <- A_info$L
  mod <- A_info$mod
  quotient <- A_info$quotient
  x <- 0.25/(2*l)
  levels <- seq(x, 1-x, by = (1-2*x)/(2*l-1))
  delta <- levels[l+1]-levels[1]
  if (structure == "standard") {
    ans <- swap(qmc, A, l, p, "stand", stand_ofat_constr, levels, delta,
                L, mod, quotient)
  } else {
    ans <- swap(qmc, A, l, p, "strict", strict_ofat_constr, levels, delta,
                L, mod, quotient)
  }
  return(ans$design)
}
