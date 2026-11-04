# Worker for diagnose_mofat_case.py; all namespace changes die with this process.
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 2L) stop("Expected diagnostic input.json and new output.json")
if (file.exists(args[[2]])) stop("Output already exists")
for (pkg in c("MOFAT", "jsonlite", "spacefillr")) {
  if (!requireNamespace(pkg, quietly=TRUE)) stop("Install R package ", pkg)
}
if (as.character(packageVersion("MOFAT")) != "2.0") stop("Requires MOFAT 2.0")
input <- jsonlite::read_json(args[[1]], simplifyVector=FALSE)
s <- input$case
if (s[["function"]] != "a_soft" || identical(s$structure,"strict") ||
    isTRUE(s$expect_error) || !is.null(s$supplied_design) || is.null(s$base)) {
  stop("Only valid shared-base standard A-SOFT cases are supported")
}
mat <- function(x) do.call(rbind, lapply(x, unlist, use.names=FALSE))
ns <- asNamespace("MOFAT")
fixed <- s$base
fixed$A <- mat(fixed$A)
assignInNamespace("A_fun", local({base <- fixed; function(p,l) base}), ns="MOFAT")
prepared <- get(".soft_prepare",ns)(s$p,s$l,s$m,total_levels=2*s$l+s$n_aug)
evaluate <- function(A) get(".soft_initial_cpu",ns)(A,prepared,"standard")
A <- prepared$A
current <- evaluate(A)
initial <- list(A=A,design=current$design,log_Q=current$log_Q)
trace <- list()
p0 <- if(is.null(prepared$remainder)) s$p else prepared$remainder
p1 <- if(is.null(prepared$quotient)) 0 else prepared$factorial_l*prepared$quotient
if (p0 > 1L) for(i in seq_len(s$l-1L)) for(j in (i+1L):s$l) for(k in 2L:p0) {
  column <- p1+k
  candidate_A <- A
  candidate_A[c(i,j),column] <- candidate_A[c(j,i),column]
  candidate <- evaluate(candidate_A)
  accepted <- candidate$log_Q > current$log_Q
  trace[[length(trace)+1L]] <- list(step=c(i,j,column),incumbent_log_Q=current$log_Q,
    candidate_log_Q=candidate$log_Q,delta=candidate$log_Q-current$log_Q,
    accepted=accepted,A=candidate_A,design=candidate$design)
  if (accepted) { A <- candidate_A; current <- candidate }
}
native <- get(".soft_optimize_cpu",ns)(prepared,"standard")
python_scores <- lapply(input$python$trace,function(step) {
  get("Q_score_cpu_rcpp",ns)(prepared$qmc,mat(step$design),prepared$alpha,
                            prepared$beta,prepared$nugget)$log_Q
})
b <- .25/(2*s$l+s$n_aug)
pool <- unique(b+(1-2*b)*matrix(spacefillr::generate_sobol_set(s$N,s$p,seed=0),ncol=s$p))
augment <- function(base) {
  result <- MOFAT::a_soft(n_aug=s$n_aug,N=s$N,ofat_design=base,backend="cpu")
  score <- MOFAT::Q(1,(2*s$l+s$n_aug)^-2,result$design,pool,backend="cpu")
  if (result$backend_used != "cpu" || score$backend_used != "cpu") stop("Expected CPU")
  list(design=result$design,Q=score$Q,normalized_Q=score$normalized_Q)
}
result <- list(initial=initial,trace=trace,reference=current,native=native,
  reference_native=list(exactly_equal=isTRUE(all.equal(current$design,native$design,tolerance=0)),
                        max_absolute_difference=max(abs(current$design-native$design))),
  python_trace_r_log_Q=python_scores,evaluation=prepared$qmc,
  augmentation=list(python=augment(mat(input$python$native$design)),r=augment(native$design)),
  environment=list(MOFAT=as.character(packageVersion("MOFAT")),R=R.version.string,
                   package_path=find.package("MOFAT"),session=capture.output(sessionInfo())))
jsonlite::write_json(result,args[[2]],digits=17,auto_unbox=TRUE,pretty=TRUE,null="null")
