# Read-only reference runner. All output is written to the supplied JSON path.
# Usage: Rscript tools/generate_mofat_fixtures.R /path/to/MOFAT output.json
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2L, requireNamespace("MOFAT", quietly = TRUE),
          requireNamespace("jsonlite", quietly = TRUE))
reference <- normalizePath(args[[1]])
out <- normalizePath(dirname(args[[2]]), mustWork = TRUE)
stopifnot(!startsWith(paste0(out, "/"), paste0(reference, "/")))
env <- new.env(parent = asNamespace("MOFAT"))
# Compile a string copy in R's temporary directory, never in the reference tree.
Rcpp::sourceCpp(code=paste(readLines(file.path(reference,"src","Q.cpp")),
                          collapse="\n"), env=env, cacheDir=tempdir())
for (file in c("soft_measure_lib.R", "soft_backend.R", "soft.R",
               "soft_augment.R", "measure.R")) {
  sys.source(file.path(reference, "R", file), envir = env)
}
original_A <- env$A_fun
cases <- list()
for (p in c(2L, 3L, 4L, 10L)) {
  l <- if (p == 3L) 2L else 4L
  set.seed(42L + p)
  base <- original_A(p, l)
  env$A_fun <- local({fixed <- base; function(p,l) fixed})
  for (structure in c("standard", "strict")) {
    result <- env$soft(p, l, structure, m = 64L, backend = "cpu")
    augmented <- env$a_soft(p, l, structure, n_aug = 3L,
                           m = 64L, N = 128L, backend = "cpu")
    cases[[paste(p,l,structure,sep="_")]] <- list(
      p=p, l=l, structure=structure, m=64L, A=base$A,
      soft=result, a_soft=augmented)
  }
}
# Actual MOFAT documentation examples, with the initial A exported for replay.
set.seed(123)
base <- original_A(2L, 4L)
env$A_fun <- local({fixed <- base; function(p,l) fixed})
examples <- list(A=base$A, soft=env$soft(2,4,backend="cpu"),
                 a_soft=env$a_soft(2,4,n_aug=8,backend="cpu"))
existing <- matrix(c(.10,.20,.35,.90,.80,.40,.70,.60,.25,.30,.50,.75),
                   ncol=2, byrow=TRUE)
provided <- list(input=existing, result=env$a_soft(n_aug=8,N=10000,
                        backend="cpu",ofat_design=existing))
# Screening example: MOFAT's Friedman function on standard and strict SOFT.
measure_cases <- list()
env$A_fun <- original_A
for (structure in c("standard", "strict")) {
  set.seed(101)
  D <- env$soft(10,4,structure,backend="cpu")$design
  y <- apply(D,1,function(x) 10*sin(pi*x[1]*x[2])+20*(x[3]-.5)^2+10*x[4]+5*x[5])
  measure_cases[[structure]] <- list(design=D,y=y,result=env$measure(D,y))
}
qmc <- lapply(list(c(1000L,2L,10L),c(32L,3L,10L),c(32L,11L,10L),
                  c(32L,3L,0L),c(2L,1024L,10L)), function(spec) {
  list(n=spec[1],p=spec[2],threshold=spec[3],
       points=env$qmc_generate(spec[1],spec[2],spec[3]))
})
source_paths <- c("R/soft.R","R/soft_augment.R","R/soft_backend.R",
                  "R/soft_measure_lib.R","R/measure.R","src/Q.cpp")
hashes <- setNames(as.list(tools::md5sum(file.path(reference,source_paths))),source_paths)
jsonlite::write_json(list(
  provenance=list(R=R.version.string, MOFAT=as.character(packageVersion("MOFAT")),
                  spacefillr=as.character(packageVersion("spacefillr")),
                  SFDesign=as.character(packageVersion("SFDesign")),
                  source="MOFAT R and CPU C++ source read-only; compilation in temporary directory",
                  md5=hashes),
  cases=cases, examples=examples, supplied=provided, measures=measure_cases,qmc=qmc
), args[[2]], digits=NA, auto_unbox=TRUE, pretty=TRUE)
cat("Wrote",args[[2]],"\n")
