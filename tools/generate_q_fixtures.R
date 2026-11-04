# Read MOFAT source without writing into its checkout.
# Usage: Rscript tools/generate_q_fixtures.R /path/to/MOFAT output.json
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==2L, requireNamespace("MOFAT", quietly=TRUE),
          requireNamespace("jsonlite", quietly=TRUE))
reference <- normalizePath(args[[1]])
out <- normalizePath(dirname(args[[2]]), mustWork=TRUE)
stopifnot(!startsWith(paste0(out,"/"),paste0(reference,"/")))
env <- new.env(parent=asNamespace("MOFAT"))
Rcpp::sourceCpp(code=paste(readLines(file.path(reference,"src/Q.cpp")),collapse="\n"),
                env=env,cacheDir=tempdir())
for (file in c("soft_backend.R","Q.R"))
  sys.source(file.path(reference,"R",file),envir=env)
cases <- list()
add <- function(name,alpha,beta,design,qmc) {
  cases[[name]] <<- list(alpha=alpha,beta=beta,design=design,qmc=qmc,
                        result=env$Q(alpha,beta,design,qmc,backend="cpu"))
}
design <- matrix(c(.1,.2,.8,.9,.4,.7),ncol=2,byrow=TRUE)
points <- matrix(c(0,0,1,1,.3,.6,.7,.2),ncol=2,byrow=TRUE)
for (alpha in c(.25,1,2.5)) add(paste0("alpha_",alpha),alpha,1/64,design,points)
add("one_dimension",.7,.2,matrix(c(-2,.3,4),ncol=1),matrix(c(-3,0,2),ncol=1))
add("duplicate_rows",1,1/64,design[c(1,1,2),],points)
add("underflow",2.5,1/144,matrix(0,2,160),matrix(1,2,160))
# Replay the actual documentation example using its shared optimized design.
fixture <- jsonlite::read_json("tests/fixtures/mofat_reference.json",simplifyVector=TRUE)
add("documentation_example",1,1/64,fixture$examples$soft$design,
    MOFAT::qmc_generate(1000,2))
augmented <- list()
for (name in names(fixture$cases)) {
  case <- fixture$cases[[name]]
  n_aug <- 3L; beta <- 1/(2*case$l+n_aug)^2; b <- .25/(2*case$l+n_aug)
  pool <- b + spacefillr::generate_sobol_set(128,case$p,seed=0)*(1-2*b)
  augmented[[name]] <- env$Q(1,beta,case$a_soft$design,pool,backend="cpu")
}
inputs <- c("R/Q.R","R/soft_backend.R","src/Q.cpp")
jsonlite::write_json(list(provenance=list(R=R.version.string,
    MOFAT=as.character(packageVersion("MOFAT")),
    source="Read-only MOFAT source; C++ compiled from a string in a temporary directory",
    md5=setNames(as.list(tools::md5sum(file.path(reference,inputs))),inputs)),
    cases=cases,augmented=augmented),args[[2]],digits=NA,auto_unbox=TRUE,pretty=TRUE)
cat("Wrote",args[[2]],"\n")
