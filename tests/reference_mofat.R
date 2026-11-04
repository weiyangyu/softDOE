# Run in a separate process: Rscript --vanilla tests/reference_mofat.R output.json version
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2L)
for (pkg in c("MOFAT", "jsonlite", "spacefillr")) {
  if (!requireNamespace(pkg, quietly = TRUE)) stop("Install R package ", pkg)
}
version <- as.character(packageVersion("MOFAT"))
if (version != args[[2]]) stop("Expected MOFAT ", args[[2]], "; found ", version)
ns <- asNamespace("MOFAT")
original_A <- get("A_fun", ns)
# This changes only this temporary R process, never the package on disk.
set_base <- function(base) {
  assignInNamespace("A_fun", local({fixed <- base; function(p, l) fixed}), ns = "MOFAT")
}
friedman <- function(D) {
  10*sin(pi*D[,1]*D[,2]) + 20*(D[,3]-.5)^2 + 10*D[,4] + 5*D[,5]
}
cases <- list()
for (spec in list(c(2L,4L), c(3L,2L), c(4L,2L), c(5L,4L))) {
  p <- spec[[1]]; l <- spec[[2]]
  set.seed(100L+p+l)
  base <- original_A(p,l)
  set_base(base)
  for (structure in c("standard", "strict")) {
    design <- MOFAT::soft(p,l,structure,m=32L,backend="cpu")
    augmented <- MOFAT::a_soft(p,l,structure,n_aug=3L,m=32L,N=64L,backend="cpu")
    bound <- .25/(2*l+3)
    pool <- bound+(1-2*bound)*spacefillr::generate_sobol_set(64L,p,seed=0)
    scores <- MOFAT::Q(1,1/(2*l+3)^2,augmented$design,pool,backend="cpu")
    y <- if (p >= 5) friedman(design$design) else
      rowSums(sweep(design$design,2,seq_len(p),"*"))+design$design[,1]*design$design[,2]
    cases[[paste(p,l,structure,sep="_")]] <- list(p=p,l=l,structure=structure,
      base=base, soft=design, augmented=augmented, augmented_Q=scores,
      y=y, measure=MOFAT::measure(design$design,y))
  }
}
assignInNamespace("A_fun", original_A, ns="MOFAT")
qmc <- lapply(list(c(17L,1L,10L),c(32L,3L,0L),c(17L,11L,10L),c(2L,1024L,10L)),
  function(s) list(n=s[[1]],p=s[[2]],threshold=s[[3]],
    points=MOFAT::qmc_generate(s[[1]],s[[2]],s[[3]])))
D <- matrix(c(.1,.2,.8,.9,.4,.7),ncol=2,byrow=TRUE)
Z <- matrix(c(0,0,1,1,.3,.6,.7,.2),ncol=2,byrow=TRUE)
q <- lapply(c(.25,1,2.5), function(a)
  list(alpha=a,beta=1/64,design=D,qmc=Z,result=MOFAT::Q(a,1/64,D,Z,backend="cpu")))
supplied <- matrix(c(.1,.2,.35,.9,.8,.4,.7,.6,.25,.3,.5,.75),ncol=2,byrow=TRUE)
extended <- MOFAT::a_soft(n_aug=3L,N=64L,backend="cpu",ofat_design=supplied)
bound <- .25/(2*2+3)
pool <- bound+(1-2*bound)*spacefillr::generate_sobol_set(64L,2L,seed=0)
extended_Q <- MOFAT::Q(1,1/(2*2+3)^2,extended$design,pool,backend="cpu")
# Include actual MOFAT() output, not only SOFT, for screening comparisons.
set.seed(1)
mofat <- MOFAT::mofat(10L,4L)
y <- friedman(mofat)
jsonlite::write_json(list(provenance=list(MOFAT=version,R=R.version.string,
  spacefillr=as.character(packageVersion("spacefillr")),
  SFDesign=as.character(packageVersion("SFDesign")),
  source="Installed MOFAT namespace in an isolated R process; shared A_fun base"),
  cases=cases,qmc=qmc,Q=q,supplied=list(input=supplied,result=extended,Q=extended_Q),
  mofat_measure=list(design=mofat,y=y,result=MOFAT::measure(mofat,y))),
  args[[1]],digits=NA,auto_unbox=TRUE,pretty=TRUE,null="null")
