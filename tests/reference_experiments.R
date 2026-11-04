# Portable MOFAT 2.0 reference worker. Invoked by run_mofat_experiments.py.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L) stop("Expected mode, manifest.json, output directory")
mode <- args[[1]]
manifest_path <- normalizePath(args[[2]], mustWork = TRUE)
out <- normalizePath(args[[3]], mustWork = TRUE)
for (pkg in c("MOFAT", "jsonlite", "spacefillr")) {
  if (!requireNamespace(pkg, quietly = TRUE)) stop("Install R package ", pkg)
}
manifest <- jsonlite::read_json(manifest_path, simplifyVector = FALSE)
version <- as.character(packageVersion("MOFAT"))
if (version != manifest$mofat_version) stop("Expected MOFAT ", manifest$mofat_version,
                                          "; found ", version)
mat <- function(x) do.call(rbind, lapply(x, unlist, use.names = FALSE))
write_line <- function(x, file) {
  cat(jsonlite::toJSON(x, auto_unbox=TRUE, digits=17, null="null", na="null"),
      "\n", file=file, append=TRUE, sep="")
}
ns <- asNamespace("MOFAT")
original_A <- get("A_fun", ns)
set_base <- function(base) {
  if (is.null(base)) {
    assignInNamespace("A_fun", original_A, ns="MOFAT")
  } else {
    base$A <- mat(base$A)
    if (any(base$A != round(base$A))) stop("Shared base ranks must be integers")
    assignInNamespace("A_fun", local({fixed <- base; function(p,l) fixed}), ns="MOFAT")
  }
}
evaluation <- function(s) {
  if (s[["function"]] == "soft") return(MOFAT::qmc_generate(s$m, s$p))
  b <- .25/(2*s$l+s$n_aug)
  unique(b+(1-2*b)*matrix(spacefillr::generate_sobol_set(s$N,s$p,seed=0),ncol=s$p))
}
score <- function(s, design, backend="cpu") {
  levels <- 2*s$l + if (s[["function"]] == "a_soft") s$n_aug else 0
  MOFAT::Q(1, 1/levels^2, design, evaluation(s), backend=backend)
}
if (mode == "baseline") {
  info <- MOFAT::mofat_backend_info()
  for (backend in unlist(manifest$r_backends)) {
    if (!isTRUE(info[[backend]]$available)) stop("R ",backend," unavailable: ", info[[backend]]$reason)
  }
  functions_path <- file.path(out,"reference-functions.txt")
  capture.output(lapply(c("soft", "a_soft", "Q", "A_fun", ".soft_prepare",
                         ".soft_optimize_backend", ".soft_augment_backend"),
                        function(n) get(n,ns)), file=functions_path)
  jsonlite::write_json(list(MOFAT=version, R=R.version.string,
    package_path=find.package("MOFAT"), backends=info,
    dependencies=lapply(c("spacefillr","SFDesign","Rcpp"),function(p)
      list(package=p,version=as.character(packageVersion(p)))),
    functions_md5=unname(tools::md5sum(functions_path)),
    session=capture.output(sessionInfo()), thread_environment=Sys.getenv(c(
      "OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","VECLIB_MAXIMUM_THREADS","MKL_NUM_THREADS"))),
    file.path(out,"r-environment.json"),auto_unbox=TRUE,pretty=TRUE,null="null")
  output <- file.path(out,"r-results.jsonl")
  for (s in manifest$cases) for (backend in unlist(manifest$r_backends)) {
    result <- tryCatch({
      set_base(s$base)
      kwargs <- list(p=s$p,l=s$l,structure=s$structure,m=s$m,backend=backend)
      if (s[["function"]] == "a_soft") {
        kwargs$n_aug <- s$n_aug; kwargs$N <- s$N
        if (!is.null(s$supplied_design)) kwargs$ofat_design <- mat(s$supplied_design)
        if (s$tag == "supplied_override") { kwargs$p <- -1; kwargs$l <- -1 }
      }
      call <- function() {set.seed(s$seed); do.call(get(s[["function"]], ns), kwargs)}
      if (isTRUE(s$expect_error)) {
        err <- tryCatch({call(); NULL}, error=function(e) conditionMessage(e))
        if (is.null(err)) stop("Expected invalid arguments to be rejected")
        list(id=s$id,backend=backend,status="expected_error",error=err)
      } else {
        warm <- call()
        if (warm$backend_used != backend) stop("Unexpected fallback: ",warm$fallback_reason)
        if (s[["function"]] == "a_soft") score(s,warm$design,backend)
        times <- scored <- numeric(manifest$repeats)
        for (i in seq_len(manifest$repeats)) {
          start <- as.numeric(Sys.time()); actual <- call()
          middle <- as.numeric(Sys.time())
          returned <- if (s[["function"]] == "soft") actual else score(s,actual$design,backend)
          finish <- as.numeric(Sys.time())
          if (actual$backend_used != backend || returned$backend_used != backend)
            stop("Unexpected fallback in timed workflow or score")
          if (!isTRUE(all.equal(actual$design,warm$design,tolerance=0)))
            stop("Repeated calls changed the design")
          times[[i]] <- middle-start; scored[[i]] <- finish-start
        }
        if (!is.null(kwargs$ofat_design) &&
            !identical(unname(actual$design[seq_len(nrow(kwargs$ofat_design)),,drop=FALSE]),
                       unname(kwargs$ofat_design))) stop("Supplied rows were changed")
        list(id=s$id,backend=backend,status="ok",design=actual$design,
             Q=returned$Q,normalized_Q=returned$normalized_Q,
             Q_origin=if(s[["function"]] == "soft") "returned" else "separate R Q()",
             cpu_score=score(s,actual$design),qmc_size=actual$qmc_size,
             device=actual$device,fallback_reason=actual$fallback_reason,
             seconds=times,scored_seconds=scored)
      }
    },error=function(e) list(id=s$id,backend=backend,status="error",error=conditionMessage(e)))
    write_line(result,output)
    cat("R",s$id,backend,result$status,"\n"); flush.console()
  }
  assignInNamespace("A_fun",original_A,ns="MOFAT")
} else if (mode == "score") {
  cases <- setNames(manifest$cases,vapply(manifest$cases,function(s) s$id,""))
  input <- file(file.path(out,"python-designs.jsonl"),"r")
  repeat {
    line <- readLines(input,n=1,warn=FALSE)
    if (!length(line)) break
    item <- jsonlite::fromJSON(line,simplifyVector=FALSE)
    result <- tryCatch(list(id=item$id,backend=item$backend,status="ok",
      score=score(cases[[item$id]],mat(item$design))),error=function(e)
      list(id=item$id,backend=item$backend,status="error",error=conditionMessage(e)))
    write_line(result,file.path(out,"r-python-scores.jsonl"))
  }
  close(input)
} else stop("Unknown mode")
