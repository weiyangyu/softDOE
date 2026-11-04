# Compare the installed MOFAT package using benchmark_strict.py's shared inputs.
# Rscript tools/benchmark_strict.R /tmp/softdoe-bench /tmp/mofat-bench.json
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2L)
suppressPackageStartupMessages(library(MOFAT))
metadata <- jsonlite::fromJSON(file.path(args[1], "timing.json"))
prepared <- MOFAT:::.soft_prepare(metadata$p, metadata$l, metadata$m)
prepared$A <- as.matrix(read.csv(file.path(args[1], "A.csv"), header = FALSE))
prepared$qmc <- as.matrix(read.csv(file.path(args[1], "qmc.csv"), header = FALSE))
info <- mofat_backend_info()
results <- list(p = metadata$p, l = metadata$l, m = metadata$m,
                A = prepared$A, MOFAT = as.character(packageVersion("MOFAT")),
                R = R.version.string)
for (backend in c("metal", "cpu")) {
    if (!info[[backend]]$available) stop(paste(backend, "is unavailable"))
    times <- numeric(3)
    for (i in seq_along(times)) {
        times[i] <- system.time(result <- MOFAT:::.soft_optimize_backend(
            prepared, "strict", backend))[["elapsed"]]
        cat(backend, times[i], "s; log(Q) =", format(result$log_Q, digits = 17), "\n")
    }
    results[[backend]] <- list(seconds = times, median_seconds = median(times),
                              log_Q = result$log_Q, design = result$design)
}
jsonlite::write_json(results, args[2], digits = 17, pretty = TRUE, auto_unbox = TRUE)
