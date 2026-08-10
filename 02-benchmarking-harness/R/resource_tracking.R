# resource_tracking.R
#################################################
# Runtime and peak-memory measurement for tool wrappers.
#
# Every wrapper reports two resource numbers alongside its predictions:
#
#   runtime_secs           wall-clock seconds for the core annotation call
#   peak_system_memory_mb  peak RAM attributable to that call, in MiB
#
# The measurement window is deliberately narrow. It covers model setup,
# training (where the method has a training step), and prediction. It excludes
# shared preprocessing, the single per-dataset marker derivation, object
# conversion (Seurat -> SingleCellExperiment / AnnData), package loading, and
# — for tools that shell out to Python — interpreter startup. Those costs are
# paid once per dataset and reused across tools, so charging them to whichever
# tool happened to run first would not be meaningful.
#
# In the original implementation this measurement was written inline in every
# wrapper as an if/else on peakRAM's availability, which meant each tool's core
# algorithm appeared *twice* in its own file — once in each branch. Keeping the
# two copies in sync was a standing source of drift. `with_resource_tracking()`
# exists so the core call is written exactly once.
#################################################


#' Run an expression, returning its value plus runtime and peak memory
#'
#' Wrap a wrapper's core annotation call in this. The expression is evaluated
#' once, in the caller's environment, so it can refer to local variables
#' normally.
#'
#' Inputs:
#'   - expr: an unevaluated expression (typically a braced block). Errors are
#'     not caught here; they propagate to the wrapper's own handler.
#' Outputs: list with
#'   - value:                 whatever `expr` evaluated to
#'   - runtime_secs:          numeric, wall-clock seconds
#'   - peak_system_memory_mb: numeric MiB, or NA_real_ if peakRAM is absent
#'
#' Usage:
#'   tracked <- with_resource_tracking({
#'     SomeTool::annotate(expr_matrix, marker_table)
#'   })
#'   results               <- tracked$value
#'   runtime_secs          <- tracked$runtime_secs
#'   peak_system_memory_mb <- tracked$peak_system_memory_mb
with_resource_tracking <- function(expr) {
  caller_env <- parent.frame()
  quoted     <- substitute(expr)

  # Environments have reference semantics, so the result survives regardless of
  # which frame the profiler chooses to evaluate `run_once()` in.
  box <- new.env(parent = emptyenv())
  run_once <- function() {
    box$value <- eval(quoted, envir = caller_env)
    invisible(NULL)
  }

  # peakRAM is the memory instrument used throughout; without it we still
  # report runtime rather than failing, and memory falls back to NA.
  if (!requireNamespace("peakRAM", quietly = TRUE)) {
    warning("peakRAM not available; reporting runtime only, memory as NA")
    start_time <- Sys.time()
    run_once()
    return(list(
      value                 = box$value,
      runtime_secs          = as.numeric(difftime(Sys.time(), start_time, units = "secs")),
      peak_system_memory_mb = NA_real_
    ))
  }

  profile <- peakRAM::peakRAM(run_once())

  list(
    value                 = box$value,
    runtime_secs          = profile$Elapsed_Time_sec[1],
    peak_system_memory_mb = profile$Peak_RAM_Used_MiB[1]
  )
}


#' Parse resource figures out of a Python subprocess's stdout
#'
#' Tools whose implementation lives in Python (CellTypist, SCSA, scAnnotate in
#' the full study) run in a separate interpreter, so peakRAM cannot see them.
#' Those helpers instead measure themselves and print two sentinel lines as
#' their last output:
#'
#'   RUNTIME_SECS:12.3456
#'   PEAK_SYSTEM_MEMORY_MB:512.00
#'
#' The R side scrapes them back out. Values are NA when a line is absent, which
#' is what happens if the helper died before reaching its own teardown.
#'
#' Inputs:  - stdout_lines: character vector, the captured stdout
#' Outputs: list(runtime_secs =, peak_system_memory_mb =)
parse_python_resource_output <- function(stdout_lines) {
  extract <- function(prefix) {
    hit <- grep(prefix, stdout_lines, value = TRUE, fixed = TRUE)
    if (length(hit) == 0) return(NA_real_)
    suppressWarnings(as.numeric(sub(paste0("^.*", prefix), "", hit[length(hit)])))
  }

  list(
    runtime_secs          = extract("RUNTIME_SECS:"),
    peak_system_memory_mb = extract("PEAK_SYSTEM_MEMORY_MB:")
  )
}
