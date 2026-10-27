#!/usr/bin/env bash
# Regenerate the paper's statistics and figures from the Zenodo data bundle.
#
#   reproduce/run_all.sh <bundle-dir> [numbers|figures|analyses|all]
#
# <bundle-dir> is the unpacked data bundle. The scripts in reproduce/code/ sit at
# the same relative paths as the data they read, so this runner copies them into
# <bundle-dir> and runs each one from its own folder. Unpack the bundle somewhere
# you are happy to write into; nothing outside <bundle-dir> is modified (except
# compute_*_numbers.py, which also drop CSV copies of their tables in /tmp).
#
#   numbers   every statistic quoted in the text (Python; minutes)
#   figures   Figures 1-4 and Supplementary Figures 1-14 (Python; minutes)
#   analyses  the per-phase R analyses behind them, incl. the UMAP tiles that
#             Supplementary Fig. 2 assembles (R; ~1 h, needs Seurat for UMAPs)
#   all       analyses, then numbers, then figures
#
# Logs go to <bundle-dir>/reproduce_logs/<step>.log; the summary lists each step
# as OK or FAILED. Interpreters: $PYTHON (default python3), $RSCRIPT (default Rscript).
set -uo pipefail

[ $# -ge 1 ] || { sed -n '2,20p' "$0"; exit 2; }
B=$(cd "$1" && pwd) || exit 2
WHAT=${2:-all}
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PYTHON:-python3}
RS=${RSCRIPT:-Rscript}
export BENCH_ROOT="$B"
LOG="$B/reproduce_logs"
FIG="$B/manuscript/publication_manuscript/figures"
NOTES="$B/manuscript/Notes"
mkdir -p "$LOG"
[ -f "$B/MANIFEST.sha256" ] || { echo "error: $B does not look like the data bundle (no MANIFEST.sha256)"; exit 2; }

# Overlay the code onto the bundle (code only; data files are never overwritten).
cp -R "$HERE/code/." "$B/"

STATUS=()
step() {  # step <name> <dir> <command...>
  local name=$1 dir=$2; shift 2
  printf '%-34s' "$name"
  if (cd "$dir" && "$@") >"$LOG/$name.log" 2>&1; then
    echo OK; STATUS+=("OK      $name")
  else
    echo "FAILED (see reproduce_logs/$name.log)"; STATUS+=("FAILED  $name")
  fi
}

analyses() {
  local R="$B/results"
  step R_P4_synthetic           "$R/P4 Synthetic Datasets V1" "$RS" Plots_Analysis.R
  step R_P4_orthogonal_factors  "$R/P4 Synthetic Datasets V1" "$RS" Plots_Analysis_OrthogonalFactors.R
  step R_P5_real                "$R/P5 Real Life Datasets V2" "$RS" Plots_Analysis.R
  step R_P5_vs_P4_wilcoxon      "$R/P5 Real Life Datasets V2" "$RS" P4_vs_P5_Wilcoxon.R
  step R_P6a_marker_synthetic   "$R/P6a Marker - Synthetic"   "$RS" Plots_Analysis.R
  step R_P6b_marker_real        "$R/P6b Marker - Real"        "$RS" Plots_Analysis.R
  step R_P7a_database_markers   "$R/P7a Marker - Real Database" "$RS" Plots_Analysis.R
  step R_P7b_llm                "$R/P7b LLM - Real Internet"  "$RS" Plots_Analysis.R
  step R_P7b_same_family        "$R/P7b LLM - Real Internet"  "$RS" sensitivity_same_family.R
  # Not run: results/P8a (foundation models on the synthetic data) is not reported in the paper,
  # and its script has failed on its own data table since a June 2026 edit (see README.md).
  step R_P8b_fm_real            "$R/P8b FM - Real"            "$RS" Plots_Analysis.R
  step R_P9_cross_platform      "$R/P9 Cross Dataset"         "$RS" Plots_Analysis.R
}

numbers() {
  local IA="$B/results/_interaction_analysis"
  step tool_interaction_analysis   "$NOTES" "$PY" tool_interaction_analysis.py
  step paradigm_level_concordance  "$NOTES" "$PY" paradigm_level_concordance.py
  step type3_eta_squared           "$NOTES" "$PY" type3_eta_squared.py
  step type3_eta_runtime_memory    "$NOTES" "$PY" type3_eta_runtime_memory.py
  step dataset_level_r2            "$NOTES" "$PY" redo_all_with_corrected_profile.py
  step results_numbers             "$NOTES" "$PY" compute_results_numbers.py
  step phase4_5_numbers            "$NOTES" "$PY" compute_phase4_5_numbers.py
  step fm_vs_purity                "$NOTES" "$PY" fm_with_profiling_real.py
  step multimetric_robustness      "$NOTES/v33_analysis" "$PY" v33_multimetric.py "$B" "$LOG"
  step tidy_kappa_metrics          "$NOTES/v33_analysis" "$PY" make_tidy_metrics.py "$B" "$LOG" "$FIG/_shared/tidy_kappa.csv"
  step fig3a_purity_sensitivity    "$NOTES/S3_S4_audit" "$PY" fit_sensitivity.py "$B" "$LOG"
  step discussion_hard_datasets    "$NOTES/S3_S4_audit" "$PY" p57_recompute.py "$B" "$LOG"
  step tidy_crossplatform          "$FIG/_shared" "$PY" build_tidy_crossplatform.py
  step tidy_tables_from_raw        "$FIG/_shared" "$PY" build_tidy_tables.py
}

figures() {
  local OUT="$B/reproduce_figures" S="$FIG/supplementary"
  for n in 1 2 3 4; do
    step "Figure$n" "$FIG/Figure$n" "$PY" "$HERE/render_figure.py" "$FIG/Figure$n/build_fig$n.py" "$OUT/Figure$n"
  done
  for n in 01 02 04 06 07 08 10 11 14; do
    step "SuppFig_S$n" "$S/S$n" "$PY" "$HERE/render_figure.py" "$S/S$n/build_s$n.py" "$OUT/SuppFig_S$n"
  done
  # These three write into their own S03/S05/S09/S12/S13 folders beside the scripts.
  mkdir -p "$S/S03" "$S/S05" "$S/S09" "$S/S12" "$S/S13"
  step SuppFig_S05_eta     "$S" "$PY" "$HERE/render_figure.py" "$S/_build_eta_figs.py"     "$OUT/_eta"
  step SuppFig_S03_S09_S13 "$S" "$PY" "$HERE/render_figure.py" "$S/_build_misc_figs.py"    "$OUT/_misc"
  step SuppFig_S12_runtime "$S" "$PY" "$HERE/render_figure.py" "$S/_build_runtime_figs.py" "$OUT/_runtime"
}

case "$WHAT" in
  analyses) analyses ;;
  numbers)  numbers ;;
  figures)  figures ;;
  all)      analyses; numbers; figures ;;
  *) echo "unknown step: $WHAT"; exit 2 ;;
esac

echo
printf '%s\n' "${STATUS[@]}" | tee "$LOG/SUMMARY.txt"
! printf '%s\n' "${STATUS[@]}" | grep -q '^FAILED'
