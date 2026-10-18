#!/usr/bin/env bash
# The complete pipeline as it was run for the paper (GPU required; ~12 h on one A100-80GB).
# Resumable: every job writes its own output and finished jobs are skipped, so just rerun after any interruption.
#   bash run_all.sh                 # all visible GPUs
#   bash run_all.sh --gpus 0,1      # extra args go to run_jobs.py
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-$(command -v python || command -v python3)}
"$PY" stage0_prepare.py                     # checks features, freezes splits (refuses to overwrite), benchmarks, projects hours
"$PY" run_jobs.py --stage 1 "$@"            # AutoML/CASH benchmark + headline numbers + feature ablations
"$PY" run_jobs.py --stage 2 "$@"            # exact Shapley: all 63 encoder subsets, re-tuned
"$PY" run_jobs.py --stage 3 "$@"            # cross-dataset: tune on one dataset, test on the other five
"$PY" run_jobs.py --stage 4 "$@"            # stage 3b: transfer decomposition (global / equal weights)
"$PY" stage3b_summary.py                    # -> ../results/tables/cross_transfer.csv
"$PY" stage4_analysis.py                    # -> ../results/tables/*.csv, ../results/report_stage4.md
"$PY" ../03_analysis/sanity_checks.py       # integrity checks; exits 1 on any problem
"$PY" ../03_analysis/shapley_fusion_gain.py # -> re-baselined Shapley values / interactions with CIs
"$PY" ../03_analysis/compare_published.py  # -> ../results/tables/comparison_published.csv
"$PY" ../03_analysis/make_figures.py        # -> ../figures/*.pdf|png
echo "ALL DONE"
