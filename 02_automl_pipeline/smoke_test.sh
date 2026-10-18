#!/usr/bin/env bash
# 5-10 minute end-to-end check of the whole pipeline on the two smallest datasets with tiny budgets.
# Uses its own _smoke/ folder, so it never touches results/.
#   bash smoke_test.sh                 # features from ../features
#   bash smoke_test.sh /path/to/features
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
FEAT=${1:-$(cd "$HERE/.." && pwd)/features}
PY=${PY:-$(command -v python || command -v python3)}
OUT=${2:-$HERE/_smoke}; mkdir -p "$OUT"
"$PY" - "$HERE/config.yaml" "$OUT/config_smoke.yaml" "$FEAT" "$OUT" <<'PYEOF'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
c["paths"] = {"features": sys.argv[3], "work": sys.argv[4] + "/work", "results": sys.argv[4] + "/results"}
c["datasets"] = ["csiq", "livec"]
c["protocol"].update(n_outer=2, n_holdout=2)
c["stage1"].update(budget_fold_evals=60, holdout_budget_evals=15, anytime_points=[0.25, 1.0])
c["stage2"].update(budget_fold_evals=10, outer_splits=[0])
c["stage3"].update(budget_fold_evals=30)
yaml.safe_dump(c, open(sys.argv[2], "w"))
PYEOF
CFG="$OUT/config_smoke.yaml"
"$PY" "$HERE/stage0_prepare.py" --config "$CFG"
for s in 1 2 3 4; do "$PY" "$HERE/run_jobs.py" --stage $s --config "$CFG"; done
CASHIQA_CONFIG="$CFG" "$PY" "$HERE/stage3b_summary.py"
"$PY" "$HERE/stage4_analysis.py" --config "$CFG"
echo "SMOKE TEST OK -> $OUT/results/report_stage4.md"
