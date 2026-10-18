#!/usr/bin/env bash
# Full feature extraction (6 datasets x 6 encoders) + fidelity check. Resumable: rerun after any interruption.
#   bash run_extraction.sh                      # everything -> ../features/{dataset}/{encoder}.h5
#   bash run_extraction.sh --encoders arniqa    # extra args are passed to extract_features.py
# Images are read from $IQA_DATA_ROOT (default ../data); see README.md for the expected layout.
set -euo pipefail
cd "$(dirname "$0")"
OUT="$(cd .. && pwd)/features"
mkdir -p "$OUT/logs"
LOG="$OUT/logs/extract_$(date +%Y%m%d_%H%M%S).log"
echo "log: $LOG"
python extract_features.py --out "$OUT" "$@" 2>&1 | tee -a "$LOG"
python verify_features.py --feat "$OUT" 2>&1 | tee -a "$LOG"
echo "DONE -> $OUT/verify_linear_probe.csv and $OUT/manifest.json"
