#!/usr/bin/env bash
# Progress of a running (or finished) pipeline.   Live view:  watch -n 60 bash status.sh
cd "$(dirname "$0")"
PY=${PY:-$(command -v python || command -v python3)}
pgrep -f "run_jobs.py" >/dev/null && echo "RUN: ACTIVE" || echo "RUN: not running"
"$PY" - <<'PYEOF' 2>/dev/null
from core import load_config
import stage1_automl_benchmark as s1, stage2_shapley_attribution as s2, stage3_cross_dataset as s3, stage3b_transfer_variants as s3b
import stage5_noretune_shapley as s5, stage6_transfer_shrinkage as s6
cfg = load_config("config.yaml")
for n, m in [("1 automl", s1), ("2 shapley", s2), ("3 cross-dataset", s3), ("3b transfer", s3b), ("5 no re-tuning", s5), ("6 shrinkage", s6)]:
    J = m.list_jobs(cfg); print(f"  stage {n:<16} {sum(bool(m.is_done(cfg, j)) for j in J):>4}/{len(J)} jobs done")
PYEOF
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || true
