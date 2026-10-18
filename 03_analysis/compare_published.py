#!/usr/bin/env python3
"""
Our numbers next to published ones (CPU, instant).  python compare_published.py
Published values come from published_baselines.csv (each row cites its source; re-verify against the PDFs before
submission). Ours come from results/tables/headline_{nested,holdout}.csv (median SROCC over 10 splits, TPE+Hyperband).
Protocols differ between papers -- compare published 70/10/20 rows with our HOLDOUT row.
Writes results/tables/comparison_published.csv.
"""
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; T = ROOT / "results" / "tables"
DS = ["kadid10k", "csiq", "tid2013", "koniq10k", "livec", "spaq"]
pub = pd.read_csv(Path(__file__).with_name("published_baselines.csv"))
rows = []
for prot in ["holdout", "nested"]:
    h = pd.read_csv(T / f"headline_{prot}.csv"); h = h[h.method == "tpe_hb"].set_index("ds")["srocc_median"]
    rows.append({"group": "ours", "method": f"Ours ({prot})", "venue": "this work", "setting": "frozen encoders + CASH",
                 "protocol": "70/10/20" if prot == "holdout" else "nested 10x80/20, 5 inner folds", "aggregation": "median of 10",
                 **{d: round(float(h[d]), 3) for d in DS}})
out = pd.concat([pub, pd.DataFrame(rows)], ignore_index=True)
order = {"frozen_encoder": 0, "ours": 1, "label_retrieval": 2, "no_labels": 3, "fine_tuned_ceiling": 4}
out = out.sort_values("group", key=lambda s: s.map(order), kind="stable")
out.to_csv(T / "comparison_published.csv", index=False)
pd.set_option("display.width", 220)
print(out[["group", "method", "aggregation"] + DS].to_string(index=False))
fz = out[out.group == "frozen_encoder"]
print("\nbest published frozen-encoder SROCC per dataset vs ours (nested / holdout):")
for d in DS:
    b = fz[d].max(); who = fz.loc[fz[d].idxmax(), "method"] if fz[d].notna().any() else "-"
    o = out[out.method == "Ours (nested)"][d].iloc[0]; oh = out[out.method == "Ours (holdout)"][d].iloc[0]
    print(f"  {d:<9} best published {b:.3f} ({who})  ours {o:.3f} / {oh:.3f}")
