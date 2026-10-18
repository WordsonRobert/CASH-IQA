#!/usr/bin/env python3
"""Compare stage 3 (full CASH, ridge_concat) with the stage 3b transfer variants. Writes results/tables/cross_transfer.csv.  python stage3b_summary.py"""
import json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import load_config

cfg = load_config(None); R = Path(cfg["paths"]["results"]); SYN = {"csiq", "kadid10k", "tid2013"}
rows = []
for f in (R / "stage3_cross_dataset").glob("*.json"):
    d = json.loads(f.read_text())
    for t, r in d["targets"].items():
        for m in ["automl", "ridge_concat"]:
            rows.append(dict(src=d["job"]["src"], tgt=t, method="full" if m == "automl" else m, srocc=r[m]["srocc"]))
for f in (R / "stage3b_transfer_variants").glob("*/*.json"):
    d = json.loads(f.read_text())
    for t, r in d["targets"].items():
        rows.append(dict(src=d["job"]["src"], tgt=t, method=d["job"]["variant"], srocc=r["srocc"]))
df = pd.DataFrame(rows)
df["pair"] = [("S" if s in SYN else "A") + "->" + ("S" if t in SYN else "A") for s, t in zip(df.src, df.tgt)]
(R / "tables").mkdir(parents=True, exist_ok=True); df.to_csv(R / "tables" / "cross_transfer.csv", index=False)
order = [m for m in ["full", "global", "eqw", "global_eqw", "ridge_concat"] if m in set(df.method)]
pd.set_option("display.width", 200)
print("mean cross-dataset SROCC by domain pair (S=synthetic source/target, A=authentic):")
t = df.pivot_table(index="method", columns="pair", values="srocc", aggfunc="mean").reindex(order)
t["ALL"] = df.groupby("method").srocc.mean().reindex(order); print(t.round(3).to_string())
print("\nper source (mean over its 5 targets):")
print(df.pivot_table(index="method", columns="src", values="srocc", aggfunc="mean").reindex(order).round(3).to_string())
n = df.groupby(["method"]).size().to_dict(); print("\npairs per method:", n, "(expect 30 each)")
