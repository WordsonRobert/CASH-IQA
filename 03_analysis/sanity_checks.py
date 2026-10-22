#!/usr/bin/env python3
"""
Integrity checks on the raw results (CPU, seconds). These are the checks run on the reported run before
any table was trusted; they print PROBLEMS: none when everything is in order.

  1. completeness: 10 splits per (featureset, protocol, dataset) in stage 1; 10 x 63 subsets in stage 2;
     6 sources in stage 3; 3 variants x 6 sources in stage 3b
  2. no NaN / zero scores anywhere
  3. budget accounting: every stage-2 search used exactly its fold-evaluation budget
  4. stage-2 budget adequacy: all-six-encoder subset at budget 800 vs stage-1 TPE+Hyperband at 1500 (same splits)
  5. selection sanity: inner-CV vs test gap of the selected model (nested protocol)
  6. stage 5 (attribution without re-tuning): 10 splits x 63 subsets per dataset; the full encoder set must
     reproduce the stage-1 TPE+Hyperband test and inner scores exactly
  7. stage 6 (transfer shrinkage): 6 sources + 60 in-domain jobs; alpha = 0 must reproduce stage 3 (cross) and
     stage 1 (in-domain) exactly

    python sanity_checks.py
"""
import json, glob, math, collections, sys
from pathlib import Path
import numpy as np, yaml

ROOT = Path(__file__).resolve().parents[1]; R = ROOT / "results"
cfg = yaml.safe_load(open(ROOT / "02_automl_pipeline" / "config.yaml"))
DS, E = cfg["datasets"], cfg["encoders"]
problems = []


def bad(x):
    return (not isinstance(x, (int, float))) or (not math.isfinite(x)) or x == 0.0


print("=" * 100, "\n1) STAGE 1 completeness (splits with every expected file)")
for fs, prot, parts in [("main", "nested", 5), ("main", "holdout", 5), ("canon", "nested", 2), ("arniqa_proj", "nested", 2)]:
    line = []
    for ds in DS:
        dirs = sorted(glob.glob(str(R / "stage1_automl" / fs / prot / ds / "split*")))
        ok = 0
        for d in dirs:
            files = glob.glob(d + "/*.json")
            if len(files) == parts: ok += 1
            for fpath in files:
                o = json.load(open(fpath))
                vals = [v["test"]["srocc"] for v in o["baselines"].values()] if "baselines" in o else [o["test"]["srocc"]]
                if any(bad(v) for v in vals): problems.append(f"stage1 {fpath}: bad score")
        line.append(f"{ds}:{ok}/10"); ok < 10 and problems.append(f"stage1 {fs}/{prot}/{ds}: {ok}/10 splits")
    print(f"  {fs:<12}{prot:<8}", "  ".join(line))

print("=" * 100, "\n2) STAGE 2 completeness + budget (expect 10 splits x 63 subsets, fold_evals == budget)")
S2 = {}
for ds in DS:
    S2[ds] = {}
    for fn in sorted(glob.glob(str(R / "stage2_shapley" / ds / "split*.jsonl"))):
        recs = {}
        for line in open(fn):
            r = json.loads(line); recs[frozenset(r["subset"])] = r
            if bad(r["test"]["srocc"]): problems.append(f"stage2 {fn}: bad score")
        k = int(Path(fn).stem[-2:]); S2[ds][k] = recs
        if len(recs) != 2 ** len(E) - 1: problems.append(f"stage2 {ds} split{k}: {len(recs)} subsets")
    fe = [r["fold_evals"] for recs in S2[ds].values() for r in recs.values()]
    if fe and set(fe) != {cfg["stage2"]["budget_fold_evals"]}: problems.append(f"stage2 {ds}: budgets {sorted(set(fe))}")
    print(f"  {ds:<9} splits {len(S2[ds]):>2} | subsets/split {sorted(set(len(x) for x in S2[ds].values()))} | "
          f"fold_evals {min(fe) if fe else 0}-{max(fe) if fe else 0}")

print("=" * 100, "\n3) STAGE 2 budget adequacy: all-encoder subset (budget 800) vs stage 1 TPE+HB (budget 1500)")
for ds in DS:
    d = []
    for k, recs in S2[ds].items():
        f1 = R / "stage1_automl" / "main" / "nested" / ds / f"split{k:02d}" / "tpe_hb.json"
        if frozenset(E) in recs and f1.exists():
            d.append(recs[frozenset(E)]["test"]["srocc"] - json.load(open(f1))["test"]["srocc"])
    if d:
        print(f"  {ds:<9} median paired difference {np.median(d):+.4f}  (|diff| <= ~0.004 -> budget adequate)")

print("=" * 100, "\n4) STAGE 3 / 3b completeness")
n3 = len(glob.glob(str(R / "stage3_cross_dataset" / "*.json")))
n3b = len(glob.glob(str(R / "stage3b_transfer_variants" / "*" / "*.json")))
print(f"  stage 3: {n3}/6 sources   stage 3b: {n3b}/18 (3 variants x 6 sources)")
n3 < 6 and problems.append("stage3 incomplete"); n3b < 18 and problems.append("stage3b incomplete")

print("=" * 100, "\n5) Selection sanity (nested, TPE+HB): median inner-CV minus test SROCC of the selected model")
for ds in DS:
    gap = []
    for fpath in glob.glob(str(R / "stage1_automl" / "main" / "nested" / ds / "split*" / "tpe_hb.json")):
        o = json.load(open(fpath)); gap.append(o["inner"] - o["test"]["srocc"])
    print(f"  {ds:<9} {np.median(gap):+.4f}")

print("=" * 100, "\n6) STAGE 5 completeness + reproduction of stage 1 by the full encoder set (no re-tuning)")
TOL = 1e-6
for ds in DS:
    fs = sorted(glob.glob(str(R / "stage5_noretune_shapley" / ds / "split*.json"))); dmax = 0.0
    for fpath in fs:
        o = json.load(open(fpath))
        if len(o["records"]) != 2 ** len(E) - 1: problems.append(f"stage5 {fpath}: {len(o['records'])} subsets")
        if any(bad(r["test"]["srocc"]) for r in o["records"]): problems.append(f"stage5 {fpath}: bad score")
        dmax = max(dmax, abs(o["check_full_test_diff"]), abs(o["check_full_inner_diff"]))
    print(f"  {ds:<9} splits {len(fs):>2}/10 | max |full-set diff vs stage 1| {dmax:.1e}")
    len(fs) < 10 and problems.append(f"stage5 {ds}: {len(fs)}/10 splits")
    dmax > TOL and problems.append(f"stage5 {ds}: full set does not reproduce stage 1 ({dmax:.1e})")

print("=" * 100, "\n7) STAGE 6 completeness + reproduction at alpha = 0 (stage 3 cross, stage 1 in-domain)")
S6 = R / "stage6_transfer_shrinkage"
nc = sorted(glob.glob(str(S6 / "cross" / "*.json"))); ni = sorted(glob.glob(str(S6 / "indomain" / "*" / "split*.json")))
dc = max([abs(x) for f in nc for x in json.load(open(f))["checks"].values()] or [0.0])
di = max([abs(json.load(open(f))["check_alpha0_test_diff"]) for f in ni] or [0.0])
print(f"  cross {len(nc)}/6 (max |alpha=0 - stage 3| {dc:.1e})   in-domain {len(ni)}/60 (max |alpha=0 - stage 1| {di:.1e})")
len(nc) < 6 and problems.append("stage6 cross incomplete"); len(ni) < 60 and problems.append("stage6 in-domain incomplete")
max(dc, di) > TOL and problems.append(f"stage6: alpha=0 does not reproduce stages 1/3 ({max(dc, di):.1e})")

print("=" * 100)
print("PROBLEMS:", problems[:20] if problems else "none")
sys.exit(1 if problems else 0)
