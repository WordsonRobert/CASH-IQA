#!/usr/bin/env python3
"""
Stage 5 -- attribution without re-tuning (a cheap approximation of the stage-2 Shapley game).

Stage 2 re-runs the full CASH search for every encoder subset S (3,780 searches). Here nothing is re-tuned:
for each (dataset, outer split) the stage-1 configuration selected by tpe_hb on the nested protocol is kept
fixed (family, K, gamma, lambda, centring and the weights of the encoders in S), and the weights of the
encoders outside S are set to zero. The fused kernel renormalises over S automatically (Bank.wvec).

    v_fixed(S)  = test SROCC of that restricted configuration      (one evaluation per subset, no search)
    v_inner(S)  = its mean SROCC over the same 5 inner folds        (needs no test data at all)

All inner scores are computed before the plan is frozen; test scores only after. No configuration is chosen
on test data (there is no selection step at all), so the test game is a legitimate evaluation.

Built-in check: for S = all encoders, v_fixed(S) must reproduce the stage-1 tpe_hb test SROCC of the split.

Pre-registered reading (03_analysis/noretune_shapley_check.py): the cheap game is reported as agreeing with
the exact game when the Spearman rank correlation of the six fusion-gain Shapley values is >= 0.9 on at least
5 of the 6 datasets. The result is reported either way.

One job = (dataset, split).   python run_jobs.py --stage 5
Output: results/stage5_noretune_shapley/{ds}/split{k:02d}.json
"""
import argparse, json, sys, time, itertools
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import *


def subsets(encoders):
    E = list(encoders)
    return [list(c) for r in range(1, len(E) + 1) for c in itertools.combinations(E, r)]


def list_jobs(cfg):
    return [{"ds": ds, "split": k} for ds in cfg["datasets"] for k in range(cfg["protocol"]["n_outer"])]


def out_file(cfg, j):
    return Path(cfg["paths"]["results"]) / "stage5_noretune_shapley" / j["ds"] / f"split{j['split']:02d}.json"


def is_done(cfg, j):
    return out_file(cfg, j).exists()


def stage1_file(cfg, ds, split, strategy="tpe_hb"):
    return Path(cfg["paths"]["results"]) / "stage1_automl" / "main" / "nested" / ds / f"split{split:02d}" / f"{strategy}.json"


def restrict(base, sub, encoders):
    c = json.loads(json.dumps(base))
    c["w"] = {e: (float(base["w"][e]) if e in sub else 0.0) for e in encoders}
    return c


@torch.no_grad()
def run_job(cfg, j):
    dev = device(); E = cfg["encoders"]; t0 = time.time()
    s1 = json.loads(stage1_file(cfg, j["ds"], j["split"]).read_text())
    base, s1_test, s1_inner = s1["selected"], s1["test"]["srocc"], s1["inner"]
    meta, feats = load_features(cfg, j["ds"], E, "main")
    sp = load_splits(cfg, j["ds"]); y = meta["q"].values
    bank = Bank(feats, E, dev); del feats
    plan = nested_plan(bank, y, sp["nested"][j["split"]], cfg["protocol"]["n_inner_folds"])
    plan.chunk = cfg["runtime"]["local_chunk_elems"]; nf = len(plan.folds)
    recs = []
    for sub in subsets(E):                                   # ---- inner scores (search folds only)
        c = restrict(base, sub, E)
        folds = [plan.eval_fold(c, f) for f in range(nf)]
        recs.append({"subset": sub, "cfg": c, "inner": float(np.mean(folds)), "inner_folds": folds})
    plan.freeze()                                             # ---- test only after this line
    for r in recs:
        _, r["test"] = plan.final(r["cfg"])
    full = recs[-1]
    res = {"job": j, "base_cfg": base, "base_family": base["family"],
           "stage1_test_srocc": s1_test, "stage1_inner": s1_inner,
           "check_full_test_diff": float(full["test"]["srocc"] - s1_test),
           "check_full_inner_diff": float(full["inner"] - s1_inner),
           "n_fold_evals": plan.n_fold_evals, "seconds": time.time() - t0,
           "records": recs, "env": env_info()}
    atomic_json(out_file(cfg, j), res)
    print(f"[{j['ds']} s{j['split']}] fam={base['family']} full: test={full['test']['srocc']:.4f} "
          f"(stage1 {s1_test:.4f}, diff {res['check_full_test_diff']:+.1e}) {res['seconds']:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None); ap.add_argument("--job", default=None); ap.add_argument("--list", action="store_true")
    a = ap.parse_args(); cfg = load_config(a.config)
    if a.list:
        for j in list_jobs(cfg): print(json.dumps(j), "DONE" if is_done(cfg, j) else "")
        return
    run_job(cfg, json.loads(a.job))


if __name__ == "__main__":
    main()
