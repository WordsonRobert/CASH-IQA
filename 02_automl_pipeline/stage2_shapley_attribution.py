#!/usr/bin/env python3
"""
Stage 2 -- exact Shapley attribution of the encoders.

For every non-empty encoder subset S (2^E - 1 = 63 for 6 encoders) and every outer split, the full
CASH search (stage2.strategy, same nested protocol, weights restricted to S) is re-run, and the
selected model is scored once on that split's test set. v(S) = test SROCC, v(empty) = 0.
Shapley values / interactions are computed exactly in stage4_analysis.py from these v(S).

One job = (dataset, split); subsets are appended to a JSONL file as they finish (resumable).
    python run_jobs.py --stage 2
"""
import argparse, json, sys, itertools, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import *


def subsets(encoders):
    E = list(encoders)
    return [list(c) for r in range(1, len(E) + 1) for c in itertools.combinations(E, r)]


def list_jobs(cfg):
    return [{"ds": ds, "split": k} for ds in cfg["datasets"] for k in cfg["stage2"]["outer_splits"]]


def out_file(cfg, j):
    return Path(cfg["paths"]["results"]) / "stage2_shapley" / j["ds"] / f"split{j['split']:02d}.jsonl"


def done_subsets(cfg, j):
    f = out_file(cfg, j)
    if not f.exists(): return set()
    out = set()
    for l in f.read_text().splitlines():
        try: out.add("+".join(json.loads(l)["subset"]))
        except Exception: pass                      # tolerate a truncated last line after a crash
    return out


def is_done(cfg, j):
    return len(done_subsets(cfg, j)) == len(subsets(cfg["encoders"]))


def run_job(cfg, j):
    dev = device()   # threads via OMP_NUM_THREADS (set by run_jobs.py); torch.set_num_threads() triggers an MKL bug
    S2 = cfg["stage2"]; done = done_subsets(cfg, j)
    meta, feats = load_features(cfg, j["ds"], cfg["encoders"], S2["featureset"])
    sp = load_splits(cfg, j["ds"]); y = meta["q"].values
    bank = Bank(feats, cfg["encoders"], dev); del feats
    f = out_file(cfg, j); f.parent.mkdir(parents=True, exist_ok=True)
    for sub in subsets(cfg["encoders"]):
        key = "+".join(sub)
        if key in done: continue
        plan = nested_plan(bank, y, sp["nested"][j["split"]], cfg["protocol"]["n_inner_folds"])
        plan.chunk = cfg["runtime"]["local_chunk_elems"]
        seed = job_seed(cfg["protocol"]["seed"], "shapley", j["ds"], j["split"], key)
        led, secs = search(S2["strategy"], Space(cfg, sub), plan, S2["budget_fold_evals"], seed)
        plan.freeze(); inner, best = led.best
        _, m = plan.final(best)
        rec = {"subset": sub, "inner": inner, "test": m, "selected": best, "fold_evals": plan.n_fold_evals,
               "seconds": secs}
        with open(f, "a") as fh:
            fh.write(json.dumps(rec, default=float) + "\n")
        print(f"[{j['ds']} s{j['split']}] {key:<45} inner={inner:.4f} test={m['srocc']:.4f} {secs:.0f}s", flush=True)


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
