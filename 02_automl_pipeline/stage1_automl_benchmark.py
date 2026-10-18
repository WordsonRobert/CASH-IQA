#!/usr/bin/env python3
"""
Stage 1 -- AutoML (CASH) benchmark + headline numbers.

One job = (featureset, protocol, dataset, split). Inside a job the dataset's Gram matrices are
built once and reused by every strategy and baseline.

  protocol 'nested'  : search on 5 grouped inner folds of the 80% outer-train part; the selected
                       configuration is refit on the full outer-train gallery and scored ONCE on test.
  protocol 'holdout' : 70/10/20 -- search scored on the 10% val part (gallery = 70% train); the selected
                       configuration is scored ONCE on test with gallery = train+val.

Per strategy (random, lhs_sh, tpe, tpe_hb) the same fold-evaluation budget is enforced. The
incumbent trajectory is logged during the search; test scores of past incumbents ("anytime" curve)
are computed only AFTER the search is frozen and are never fed back.

Outputs  results/stage1_automl/{featureset}/{protocol}/{ds}/split{k}/{strategy|baselines}.json + preds.npz
    python stage1_automl_benchmark.py --list                       # print job ids
    python stage1_automl_benchmark.py --job '{"fs":"main","protocol":"nested","ds":"csiq","split":0}'
    python run_jobs.py --stage 1                     # all jobs, parallel over GPUs
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import *


def list_jobs(cfg):
    P, S = cfg["protocol"], cfg["stage1"]; jobs = []
    for ds in cfg["datasets"]:
        for k in range(P["n_outer"]):
            jobs.append({"fs": "main", "protocol": "nested", "ds": ds, "split": k})
        for k in range(P["n_holdout"]):
            jobs.append({"fs": "main", "protocol": "holdout", "ds": ds, "split": k})
        for fs in S["ablation_featuresets"]:
            for k in range(P["n_outer"]):
                jobs.append({"fs": fs, "protocol": "nested", "ds": ds, "split": k})
    return jobs


def job_dir(cfg, j):
    return Path(cfg["paths"]["results"]) / "stage1_automl" / j["fs"] / j["protocol"] / j["ds"] / f"split{j['split']:02d}"


def expected_parts(cfg, j):
    st = cfg["stage1"]["strategies"] if j["fs"] == "main" else [cfg["stage1"]["ablation_strategy"]]
    return [f"{s}.json" for s in st] + ["baselines.json"]


def is_done(cfg, j):
    d = job_dir(cfg, j); return all((d / p).exists() for p in expected_parts(cfg, j))


def make_plan(cfg, bank, y, sp, j):
    if j["protocol"] == "nested":
        return nested_plan(bank, y, sp["nested"][j["split"]], cfg["protocol"]["n_inner_folds"])
    return holdout_plan(bank, y, sp["holdout"][j["split"]], cfg["protocol"]["holdout_refit_on_train_val"])


def run_job(cfg, j):
    dev = device()   # threads via OMP_NUM_THREADS (set by run_jobs.py); torch.set_num_threads() triggers an MKL bug
    out = job_dir(cfg, j); out.mkdir(parents=True, exist_ok=True)
    meta, feats = load_features(cfg, j["ds"], cfg["encoders"], j["fs"])
    sp = load_splits(cfg, j["ds"]); y = meta["q"].values
    bank = Bank(feats, cfg["encoders"], dev); del feats
    strategies = cfg["stage1"]["strategies"] if j["fs"] == "main" else [cfg["stage1"]["ablation_strategy"]]
    budget = cfg["stage1"]["budget_fold_evals"] if j["protocol"] == "nested" else cfg["stage1"]["holdout_budget_evals"]
    space = Space(cfg, cfg["encoders"])
    preds = {}
    for si, strat in enumerate(strategies):
        f = out / f"{strat}.json"
        if f.exists():
            continue
        plan = make_plan(cfg, bank, y, sp, j); plan.chunk = cfg["runtime"]["local_chunk_elems"]
        seed = job_seed(cfg["protocol"]["seed"], j["fs"], j["protocol"], j["ds"], j["split"], strat)
        led, secs = search(strat, space, plan, budget, seed)
        plan.freeze()                                                     # ---- no more search after this line
        inner, best = led.best
        p, m = plan.final(best)
        preds[strat] = p.astype(np.float32)
        anytime, cache = [], {}
        for frac in cfg["stage1"]["anytime_points"]:
            inc = [t for t in led.traj if t["fold_evals"] <= frac * budget]
            if not inc: continue
            c = inc[-1]["cfg"]; k = Ledger.key(c)
            if k not in cache: cache[k] = plan.final(c)[1]
            anytime.append({"frac": frac, "fold_evals": inc[-1]["fold_evals"], "seconds": inc[-1]["seconds"],
                            "inner": inc[-1]["inner"], "test": cache[k]})
        fam = {}
        if strat == "random":                                             # per-family model-selection table
            for fa, (s, c) in per_family_best(led, space.fam).items():
                fam[fa] = {"inner": s, "cfg": c, "test": plan.final(c)[1]}
        atomic_json(f, {"job": j, "strategy": strat, "budget": budget, "fold_evals_used": plan.n_fold_evals,
                        "n_full_evals": led.n_full, "search_seconds": secs, "eval_seconds": plan.t_eval,
                        "selected": best, "inner": inner, "test": m, "trajectory": led.traj, "anytime": anytime,
                        "per_family": fam, "n_test": len(plan._final[0]), "env": env_info()})
        print(f"[{j['ds']} {j['protocol']} s{j['split']} {j['fs']}] {strat:<7} inner={inner:.4f} test={m['srocc']:.4f} "
              f"fam={best['family']} evals={plan.n_fold_evals} {secs:.0f}s", flush=True)
    f = out / "baselines.json"
    if not f.exists():
        plan = make_plan(cfg, bank, y, sp, j); plan.chunk = cfg["runtime"]["local_chunk_elems"]
        t = time.time(); chosen = run_baselines(plan, cfg["encoders"]); plan.freeze()
        res = {}
        for name, (s, c) in chosen.items():
            p, m = plan.final(c); res[name] = {"inner": s, "cfg": c, "test": m}; preds["base_" + name] = p.astype(np.float32)
        atomic_json(f, {"job": j, "baselines": res, "seconds": time.time() - t, "env": env_info()})
        print(f"[{j['ds']} {j['protocol']} s{j['split']} {j['fs']}] baselines " +
              " ".join(f"{k}={v['test']['srocc']:.3f}" for k, v in res.items()), flush=True)
    if preds:
        te = np.array(make_plan(cfg, bank, y, sp, j)._final[0])
        old = dict(np.load(out / "preds.npz")) if (out / "preds.npz").exists() else {}
        old.update(preds); old["test_idx"] = te; old["y"] = y[te].astype(np.float32)
        np.savez_compressed(out / "preds.npz", **old)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--job", default=None)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(); cfg = load_config(a.config)
    if a.list:
        for j in list_jobs(cfg):
            print(json.dumps(j), "DONE" if is_done(cfg, j) else "")
        return
    run_job(cfg, json.loads(a.job))


if __name__ == "__main__":
    main()
