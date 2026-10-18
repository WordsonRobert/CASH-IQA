#!/usr/bin/env python3
"""
Stage 0 -- prepare + benchmark + time projection  (run once, ~5-15 min on one GPU)

  1. validates every feature file (key order, dims, finiteness, all featureset variants present)
  2. builds and freezes ALL splits (nested 10x80/20 with 5 grouped inner folds, 10x70/10/20 holdout,
     5-fold grouped CV over the full set for cross-dataset) -> work/{ds}/splits.json
     (re-running regenerates and checks they are bit-identical)
  3. benchmarks the per-fold evaluation time of every regressor family on every dataset and the
     Optuna overhead, then projects GPU-hours and wall-clock for stages 1-3 under config.yaml
     -> work/plan.json

    python stage0_prepare.py              # all of the above
    python stage0_prepare.py --no-bench   # splits + checks only
"""
import argparse, json, time, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import *


def check_features(cfg, ds):
    import h5py
    meta = load_meta(cfg, ds); need = {}
    for fs, sel in cfg["featuresets"].items():
        for e in cfg["encoders"]:
            need.setdefault(e, set()).add((sel or {}).get(e, "main"))
    dims = {}
    for e, vs in need.items():
        p = Path(cfg["paths"]["features"]) / ds / f"{e}.h5"
        with h5py.File(p, "r") as f:
            keys = [k.decode() for k in f["keys"][:]]
            assert keys == meta["key"].tolist(), f"{p}: key order mismatch"
            for v in vs:
                assert v in f, f"{p}: missing variant '{v}'"
                X = f[v][:]
                assert X.shape[0] == len(meta) and np.isfinite(X).all(), f"{p}:{v} bad values"
                dims[f"{e}:{v}"] = int(X.shape[1])
    assert np.isfinite(meta["q"]).all()
    return meta, dims


def bench_dataset(cfg, ds, dev, n_per_family=6):
    meta, feats = load_features(cfg, ds, cfg["encoders"], "main")
    sp = load_splits(cfg, ds)
    torch.cuda.synchronize() if dev == "cuda" else None
    t = time.time(); bank = Bank(feats, cfg["encoders"], dev); t_bank = time.time() - t
    plan = nested_plan(bank, meta["q"].values, sp["nested"][0], cfg["protocol"]["n_inner_folds"])
    space = Space(cfg, cfg["encoders"]); rng = np.random.RandomState(0)
    res = {}
    for fam in cfg["search_space"]["families"]:
        sp1 = Space(cfg, cfg["encoders"], [fam]); ts = []
        for i in range(n_per_family + 2):
            c = sp1.from_unit(rng.uniform(size=sp1.dim))
            if fam in FAMILIES_LOCAL: c["K"] = int(rng.choice([20, 60, 120, 200]))
            if dev == "cuda": torch.cuda.synchronize()
            t = time.perf_counter(); plan.eval_fold(c, i % len(plan.folds))
            if dev == "cuda": torch.cuda.synchronize()
            if i >= 2: ts.append(time.perf_counter() - t)
        res[fam] = float(np.median(ts))
    peak = torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0
    del bank, plan; torch.cuda.empty_cache() if dev == "cuda" else None
    return {"n": len(meta), "bank_s": t_bank, "fold_eval_s": res, "fold_eval_mean_s": float(np.mean(list(res.values()))),
            "peak_gb": peak}


def optuna_overhead(n=300):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0, multivariate=True, group=True))
    t = time.time()
    for _ in range(n):
        tr = st.ask(); fam = tr.suggest_categorical("family", ["a", "b", "c", "d", "e", "f"])
        x = sum(tr.suggest_float(f"w{i}", 0, 1) for i in range(6))
        if fam in "abcd": x += tr.suggest_int("K", 5, 200, log=True) / 200
        x += tr.suggest_float("g", -2, 2) + tr.suggest_float("l", -4, 2)
        st.tell(tr, x)
    return (time.time() - t) / n


def project(cfg, bench, t_opt):
    P, S1, S2, S3 = cfg["protocol"], cfg["stage1"], cfg["stage2"], cfg["stage3"]
    E = len(cfg["encoders"]); n_sub = 2 ** E - 1
    nf = P["n_inner_folds"]; proj = {"stage1": 0.0, "stage2": 0.0, "stage3": 0.0, "per_dataset": {}}
    for ds, b in bench.items():
        t = b["fold_eval_mean_s"]
        s1 = P["n_outer"] * len(S1["strategies"]) * S1["budget_fold_evals"] * t
        s1 += P["n_outer"] * len(S1["strategies"]) * S1["holdout_budget_evals"] * t * 0.8   # holdout: 1 fold = val
        s1 += P["n_outer"] * len(S1["ablation_featuresets"]) * S1["budget_fold_evals"] * t
        s1 += (P["n_outer"] * (len(S1["strategies"]) + len(S1["ablation_featuresets"])) * S1["budget_fold_evals"] / nf
               + P["n_holdout"] * len(S1["strategies"]) * S1["holdout_budget_evals"]) * t_opt          # optuna overhead (upper bound)
        s1 += P["n_outer"] * (3 + len(S1["ablation_featuresets"])) * (E + 4) * 13 * nf * t          # baselines
        s2 = len(S2["outer_splits"]) * n_sub * S2["budget_fold_evals"] * (t + t_opt / nf)
        s3 = S3["budget_fold_evals"] * t * 1.25 + (len(bench) - 1) * 5 * t
        proj["stage1"] += s1; proj["stage2"] += s2; proj["stage3"] += s3
        proj["per_dataset"][ds] = {"stage1_h": s1 / 3600, "stage2_h": s2 / 3600, "stage3_h": s3 / 3600}
    return proj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-bench", action="store_true")
    ap.add_argument("--gpus", type=int, default=max(1, torch.cuda.device_count()))
    a = ap.parse_args()
    cfg = load_config(a.config); dev = device()
    print(f"config={cfg['_path']} device={dev} {env_info()['gpu']}")
    for ds in cfg["datasets"]:
        meta, dims = check_features(cfg, ds)
        sp = make_splits(meta, cfg)
        p = Path(cfg["paths"]["work"]) / ds / "splits.json"
        if p.exists():
            old = json.loads(p.read_text())
            assert old == json.loads(json.dumps(sp)), f"{p} exists and differs from regenerated splits -- refusing to overwrite"
        else:
            atomic_json(p, sp)
        print(f"  {ds:<9} N={len(meta):>5} groups={sp['groups']:>5} "
              f"outer test={len(sp['nested'][0]['test'])} holdout test={len(sp['holdout'][0]['test'])} dims={dims}")
    if a.no_bench:
        return
    bench = {}
    for ds in cfg["datasets"]:
        bench[ds] = bench_dataset(cfg, ds, dev)
        b = bench[ds]
        print(f"  bench {ds:<9} bank {b['bank_s']:.2f}s peak {b['peak_gb']:.1f} GB | per fold-eval: " +
              " ".join(f"{k}={v*1e3:.0f}ms" for k, v in b["fold_eval_s"].items()))
    t_opt = optuna_overhead()
    proj = project(cfg, bench, t_opt)
    W = a.gpus * cfg["runtime"]["workers_per_gpu"]; eff = 0.5     # conservative concurrency efficiency
    print(f"\n  optuna overhead per trial ~{t_opt*1e3:.0f} ms")
    print(f"  projected single-stream GPU hours: stage1 {proj['stage1']/3600:.1f} | stage2 {proj['stage2']/3600:.1f} | "
          f"stage3 {proj['stage3']/3600:.1f}")
    print(f"  projected wall-clock with {a.gpus} GPU(s) x {cfg['runtime']['workers_per_gpu']} workers (eff {eff}): "
          f"stage1 {proj['stage1']/3600/(W*eff):.1f} h | stage2 {proj['stage2']/3600/(W*eff):.1f} h | "
          f"stage3 {proj['stage3']/3600/(W*eff):.2f} h")
    for ds, v in proj["per_dataset"].items():
        print(f"    {ds:<9} " + " ".join(f"{k}={x:.2f}" for k, x in v.items()))
    atomic_json(Path(cfg["paths"]["work"]) / "plan.json",
                {"bench": bench, "optuna_overhead_s": t_opt, "projection_s": proj, "env": env_info(),
                 "config": {k: cfg[k] for k in ("protocol", "stage1", "stage2", "stage3", "runtime")}})


if __name__ == "__main__":
    main()
