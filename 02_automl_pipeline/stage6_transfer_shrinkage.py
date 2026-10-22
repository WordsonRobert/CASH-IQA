#!/usr/bin/env python3
"""
Stage 6 -- transfer-aware fusion: shrinking the tuned encoder weights toward equal fusion.

For a selected configuration with effective weights  omega_e = w_e^2 / sum w^2  (stage 3, or stage 1), and
a shrinkage level alpha in ALPHAS,

    omega_alpha = (1 - alpha) * omega  +  alpha / E          w_e(alpha) = sqrt(omega_alpha,e)

Family and every other hyperparameter stay as selected; nothing is re-searched. alpha = 0 is the tuned model,
alpha = 1 the same model with equal fusion weights.

Two job kinds:
  cross     one per source dataset: the stage-3 configuration (selected on the full source by grouped CV, no
            target information), the full source as gallery, every other dataset predicted at every alpha.
            alpha = 0 must reproduce the stage-3 automl SROCC (built-in check).
  indomain  one per (dataset, nested split): the stage-1 tpe_hb configuration at every alpha, inner-CV score
            (before freeze) and test score (after freeze) -- the in-domain cost of shrinking.

The rule that turns this into a method is fixed in advance (03_analysis/transfer_shrinkage.py):
  for a target T, alpha*_T maximises the mean SROCC over all transfer pairs among the OTHER five datasets
  (source != T, target != T); ties go to the larger alpha. T's own images and labels are never used to pick
  alpha*_T. Pre-registered reading: the rule is reported as closing the transfer gap when its mean SROCC over
  the 30 pairs is above full CASH and within 0.010 of ridge_concat. Reported either way.

    python run_jobs.py --stage 6
Output: results/stage6_transfer_shrinkage/cross/{src}.json, .../indomain/{ds}/split{k:02d}.json
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import *
from stage3_cross_dataset import cross_predict

ALPHAS = [round(0.1 * i, 1) for i in range(11)]


def shrink(sel, encoders, alpha):
    w = np.array([float(sel["w"].get(e, 0.0)) for e in encoders]) ** 2
    om = w / w.sum(); om_a = (1.0 - alpha) * om + alpha / len(encoders)
    c = json.loads(json.dumps(sel)); c["w"] = {e: float(np.sqrt(x)) for e, x in zip(encoders, om_a)}
    return c, om_a


def list_jobs(cfg):
    jobs = [{"kind": "cross", "src": ds} for ds in cfg["datasets"]]
    jobs += [{"kind": "indomain", "ds": ds, "split": k} for ds in cfg["datasets"] for k in range(cfg["protocol"]["n_outer"])]
    return jobs


def out_file(cfg, j):
    R = Path(cfg["paths"]["results"]) / "stage6_transfer_shrinkage"
    if j["kind"] == "cross":
        return R / "cross" / f"{j['src']}.json"
    return R / "indomain" / j["ds"] / f"split{j['split']:02d}.json"


def is_done(cfg, j):
    return out_file(cfg, j).exists()


@torch.no_grad()
def run_cross(cfg, j):
    dev = device(); E = cfg["encoders"]; src = j["src"]; t0 = time.time()
    s3 = json.loads((Path(cfg["paths"]["results"]) / "stage3_cross_dataset" / f"{src}.json").read_text())
    sel = s3["selected"]
    meta, feats = load_features(cfg, src, E, cfg["stage3"]["featureset"]); y = meta["q"].values
    bank = Bank(feats, E, dev); Xs = [torch.from_numpy(feats[e]).to(dev) for e in E]; del feats
    y_src = torch.from_numpy(y.astype(np.float64)).to(dev)
    chunk = cfg["runtime"]["local_chunk_elems"]
    res = {"job": j, "selected": sel, "alphas": ALPHAS,
           "omega": {str(a): shrink(sel, E, a)[1].tolist() for a in ALPHAS}, "targets": {}, "checks": {}}
    for tgt in cfg["datasets"]:
        if tgt == src: continue
        mt, ft = load_features(cfg, tgt, E, cfg["stage3"]["featureset"])
        Xt = [torch.from_numpy(ft[e]).to(dev) for e in E]; del ft; yt = mt["q"].values
        r = {}
        for a in ALPHAS:
            r[str(a)] = metrics(yt, cross_predict(bank, Xs, Xt, shrink(sel, E, a)[0], y_src, chunk_elems=chunk))
        res["targets"][tgt] = r; del Xt
        ref = s3["targets"][tgt]["automl"]["srocc"]
        res["checks"][tgt] = float(r["0.0"]["srocc"] - ref)
        print(f"[{src} -> {tgt}] a=0 {r['0.0']['srocc']:.4f} (stage3 {ref:.4f}) a=0.5 {r['0.5']['srocc']:.4f} "
              f"a=1 {r['1.0']['srocc']:.4f} ridge {s3['targets'][tgt]['ridge_concat']['srocc']:.4f}", flush=True)
    res["seconds"] = time.time() - t0; res["env"] = env_info()
    atomic_json(out_file(cfg, j), res)


@torch.no_grad()
def run_indomain(cfg, j):
    dev = device(); E = cfg["encoders"]; t0 = time.time()
    s1 = json.loads((Path(cfg["paths"]["results"]) / "stage1_automl" / "main" / "nested" / j["ds"] /
                     f"split{j['split']:02d}" / "tpe_hb.json").read_text())
    sel = s1["selected"]
    meta, feats = load_features(cfg, j["ds"], E, "main"); sp = load_splits(cfg, j["ds"]); y = meta["q"].values
    bank = Bank(feats, E, dev); del feats
    plan = nested_plan(bank, y, sp["nested"][j["split"]], cfg["protocol"]["n_inner_folds"])
    plan.chunk = cfg["runtime"]["local_chunk_elems"]; nf = len(plan.folds)
    cfgs = {str(a): shrink(sel, E, a)[0] for a in ALPHAS}
    inner = {a: float(np.mean([plan.eval_fold(c, f) for f in range(nf)])) for a, c in cfgs.items()}
    plan.freeze()                                             # ---- test only after this line
    test = {a: plan.final(c)[1] for a, c in cfgs.items()}
    res = {"job": j, "selected": sel, "alphas": ALPHAS, "inner": inner, "test": test,
           "check_alpha0_test_diff": float(test["0.0"]["srocc"] - s1["test"]["srocc"]),
           "seconds": time.time() - t0, "env": env_info()}
    atomic_json(out_file(cfg, j), res)
    print(f"[{j['ds']} s{j['split']}] test a=0 {test['0.0']['srocc']:.4f} (stage1 {s1['test']['srocc']:.4f}) "
          f"a=0.5 {test['0.5']['srocc']:.4f} a=1 {test['1.0']['srocc']:.4f}", flush=True)


def run_job(cfg, j):
    (run_cross if j["kind"] == "cross" else run_indomain)(cfg, j)


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
