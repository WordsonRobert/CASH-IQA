#!/usr/bin/env python3
"""
Stage 3 -- cross-dataset generalisation.

One job = one source dataset. The CASH search runs on 5 grouped folds over ALL source images
(no target information of any kind), the selected model uses the full source set as gallery,
and is scored on every other dataset (all of its images). Fixed baselines (ridge probes per
encoder / concat) are selected the same way. Scores are q (higher = better) on both sides.

    python run_jobs.py --stage 3
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import *


def list_jobs(cfg):
    return [{"src": ds} for ds in cfg["datasets"]]


def out_file(cfg, j):
    return Path(cfg["paths"]["results"]) / "stage3_cross_dataset" / f"{j['src']}.json"


def is_done(cfg, j):
    return out_file(cfg, j).exists()


@torch.no_grad()
def cross_predict(bank, Xs, Xt, cfg_m, y_src, chunk_rows=2048, chunk_elems=2e8):
    w = bank.wvec(cfg_m); g = torch.arange(bank.N, device=bank.dev); out = []
    for s in range(0, Xt[0].shape[0], chunk_rows):
        q = None
        for ei, we in enumerate(w.tolist()):
            if we == 0.0: continue
            blk = (Xt[ei][s:s + chunk_rows] @ Xs[ei].T) * we
            q = blk if q is None else q.add_(blk)
        out.append(predict(bank, cfg_m, q, g, y_src, chunk_elems))
    return torch.cat(out).cpu().numpy()


def run_job(cfg, j):
    dev = device(); S3 = cfg["stage3"]   # threads via OMP_NUM_THREADS (set by run_jobs.py)
    E = cfg["encoders"]; src = j["src"]
    meta, feats = load_features(cfg, src, E, S3["featureset"]); sp = load_splits(cfg, src); y = meta["q"].values
    bank = Bank(feats, E, dev); Xs = [torch.from_numpy(feats[e]).to(dev) for e in E]; del feats
    fo = np.array(sp["cv_full"]); allidx = np.arange(len(y))
    folds = [(allidx[fo == f], allidx[fo != f]) for f in range(cfg["protocol"]["n_inner_folds"])]
    plan = Plan(bank, y, folds, (np.array([], int), allidx)); plan.chunk = cfg["runtime"]["local_chunk_elems"]
    seed = job_seed(cfg["protocol"]["seed"], "cross", src)
    led, secs = search(S3["strategy"], Space(cfg, E), plan, S3["budget_fold_evals"], seed)
    plan.freeze(); inner, best = led.best
    bplan = Plan(bank, y, folds, (np.array([], int), allidx)); bplan.chunk = plan.chunk
    base = run_baselines(bplan, E); bplan.freeze()
    y_src = torch.from_numpy(y.astype(np.float64)).to(dev)
    res = {"job": j, "selected": best, "inner": inner, "search_seconds": secs, "targets": {}, "env": env_info()}
    for tgt in cfg["datasets"]:
        if tgt == src: continue
        mt, ft = load_features(cfg, tgt, E, S3["featureset"]); Xt = [torch.from_numpy(ft[e]).to(dev) for e in E]; del ft
        yt = mt["q"].values; r = {"automl": metrics(yt, cross_predict(bank, Xs, Xt, best, y_src, chunk_elems=plan.chunk))}
        for name, (s, c) in base.items():
            r[name] = metrics(yt, cross_predict(bank, Xs, Xt, c, y_src, chunk_elems=plan.chunk))
        res["targets"][tgt] = r; del Xt
        print(f"[{src} -> {tgt}] automl={r['automl']['srocc']:.4f} ridge_concat={r['ridge_concat']['srocc']:.4f}", flush=True)
    res["baselines_selected"] = {k: {"inner": s, "cfg": c} for k, (s, c) in base.items()}
    atomic_json(out_file(cfg, j), res)


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
