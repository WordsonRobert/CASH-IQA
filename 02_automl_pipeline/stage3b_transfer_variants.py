#!/usr/bin/env python3
"""
Stage 3b -- transfer decomposition: restricted variants of the stage-3 cross-dataset test.

Separates the two components of a tuned configuration that can affect transfer: the regressor family
(local vs global) and the learned encoder weights. Same protocol as stage 3 (5 grouped folds on the FULL source set; no target
information of any kind; budget = stage3.budget_fold_evals; tpe_hb), three restricted search spaces:

  global      families {global_krr_lin, global_krr_rbf}, encoder weights searched
  eqw         all six families,                         encoder weights fixed equal
  global_eqw  families {global_krr_lin, global_krr_rbf}, encoder weights fixed equal

    python run_jobs.py --stage 4            # (run_jobs.py maps stage 4 -> this file)
Output: results/stage3b_transfer_variants/{variant}/{src}.json
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import *
from stage3_cross_dataset import cross_predict

VARIANTS = {"global": (list(FAMILIES_GLOBAL), False),
            "eqw": (None, True),
            "global_eqw": (list(FAMILIES_GLOBAL), True)}


class EqWSpace(Space):
    """same space, but every encoder weight is fixed to 1 (equal fusion)."""

    def from_unit(self, u):
        c = super().from_unit(u); c["w"] = {e: 1.0 for e in self.enc}; return c

    def optuna(self, trial):
        enc = self.enc; self.enc = enc[:1]                   # suppress the weight suggestions
        try:
            c = super().optuna(trial)
        finally:
            self.enc = enc
        c["w"] = {e: 1.0 for e in enc}
        return self.clean(c)


def list_jobs(cfg):
    return [{"src": ds, "variant": v} for v in VARIANTS for ds in cfg["datasets"]]


def out_file(cfg, j):
    return Path(cfg["paths"]["results"]) / "stage3b_transfer_variants" / j["variant"] / f"{j['src']}.json"


def is_done(cfg, j):
    return out_file(cfg, j).exists()


def run_job(cfg, j):
    dev = device(); S3 = cfg["stage3"]; E = cfg["encoders"]; src = j["src"]
    fams, eqw = VARIANTS[j["variant"]]
    meta, feats = load_features(cfg, src, E, S3["featureset"]); sp = load_splits(cfg, src); y = meta["q"].values
    bank = Bank(feats, E, dev); Xs = [torch.from_numpy(feats[e]).to(dev) for e in E]; del feats
    fo = np.array(sp["cv_full"]); allidx = np.arange(len(y))
    folds = [(allidx[fo == f], allidx[fo != f]) for f in range(cfg["protocol"]["n_inner_folds"])]
    plan = Plan(bank, y, folds, (np.array([], int), allidx)); plan.chunk = cfg["runtime"]["local_chunk_elems"]
    space = (EqWSpace if eqw else Space)(cfg, E, fams)
    seed = job_seed(cfg["protocol"]["seed"], "cross3b", j["variant"], src)
    led, secs = search(S3["strategy"], space, plan, S3["budget_fold_evals"], seed)
    plan.freeze(); inner, best = led.best
    y_src = torch.from_numpy(y.astype(np.float64)).to(dev)
    res = {"job": j, "families": fams or cfg["search_space"]["families"], "equal_weights": eqw,
           "selected": best, "inner": inner, "search_seconds": secs, "targets": {}, "env": env_info()}
    for tgt in cfg["datasets"]:
        if tgt == src: continue
        mt, ft = load_features(cfg, tgt, E, S3["featureset"]); Xt = [torch.from_numpy(ft[e]).to(dev) for e in E]; del ft
        r = metrics(mt["q"].values, cross_predict(bank, Xs, Xt, best, y_src, chunk_elems=plan.chunk))
        res["targets"][tgt] = r; del Xt
        print(f"[{j['variant']}: {src} -> {tgt}] srocc={r['srocc']:.4f} (family={best['family']})", flush=True)
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
