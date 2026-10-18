#!/usr/bin/env python3
"""
Sanity check of the extracted features BEFORE shipping them to the server.

For every (dataset, encoder) it runs the standard frozen-feature linear probe used by
CONTRIQUE / ARNIQA (ridge regression, alpha chosen on the validation split, 70/10/20,
reference-disjoint for synthetic sets) on a few random splits and prints the median SROCC
next to the numbers published by the encoders' authors.

This is a fidelity check of the extraction, not a reported result: if ARNIQA / CONTRIQUE land
within ~0.01-0.02 of their papers, the extraction protocol is right.
"""
import argparse, os
from pathlib import Path
import numpy as np, pandas as pd, h5py
from scipy.stats import spearmanr

PUBLISHED = {  # SROCC, frozen encoder + ridge, as reported by the authors
    "arniqa":    {"live": .966, "csiq": .962, "tid2013": .880, "kadid10k": .908, "spaq": .905},
    "contrique": {"live": .960, "csiq": .942, "tid2013": .843, "kadid10k": .934, "koniq10k": .894,
                  "livec": .845, "spaq": .914},
}


def splits(meta, n, seed):
    g = meta["ref"].values; u = np.array(sorted(set(g)))
    r = np.random.RandomState(seed); r.shuffle(u)
    a, b = int(.7 * len(u)), int(.8 * len(u))
    s = {x: 0 for x in u[:a]} | {x: 1 for x in u[a:b]} | {x: 2 for x in u[b:]}
    lab = np.array([s[x] for x in g])
    return [np.where(lab == k)[0] for k in range(3)]


def ridge_path(X, y, alphas):
    mu = X.mean(0); ym = y.mean(); Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Uy = U.T @ (y - ym)
    return [(Vt.T @ (S / (S ** 2 + a) * Uy), mu, ym) for a in alphas]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default=str(Path(__file__).resolve().parents[1] / "features"))
    ap.add_argument("--splits", type=int, default=3)
    a = ap.parse_args()
    alphas = np.logspace(-3, 3, 13)
    rows = []
    for ds_dir in sorted(Path(a.feat).iterdir()):
        if not (ds_dir / "meta.csv").exists():
            continue
        meta = pd.read_csv(ds_dir / "meta.csv"); ds = ds_dir.name; y = meta["q"].values.astype(np.float64)
        for h5 in sorted(ds_dir.glob("*.h5")):
            with h5py.File(h5) as f:
                keys = [k.decode() for k in f["keys"][:]]
                assert keys == meta["key"].tolist(), f"{h5}: key order differs from meta.csv"
                variants = [k for k in f.keys() if k != "keys"]
                for v in variants:
                    X = f[v][:].astype(np.float64)
                    res = []
                    for s in range(a.splits):
                        tr, va, te = splits(meta, len(y), s)
                        path = ridge_path(X[tr], y[tr], alphas)
                        vs = [spearmanr(y[va], (X[va] - mu) @ w + ym)[0] for w, mu, ym in path]
                        w, mu, ym = path[int(np.argmax(vs))]
                        res.append(spearmanr(y[te], (X[te] - mu) @ w + ym)[0])
                    pub = PUBLISHED.get(h5.stem, {}).get(ds) if v == "main" else None
                    rows.append(dict(dataset=ds, encoder=h5.stem, variant=v, dim=X.shape[1],
                                     srocc=round(float(np.median(res)), 4), published=pub))
                    print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    print("\n", df.pivot_table(index=["encoder", "variant"], columns="dataset", values="srocc").round(3).to_string())
    df.to_csv(Path(a.feat) / "verify_linear_probe.csv", index=False)


if __name__ == "__main__":
    main()
