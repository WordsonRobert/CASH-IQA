#!/usr/bin/env python3
"""
Exact Shapley attribution of the FUSION GAIN (post-processing of stage 2; CPU, seconds).

stage4_analysis.py computes Shapley values for the raw game v(S) = test SROCC of the model re-tuned on
encoder subset S, with v(empty) = 0. Because every single encoder already scores ~0.8-0.9, that convention
gives each encoder ~1/6 of the total "for existing" and makes every pairwise interaction strongly negative.

Here the game is re-baselined per split:
    g(S) = v(S) - c,   c = mean_e v({e})  (mean single-encoder score),   g(empty) = 0
so that  sum_e phi_e = v(all) - c  = the fusion gain over an average single encoder.  Rankings of phi are
unchanged by the shift (it moves every phi by -c/n); interactions become interpretable.

    phi_i  = sum_{S subset N\\{i}}  |S|!(n-|S|-1)!/n!        [g(S+i) - g(S)]
    I_ij   = sum_{S subset N\\{i,j}} |S|!(n-|S|-2)!/(n-1)!   [g(S+ij) - g(S+i) - g(S+j) + g(S)]   (Grabisch)

Both the test game and the inner-CV game are computed; 95% CIs = 5000-sample bootstrap over the 10 splits.

    python shapley_fusion_gain.py          # reads ../results/stage2_shapley, writes ../results/tables/shapley_*gain*.csv
"""
import json, glob, math, itertools, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
S2 = ROOT / "results" / "stage2_shapley"; T = ROOT / "results" / "tables"
DS = ["kadid10k", "csiq", "tid2013", "koniq10k", "livec", "spaq"]
E = ["arniqa", "contrique", "resnet50", "dinov2", "clip", "mae"]; n = len(E)
f = math.factorial
W1 = {k: f(k) * f(n - k - 1) / f(n) for k in range(n)}          # Shapley weights
W2 = {k: f(k) * f(n - k - 2) / f(n - 1) for k in range(n - 1)}  # Shapley interaction weights
rng = np.random.default_rng(0)


def game(recs, key):
    v = {frozenset(s): r[key] for s, r in recs.items()}
    c = float(np.mean([v[frozenset([e])] for e in E]))
    g = {S: x - c for S, x in v.items()}; g[frozenset()] = 0.0
    return g, c


def shapley(g):
    out = {}
    for e in E:
        rest = [x for x in E if x != e]
        out[e] = sum(W1[k] * (g[frozenset(S) | {e}] - g[frozenset(S)]) for k in range(n) for S in itertools.combinations(rest, k))
    return out


def interactions(g):
    out = {}
    for i, j in itertools.combinations(E, 2):
        rest = [x for x in E if x not in (i, j)]; s = 0.0
        for k in range(n - 1):
            for S in itertools.combinations(rest, k):
                S = frozenset(S); s += W2[k] * (g[S | {i, j}] - g[S | {i}] - g[S | {j}] + g[S])
        out[f"{i}x{j}"] = s
    return out


def ci(x):
    x = np.asarray(x); b = [rng.choice(x, len(x)).mean() for _ in range(5000)]
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    rows_s, rows_i, rows_b = [], [], []
    for ds in DS:
        for val in ["test", "inner"]:
            Sv, Iv, Cv, Gv = [], [], [], []
            for fn in sorted(glob.glob(str(S2 / ds / "split*.jsonl"))):
                recs = {}
                for line in open(fn):
                    r = json.loads(line); recs[tuple(r["subset"])] = {"test": r["test"]["srocc"], "inner": r["inner"]}
                assert len(recs) == 2 ** n - 1, f"{fn}: {len(recs)} subsets"
                g, c = game(recs, val); Sv.append(shapley(g)); Iv.append(interactions(g)); Cv.append(c); Gv.append(g[frozenset(E)])
            for e in E:
                m, lo, hi = ci([s[e] for s in Sv]); rows_s.append(dict(value=val, ds=ds, encoder=e, mean=m, lo95=lo, hi95=hi))
            for p in Iv[0]:
                m, lo, hi = ci([i[p] for i in Iv]); rows_i.append(dict(value=val, ds=ds, pair=p, mean=m, lo95=lo, hi95=hi))
            rows_b.append(dict(value=val, ds=ds, mean_single=float(np.mean(Cv)), fusion_gain=float(np.mean(Gv)), n_splits=len(Sv)))
    s, i, b = pd.DataFrame(rows_s), pd.DataFrame(rows_i), pd.DataFrame(rows_b)
    T.mkdir(parents=True, exist_ok=True)
    s.to_csv(T / "shapley_gain.csv", index=False); i.to_csv(T / "shapley_interactions_gain.csv", index=False)
    b.to_csv(T / "shapley_baseline.csv", index=False)
    pd.set_option("display.width", 220)
    print("mean single-encoder SROCC and fusion gain (test):")
    print(b[b.value == "test"].set_index("ds")[["mean_single", "fusion_gain"]].round(4).T.to_string())
    t = s[s.value == "test"]
    print("\nShapley share of the fusion gain (test) [95% CI]:")
    print(t.assign(txt=[f"{m:+.4f} [{l:+.4f},{h:+.4f}]" for m, l, h in zip(t["mean"], t.lo95, t.hi95)])
          .pivot(index="encoder", columns="ds", values="txt")[DS].to_string())
    print("\ninner-vs-test rank agreement of the Shapley values (Spearman rho):")
    for ds in DS:
        a = s[(s.ds == ds) & (s.value == "test")].set_index("encoder")["mean"][E]
        c = s[(s.ds == ds) & (s.value == "inner")].set_index("encoder")["mean"][E]
        print(f"  {ds:<9} {spearmanr(a, c)[0]:.2f}")
    print(f"\nwrote {T}/shapley_gain.csv, shapley_interactions_gain.csv, shapley_baseline.csv")


if __name__ == "__main__":
    main()
