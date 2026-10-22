#!/usr/bin/env python3
"""
Transfer-aware shrinkage of the fusion weights (analysis of stage 6).

Rule (fixed before the run, see stage6_transfer_shrinkage.py):
  for each target T, alpha*_T = argmax_alpha  mean SROCC over pairs (s -> t), s != T, t != T, t != s
  (the 20 transfer pairs among the other five datasets); ties go to the larger alpha.
  Every pair (s -> T) is then scored at alpha*_T.  T is never used to choose alpha*_T.

Compared on the same 30 (source, target) pairs:
  full CASH (alpha = 0, = stage 3), LODO shrinkage (alpha*_T), equal weights with the tuned family and
  hyperparameters (alpha = 1), ridge_concat (stage 3), and the re-searched restricted spaces of stage 3b.

Pre-registered reading: shrinkage closes the transfer gap when its mean over the 30 pairs is above full CASH
and within 0.010 of ridge_concat.  The paired tests over 30 pairs are descriptive (pairs share sources and
targets, so they are not independent).

Also: the in-domain cost of shrinking (nested protocol, stage-1 tpe_hb configurations, median over splits).

Descriptive outputs (added after the run; they do not enter the criterion):
  share of the full-vs-ridge gap recovered; means by source and by target; pairs that shrinkage makes worse;
  in-domain alpha = 0 vs alpha = 1 per dataset (paired Wilcoxon over the 10 splits, Holm over 6 datasets) and the
  alpha that inner CV itself would choose on each split.

    python transfer_shrinkage.py      # writes ../results/tables/transfer_shrinkage_*.csv, shrinkage_indomain.csv
"""
import json, glob, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); out = np.empty_like(p); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (len(p) - r) * p[i])); out[i] = run
    return out

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"; T = R / "tables"
DS = ["kadid10k", "csiq", "tid2013", "koniq10k", "livec", "spaq"]
SYN = {"kadid10k", "csiq", "tid2013"}
TOL = 0.010


def dom(d): return "S" if d in SYN else "A"


def main():
    S6 = R / "stage6_transfer_shrinkage"
    cross = {}
    for ds in DS:
        f = S6 / "cross" / f"{ds}.json"
        if f.exists(): cross[ds] = json.loads(f.read_text())
    if len(cross) < len(DS):
        print(f"WARNING: cross results for {sorted(cross)} only; LODO needs all six sources for the full table")
    if not cross: sys.exit("no stage-6 cross results")
    alphas = [str(a) for a in next(iter(cross.values()))["alphas"]]
    sr = {(s, t): {a: cross[s]["targets"][t][a]["srocc"] for a in alphas} for s in cross for t in cross[s]["targets"]}
    s3 = {s: json.loads((R / "stage3_cross_dataset" / f"{s}.json").read_text()) for s in cross}
    ridge = {(s, t): s3[s]["targets"][t]["ridge_concat"]["srocc"] for (s, t) in sr}
    v3b = {}
    for var in ["global", "eqw", "global_eqw"]:
        for s in cross:
            f = R / "stage3b_transfer_variants" / var / f"{s}.json"
            if f.exists():
                d = json.loads(f.read_text())
                for t, m in d["targets"].items(): v3b.setdefault(var, {})[(s, t)] = m["srocc"]
    chk = max(abs(x) for s in cross for x in cross[s]["checks"].values())
    print(f"check: alpha=0 reproduces stage 3, max |diff| = {chk:.2e}")

    # ---- leave-one-dataset-out choice of alpha per target
    astar, rows_a = {}, []
    for Tg in DS:
        pairs = [(s, t) for (s, t) in sr if s != Tg and t != Tg]
        if not pairs: continue
        sc = {a: float(np.mean([sr[p][a] for p in pairs])) for a in alphas}
        best = max(sc.values()); astar[Tg] = max((a for a in alphas if sc[a] >= best - 1e-12), key=float)
        rows_a.append(dict(target=Tg, alpha_star=float(astar[Tg]), n_selection_pairs=len(pairs),
                           **{f"sel_mean_a{a}": sc[a] for a in alphas}))
    A = pd.DataFrame(rows_a)

    rows = []
    for (s, t) in sorted(sr):
        if t not in astar: continue
        r = dict(src=s, tgt=t, pair=f"{dom(s)}->{dom(t)}", full=sr[(s, t)]["0.0"], shrink_lodo=sr[(s, t)][astar[t]],
                 alpha_star=float(astar[t]), eqw_fixed_hp=sr[(s, t)]["1.0"], ridge_concat=ridge[(s, t)])
        for var, d in v3b.items():
            if (s, t) in d: r[f"s3b_{var}"] = d[(s, t)]
        rows.append(r)
    P = pd.DataFrame(rows)
    methods = [c for c in ["full", "shrink_lodo", "eqw_fixed_hp", "s3b_global", "s3b_eqw", "s3b_global_eqw", "ridge_concat"]
               if c in P]
    G = P.groupby("pair")[methods].mean().reindex(["A->A", "A->S", "S->A", "S->S"])
    G.loc["ALL (%d pairs)" % len(P)] = P[methods].mean()

    curve = pd.DataFrame([{"alpha": float(a), "mean_all_pairs": float(np.mean([sr[p][a] for p in sr])),
                           **{k: float(np.mean([sr[p][a] for p in sr if f"{dom(p[0])}->{dom(p[1])}" == k]))
                              for k in ["A->A", "A->S", "S->A", "S->S"]}} for a in alphas])

    # ---- in-domain cost
    rows_i = []
    for ds in DS:
        for f in sorted(glob.glob(str(S6 / "indomain" / ds / "split*.json"))):
            d = json.loads(open(f).read())
            for a in alphas:
                rows_i.append(dict(ds=ds, split=d["job"]["split"], alpha=float(a), test=d["test"][a]["srocc"],
                                   inner=d["inner"][a], check=d["check_alpha0_test_diff"]))
    I = pd.DataFrame(rows_i)

    T.mkdir(parents=True, exist_ok=True)
    P.to_csv(T / "transfer_shrinkage_pairs.csv", index=False); A.to_csv(T / "transfer_shrinkage_alpha_lodo.csv", index=False)
    G.to_csv(T / "transfer_shrinkage_summary.csv"); curve.to_csv(T / "transfer_shrinkage_curve.csv", index=False)
    if len(I): I.to_csv(T / "shrinkage_indomain.csv", index=False)

    pd.set_option("display.width", 220)
    print("\nalpha*_T chosen without the target (leave-one-dataset-out):")
    print(A[["target", "alpha_star", "n_selection_pairs"]].to_string(index=False))
    print("\nmean transfer SROCC by domain pair:")
    print(G.round(3).to_string())
    print("\ndescriptive only (not used for selection): mean SROCC vs alpha")
    print(curve.round(3).to_string(index=False))
    for other in ["full", "ridge_concat"]:
        d = P.shrink_lodo - P[other]
        try: p = wilcoxon(P.shrink_lodo, P[other]).pvalue
        except ValueError: p = float("nan")
        print(f"\nshrink_lodo - {other}: mean {d.mean():+.4f}, wins {(d > 0).sum()}/{len(d)}, Wilcoxon p {p:.3g} (descriptive)")
    if len(P) == 30:
        m_s, m_f, m_r = P.shrink_lodo.mean(), P.full.mean(), P.ridge_concat.mean()
        ok = (m_s > m_f) and (m_s >= m_r - TOL)
        print(f"\nPRE-REGISTERED: shrink {m_s:.3f} vs full {m_f:.3f}, ridge {m_r:.3f} (tolerance {TOL}) -> "
              f"{'CLOSES THE GAP' if ok else 'DOES NOT CLOSE THE GAP'}")
    if len(I):
        print(f"\nin-domain (nested, median test SROCC over splits; alpha=0 check max |diff| {I.check.abs().max():.2e}):")
        M = I.pivot_table(index="ds", columns="alpha", values="test", aggfunc="median").reindex(DS)
        print(M.round(3).to_string())
        print("\nin-domain inner-CV (median): does inner CV itself prefer alpha=0?")
        print(I.pivot_table(index="ds", columns="alpha", values="inner", aggfunc="median").reindex(DS).round(3).to_string())
    # ---- descriptive extras
    print("\n" + "-" * 100 + "\ndescriptive (not part of the criterion)")
    if len(P) == 30:
        print(f"share of the full -> ridge gap recovered by shrinkage: "
              f"{(P.shrink_lodo.mean() - P.full.mean()) / (P.ridge_concat.mean() - P.full.mean()):.1%}")
    cols = [c for c in ["full", "shrink_lodo", "s3b_eqw", "s3b_global_eqw", "ridge_concat"] if c in P]
    BS = P.groupby("src")[cols].mean().reindex(DS); BT = P.groupby("tgt")[cols].mean().reindex(DS)
    BS.to_csv(T / "transfer_shrinkage_by_source.csv"); BT.to_csv(T / "transfer_shrinkage_by_target.csv")
    print("\nmean over the 5 targets, by source:"); print(BS.round(3).to_string())
    print("\nmean over the 5 sources, by target:"); print(BT.round(3).to_string())
    worse = P[P.shrink_lodo < P.full]
    print(f"\npairs where shrinkage is worse than full CASH ({len(worse)}/{len(P)}):")
    print(worse[["src", "tgt", "full", "shrink_lodo"]].round(3).to_string(index=False))
    orc = np.array([max(sr[(r.src, r.tgt)][a] for a in alphas) for r in P.itertuples()])
    arg = [max(alphas, key=lambda a: sr[(r.src, r.tgt)][a]) for r in P.itertuples()]
    print(f"\noracle shrinkage (best alpha chosen per pair WITH the target's labels; an upper bound for any alpha rule, "
          f"not a method): mean {orc.mean():.3f} vs ridge {P.ridge_concat.mean():.3f}")
    print("  alpha that is best per pair:", pd.Series(arg).value_counts().sort_index().to_dict())
    print("  LODO selection score strictly increasing in alpha for every target:",
          all(np.all(np.diff(A.loc[A.target == t, [f"sel_mean_a{a}" for a in alphas]].values[0]) > 0) for t in A.target))
    if "s3b_eqw" in P:
        d = P.shrink_lodo - P.s3b_eqw
        print(f"\nshrink_lodo - s3b_eqw (equal weights re-searched): mean {d.mean():+.4f}, wins {(d > 0).sum()}/{len(d)}")
    if len(I):
        rows = []
        for ds in DS:
            a0 = I[(I.ds == ds) & (I.alpha == 0.0)].sort_values("split"); a1 = I[(I.ds == ds) & (I.alpha == 1.0)].sort_values("split")
            if not len(a0): continue
            arg = I[I.ds == ds].groupby("split").apply(lambda g: g.loc[g.inner.idxmax(), "alpha"])
            rows.append(dict(ds=ds, median_a0=a0.test.median(), median_a1=a1.test.median(),
                             mean_diff_a1_minus_a0=float((a1.test.values - a0.test.values).mean()),
                             a0_wins=int((a0.test.values > a1.test.values).sum()), n=len(a0),
                             p=wilcoxon(a0.test.values, a1.test.values).pvalue, inner_argmax_alpha0=int((arg == 0.0).sum())))
        D = pd.DataFrame(rows); D["p_holm"] = holm(D.p.values); D.to_csv(T / "shrinkage_indomain_tests.csv", index=False)
        print("\nin-domain alpha=0 vs alpha=1 (paired over splits; Holm over datasets) and inner-CV choice of alpha:")
        print(D.round(4).to_string(index=False))
    print(f"\nwrote {T}/transfer_shrinkage_*.csv" + (", shrinkage_indomain*.csv" if len(I) else ""))


if __name__ == "__main__":
    main()
