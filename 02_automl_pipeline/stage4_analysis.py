#!/usr/bin/env python3
"""
Stage 4 -- aggregation, statistics, encoder story (CPU only, minutes).

Writes results/tables/*.csv and results/report_stage4.md:
  headline_{nested,holdout}.csv   median / mean / std of SROCC, PLCC(4-p logistic), KRCC, RMSE per method
  significance_{protocol}.csv     primary AutoML vs every baseline: paired Wilcoxon over splits, Holm-corrected
  automl_benchmark.csv            per strategy: final test/inner, wall time, budget-to-99%-of-final
  automl_anytime.csv              anytime curves (median over splits) -- inner score and test score of each incumbent
  family_selection.csv            which regressor family AutoML picked + per-family test scores (random search)
  encoder_weights.csv             effective kernel weight w_e^2/sum w^2 of the selected models
  featureset_ablation.csv         main vs canonical-view vs ARNIQA-projector features
  shapley.csv / shapley_interactions.csv / subsets.csv   exact Shapley values (test and inner), CIs
  cross_dataset.csv               source x target SROCC
  per_distortion.csv / per_reference.csv                  failure analysis (synthetic sets)
"""
import argparse, json, sys, itertools, math
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import load_config, load_meta, load_splits

SYN = ["kadid10k", "csiq", "tid2013", "live"]


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj


def load_stage1(R, fs="main"):
    rows = []
    for f in (R / "stage1_automl" / fs).glob("*/*/split*/*.json"):
        d = json.loads(f.read_text()); j = d["job"]
        if f.stem == "baselines":
            for name, v in d["baselines"].items():
                rows.append(dict(protocol=j["protocol"], ds=j["ds"], split=j["split"], method=name, kind="baseline",
                                 **v["test"], inner=v["inner"]))
        else:
            rows.append(dict(protocol=j["protocol"], ds=j["ds"], split=j["split"], method=d["strategy"], kind="automl",
                             **d["test"], inner=d["inner"], seconds=d["search_seconds"], family=d["selected"]["family"]))
    return pd.DataFrame(rows)


def summarize(df):
    g = df.groupby(["ds", "method"])
    return g.agg(n=("srocc", "size"), srocc_median=("srocc", "median"), srocc_mean=("srocc", "mean"),
                 srocc_std=("srocc", "std"), plcc_median=("plcc", "median"), plcc_mean=("plcc", "mean"),
                 krcc_median=("krcc", "median"), rmse_median=("rmse", "median")).reset_index()


def significance(df, primary):
    out = []
    for base in sorted(set(df[df.kind == "baseline"].method)):
        rows = []
        for ds in sorted(df.ds.unique()):
            a = df[(df.ds == ds) & (df.method == primary)].set_index("split").srocc
            b = df[(df.ds == ds) & (df.method == base)].set_index("split").srocc
            k = a.index.intersection(b.index); d = (a[k] - b[k]).values
            p = wilcoxon(d).pvalue if len(d) >= 5 and np.any(d != 0) else 1.0
            rows.append(dict(ds=ds, baseline=base, n=len(d), mean_diff=d.mean() if len(d) else np.nan,
                             wins=int((d > 0).sum()), losses=int((d < 0).sum()), p=p))
        adj = holm([r["p"] for r in rows])
        for r, q in zip(rows, adj): r["p_holm"] = q
        out += rows
    return pd.DataFrame(out)


def automl_tables(R, budget_frac_target=0.99):
    bench, any_ = [], []
    for f in (R / "stage1_automl" / "main").glob("*/*/split*/*.json"):
        if f.stem == "baselines": continue
        d = json.loads(f.read_text()); j = d["job"]
        fin = d["test"]["srocc"]; tr = d["trajectory"]
        reach = next((t["fold_evals"] / d["budget"] for t in tr if t["inner"] >= d["inner"] - (1 - budget_frac_target) * abs(d["inner"])), 1.0)
        bench.append(dict(protocol=j["protocol"], ds=j["ds"], split=j["split"], strategy=d["strategy"], test=fin,
                          inner=d["inner"], seconds=d["search_seconds"], eval_seconds=d["eval_seconds"],
                          n_full=d["n_full_evals"], budget_to_99pct_inner=reach))
        for a in d["anytime"]:
            any_.append(dict(protocol=j["protocol"], ds=j["ds"], split=j["split"], strategy=d["strategy"], frac=a["frac"],
                             inner=a["inner"], test=a["test"]["srocc"]))
    b = pd.DataFrame(bench); a = pd.DataFrame(any_)
    bs = b.groupby(["protocol", "ds", "strategy"]).agg(test_median=("test", "median"), test_mean=("test", "mean"),
                                                       inner_median=("inner", "median"), seconds_median=("seconds", "median"),
                                                       n_full_median=("n_full", "median"),
                                                       budget_to_99pct=("budget_to_99pct_inner", "median")).reset_index()
    asum = a.groupby(["protocol", "ds", "strategy", "frac"]).agg(inner_median=("inner", "median"),
                                                                 test_median=("test", "median")).reset_index() if len(a) else a
    return bs, asum


def family_tables(R, primary):
    sel, fam = [], []
    for f in (R / "stage1_automl" / "main" / "nested").glob("*/split*/*.json"):
        if f.stem == "baselines": continue
        d = json.loads(f.read_text()); j = d["job"]
        if d["strategy"] == primary:
            c = d["selected"]; w = np.array([c["w"].get(e, 0) ** 2 for e in sorted(c["w"])]); w = w / max(w.sum(), 1e-12)
            sel.append(dict(ds=j["ds"], split=j["split"], family=c["family"], K=c.get("K"), **{f"weff_{e}": x for e, x in zip(sorted(c["w"]), w)}))
        if d["strategy"] == "random":
            for fa, v in d["per_family"].items():
                fam.append(dict(ds=j["ds"], split=j["split"], family=fa, inner=v["inner"], test=v["test"]["srocc"]))
    s = pd.DataFrame(sel); fm = pd.DataFrame(fam)
    counts = s.groupby(["ds", "family"]).size().unstack(fill_value=0) if len(s) else s
    pf = fm.groupby(["ds", "family"]).agg(inner_median=("inner", "median"), test_median=("test", "median")).reset_index() if len(fm) else fm
    weights = s.filter(like="weff_").groupby(s.ds).agg(["mean", "std"]) if len(s) else s
    return counts, pf, weights


def shapley_from_v(v, players):
    """exact Shapley values and pairwise Shapley interaction indices; v: dict frozenset -> value, v(empty)=0."""
    n = len(players); phi = {}; inter = {}
    fact = [math.factorial(i) for i in range(n + 1)]
    for i in players:
        others = [p for p in players if p != i]; s = 0.0
        for r in range(len(others) + 1):
            for S in itertools.combinations(others, r):
                S = frozenset(S); s += fact[len(S)] * fact[n - len(S) - 1] / fact[n] * (v[S | {i}] - v[S])
        phi[i] = s
    for i, j in itertools.combinations(players, 2):
        others = [p for p in players if p not in (i, j)]; s = 0.0
        for r in range(len(others) + 1):
            for S in itertools.combinations(others, r):
                S = frozenset(S)
                s += fact[len(S)] * fact[n - len(S) - 2] / fact[n - 1] * (v[S | {i, j}] - v[S | {i}] - v[S | {j}] + v[S])
        inter[(i, j)] = s
    return phi, inter


def shapley_tables(R, encoders, n_boot=2000, seed=0):
    rows, irows, srows = [], [], []
    for f in sorted((R / "stage2_shapley").glob("*/split*.jsonl")):
        ds = f.parent.name; split = int(f.stem[5:])
        recs = []
        for l in f.read_text().splitlines():
            try: recs.append(json.loads(l))
            except Exception: pass
        if len({"+".join(r["subset"]) for r in recs}) < 2 ** len(encoders) - 1: continue
        for what in ("test", "inner"):
            v = {frozenset(): 0.0}
            for r in recs:
                v[frozenset(r["subset"])] = r["test"]["srocc"] if what == "test" else r["inner"]
            phi, inter = shapley_from_v(v, encoders)
            for e, x in phi.items():
                rows.append(dict(ds=ds, split=split, value=what, encoder=e, phi=x, share=x / max(v[frozenset(encoders)], 1e-9)))
            for (a, b), x in inter.items():
                irows.append(dict(ds=ds, split=split, value=what, pair=f"{a}x{b}", I=x))
        for r in recs:
            srows.append(dict(ds=ds, split=split, subset="+".join(r["subset"]), size=len(r["subset"]),
                              test=r["test"]["srocc"], inner=r["inner"], family=r["selected"]["family"]))
    sh = pd.DataFrame(rows); it = pd.DataFrame(irows); sub = pd.DataFrame(srows)
    if not len(sh): return sh, it, sub
    rng = np.random.RandomState(seed)

    def ci(x):
        x = np.asarray(x); bs = [rng.choice(x, len(x)).mean() for _ in range(n_boot)]
        return pd.Series({"mean": x.mean(), "lo95": np.percentile(bs, 2.5), "hi95": np.percentile(bs, 97.5), "n": len(x)})
    shs = sh.groupby(["value", "ds", "encoder"]).phi.apply(ci).unstack().reset_index()
    sh["regime"] = np.where(sh.ds.isin(SYN), "synthetic", "authentic")
    reg = sh.groupby(["value", "regime", "encoder"]).phi.apply(ci).unstack().reset_index(); reg["ds"] = reg.pop("regime")
    its = it.groupby(["value", "ds", "pair"]).I.apply(ci).unstack().reset_index()
    subs = sub.groupby(["ds", "subset", "size"]).agg(test_median=("test", "median"), inner_median=("inner", "median")).reset_index()
    return pd.concat([shs, reg]), its, subs


def cross_table(R):
    rows = []
    for f in (R / "stage3_cross_dataset").glob("*.json"):
        d = json.loads(f.read_text())
        for t, r in d["targets"].items():
            for m, v in r.items():
                rows.append(dict(src=d["job"]["src"], tgt=t, method=m, srocc=v["srocc"], plcc=v["plcc"]))
    return pd.DataFrame(rows)


def failure_tables(cfg, R, primary):
    pdist, pref = [], []
    for ds in [d for d in cfg["datasets"] if d in SYN]:
        meta = load_meta(cfg, ds)
        for f in sorted((R / "stage1_automl" / "main" / "nested" / ds).glob("split*/preds.npz")):
            z = np.load(f)
            if primary not in z: continue
            te = z["test_idx"]; y = z["y"]; p = z[primary]; m = meta.iloc[te]
            for dt, idx in m.groupby("dist_type").indices.items():
                if len(idx) > 3: pdist.append(dict(ds=ds, split=f.parent.name, dist_type=dt, srocc=spearmanr(y[idx], p[idx])[0], n=len(idx)))
            s_all = spearmanr(y, p)[0]
            for ref in m.ref.unique():
                pref.append(dict(ds=ds, split=f.parent.name, ref=ref, split_srocc=s_all))
    pdist = pd.DataFrame(pdist); pref = pd.DataFrame(pref)
    d = pdist.groupby(["ds", "dist_type"]).agg(srocc_median=("srocc", "median"), n=("n", "sum")).reset_index() if len(pdist) else pdist
    r = pref.groupby(["ds", "ref"]).agg(mean_split_srocc_when_in_test=("split_srocc", "mean"),
                                        n_splits=("split_srocc", "size")).reset_index().sort_values(["ds", "mean_split_srocc_when_in_test"]) if len(pref) else pref
    return d, r


def md(df, n=60):
    try: return df.head(n).to_markdown(index=False, floatfmt=".4f")
    except Exception: return df.head(n).to_string()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None); ap.add_argument("--primary", default="tpe_hb")
    a = ap.parse_args(); cfg = load_config(a.config)
    R = Path(cfg["paths"]["results"]); T = R / "tables"; T.mkdir(parents=True, exist_ok=True)
    rep = ["# CASH-IQA results", f"primary AutoML strategy (fixed a priori): **{a.primary}**", ""]
    s1 = load_stage1(R)
    if len(s1):
        for prot in ("nested", "holdout"):
            d = s1[s1.protocol == prot]
            if not len(d): continue
            h = summarize(d); h.to_csv(T / f"headline_{prot}.csv", index=False)
            sg = significance(d, a.primary); sg.to_csv(T / f"significance_{prot}.csv", index=False)
            piv = h.pivot(index="method", columns="ds", values="srocc_median")
            rep += [f"## Headline SROCC (median over splits) -- {prot}", md(piv.reset_index()), "",
                    f"### {a.primary} vs baselines ({prot})", md(sg[sg.baseline.isin(['ridge_concat', 'knn_mean_concat', 'local_krr_default'])]), ""]
        bs, an = automl_tables(R); bs.to_csv(T / "automl_benchmark.csv", index=False); an.to_csv(T / "automl_anytime.csv", index=False)
        rep += ["## AutoML strategy benchmark (equal fold-evaluation budget)", md(bs), ""]
        cnt, pf, wt = family_tables(R, a.primary)
        cnt.to_csv(T / "family_selection_counts.csv"); pf.to_csv(T / "family_selection.csv", index=False); wt.to_csv(T / "encoder_weights.csv")
        rep += ["## Regressor family chosen by AutoML (nested, counts over splits)", md(cnt.reset_index()), "",
                "## Per-family best (random search, selected by inner CV)", md(pf), ""]
    abl = []
    for fs in ["main"] + cfg["stage1"]["ablation_featuresets"]:
        d = load_stage1(R, fs)
        if len(d): abl.append(d[(d.protocol == "nested") & (d.method == cfg["stage1"]["ablation_strategy"])].assign(featureset=fs))
    if abl:
        ab = pd.concat(abl).groupby(["ds", "featureset"]).srocc.median().unstack(); ab.to_csv(T / "featureset_ablation.csv")
        rep += ["## Feature-protocol ablation (SROCC median, nested)", md(ab.reset_index()), ""]
    sh, it, sub = shapley_tables(R, cfg["encoders"])
    if len(sh):
        sh.to_csv(T / "shapley.csv", index=False); it.to_csv(T / "shapley_interactions.csv", index=False); sub.to_csv(T / "subsets.csv", index=False)
        rep += ["## Exact Shapley values of encoders (test SROCC game, mean over splits)",
                md(sh[sh.value == "test"].pivot(index="encoder", columns="ds", values="mean").reset_index()), ""]
    cr = cross_table(R)
    if len(cr):
        cr.to_csv(T / "cross_dataset.csv", index=False)
        rep += ["## Cross-dataset SROCC (AutoML)", md(cr[cr.method == "automl"].pivot(index="src", columns="tgt", values="srocc").reset_index()), "",
                "## Cross-dataset SROCC (ridge on concatenated features)", md(cr[cr.method == "ridge_concat"].pivot(index="src", columns="tgt", values="srocc").reset_index()), ""]
    if len(s1):
        pdist, pref = failure_tables(cfg, R, a.primary)
        if len(pdist):   # needs results/stage1_automl/main/nested/*/split*/preds.npz (not shipped in the repo: large)
            pdist.to_csv(T / "per_distortion.csv", index=False); pref.to_csv(T / "per_reference.csv", index=False)
            rep += ["## Hardest reference groups (mean test SROCC of splits containing them)", md(pref.groupby('ds').head(3)), ""]
        else:
            print("no preds.npz found -> keeping the shipped per_distortion.csv / per_reference.csv")
    (R / "report_stage4.md").write_text("\n".join(rep))
    print("\n".join(rep[:60])); print(f"\nwrote {T} and {R/'report_stage4.md'}")


if __name__ == "__main__":
    main()
