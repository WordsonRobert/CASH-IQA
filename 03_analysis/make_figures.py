#!/usr/bin/env python3
"""
All paper figures from the shipped results (CPU, ~10 s).  python make_figures.py  -> ../figures/figN_*.pdf + .png

  fig1_paired_gain_over_ridge     per-split SROCC gain of the tuned model over ridge on concatenated features
  fig2_anytime_curves             incumbent test SROCC vs share of the fold-evaluation budget, per dataset
  fig3_family_selection           which regressor family the search selects, per dataset (10 nested splits)
  fig4_shapley_fusion_gain        exact Shapley share of the fusion gain, encoder x dataset (95% CI)
  fig5_shapley_interactions       pairwise Shapley interaction indices, synthetic vs authentic means
  fig6_cross_dataset_matrices     source x target SROCC: full CASH, equal-weight global variant, ridge concat
  fig7_transfer_decomposition     mean cross-dataset SROCC by domain pair for the five transfer variants
  fig8_per_distortion             within-type SROCC per distortion type (KADID-10k, TID2013)
  fig9_encoder_weights            effective fusion weight w_e^2 / sum w^2 of the selected models
"""
import json, glob
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]; R = ROOT / "results"; T = R / "tables"; OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
DS = ["kadid10k", "csiq", "tid2013", "koniq10k", "livec", "spaq"]
NAME = {"kadid10k": "KADID-10k", "csiq": "CSIQ", "tid2013": "TID2013", "koniq10k": "KonIQ-10k", "livec": "LIVE Challenge", "spaq": "SPAQ"}
SYN = {"kadid10k", "csiq", "tid2013"}
E = ["arniqa", "contrique", "resnet50", "dinov2", "clip", "mae"]
ENAME = {"arniqa": "ARNIQA", "contrique": "CONTRIQUE", "resnet50": "ResNet-50", "dinov2": "DINOv2", "clip": "CLIP", "mae": "MAE"}
ACC, GREY, LGREY, DARK = "#2b6cb0", "#8a8f98", "#c9ccd1", "#1f2328"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 10,
                     "axes.titleweight": "bold", "figure.dpi": 150, "savefig.bbox": "tight", "font.family": "DejaVu Sans"})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200 if ext == "png" else None)
    plt.close(fig); print("wrote", OUT / f"{name}.pdf")


def stage1_rows(protocol):
    rows = []
    for f in glob.glob(str(R / "stage1_automl" / "main" / protocol / "*" / "split*" / "*.json")):
        d = json.load(open(f)); j = d["job"]
        if "baselines" in d:
            for n, v in d["baselines"].items(): rows.append(dict(ds=j["ds"], split=j["split"], method=n, srocc=v["test"]["srocc"]))
        else:
            rows.append(dict(ds=j["ds"], split=j["split"], method=d["strategy"], srocc=d["test"]["srocc"]))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- fig1
def fig1():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, prot in zip(axes, ["nested", "holdout"]):
        d = stage1_rows(prot); p = d.pivot_table(index=["ds", "split"], columns="method", values="srocc")
        gain = (p["tpe_hb"] - p["ridge_concat"]).reset_index(name="gain")
        for i, ds in enumerate(DS):
            g = gain[gain.ds == ds].gain.values
            x = i + (np.random.default_rng(i).uniform(-0.12, 0.12, len(g)))
            ax.scatter(x, g, s=14, color=ACC if ds in SYN else GREY, alpha=0.75, lw=0)
            ax.plot([i - 0.25, i + 0.25], [np.median(g)] * 2, color=DARK, lw=1.6)
            ax.text(i, max(g.max(), 0) + 0.006, f"{(g > 0).sum()}/10", ha="center", fontsize=8, color=DARK)
        ax.axhline(0, color=DARK, lw=0.8); ax.set_xticks(range(len(DS))); ax.set_xticklabels([NAME[d] for d in DS], rotation=30, ha="right")
        ax.set_title(f"{prot} protocol"); ax.grid(axis="y", color="#eceef1", lw=0.6)
    axes[0].set_ylabel("SROCC gain over ridge (concat)\nper split; bar = median")
    fig.suptitle("Tuned CASH model vs ridge on concatenated features (TPE + Hyperband, 10 splits)", fontsize=10, y=1.03)
    save(fig, "fig1_paired_gain_over_ridge")


# ---------------------------------------------------------------- fig2
def fig2():
    a = pd.read_csv(T / "automl_anytime.csv"); a = a[a.protocol == "nested"]
    style = {"random": (LGREY, "--", "Random search"), "lhs_sh": (DARK, None, "LHS + successive halving"),
             "tpe": (GREY, "-", "TPE"), "tpe_hb": (ACC, "-", "TPE + Hyperband")}
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.4))
    for ax, ds in zip(axes.flat, DS):
        d = a[a.ds == ds]
        for s, (c, ls, lab) in style.items():
            e = d[d.strategy == s].sort_values("frac")
            if s == "lhs_sh":
                ax.scatter(e.frac, e.test_median, marker="s", facecolor="none", edgecolor=c, s=28, label=lab, zorder=3)
            else:
                ax.plot(e.frac, e.test_median, ls, color=c, lw=2.2 if s == "tpe_hb" else 1.3, marker="o", ms=3, label=lab)
        ax.set_xscale("log"); ax.set_xticks([0.02, 0.05, 0.1, 0.2, 0.5, 1]); ax.set_xticklabels(["2%", "5%", "10%", "20%", "50%", "100%"])
        ax.set_title(NAME[ds]); ax.grid(color="#eceef1", lw=0.6)
    axes[0, 0].legend(frameon=False, fontsize=7.5, loc="lower right")
    for ax in axes[:, 0]: ax.set_ylabel("test SROCC of incumbent")
    for ax in axes[1]: ax.set_xlabel("share of 1,500 fold-evaluation budget")
    fig.suptitle("Anytime performance of the four CASH strategies (nested; median over 10 splits; incumbents chosen on inner CV)", fontsize=10)
    fig.tight_layout(); save(fig, "fig2_anytime_curves")


# ---------------------------------------------------------------- fig3
def fig3():
    c = pd.read_csv(T / "family_selection_counts.csv").set_index("ds").reindex(DS)
    fams = ["nw", "local_krr_lin", "local_krr_rbf", "lwlr", "global_krr_lin", "global_krr_rbf"]
    lab = {"nw": "Nadaraya-Watson", "local_krr_lin": "local KRR (linear)", "local_krr_rbf": "local KRR (RBF)", "lwlr": "LWLR",
           "global_krr_lin": "global KRR (linear)", "global_krr_rbf": "global KRR (RBF)"}
    cols = ["#9cc3e6", "#5b9bd5", "#2b6cb0", "#173f6b", "#c9ccd1", "#8a8f98"]
    fig, ax = plt.subplots(figsize=(7, 3)); left = np.zeros(len(DS))
    for f, col in zip(fams, cols):
        v = c[f].values; ax.barh(range(len(DS)), v, left=left, color=col, label=lab[f], edgecolor="white", lw=0.6); left += v
    ax.set_yticks(range(len(DS))); ax.set_yticklabels([NAME[d] for d in DS]); ax.invert_yaxis(); ax.set_xlabel("nested splits (of 10)")
    ax.axhline(2.5, color=DARK, lw=0.8, ls=":"); ax.text(10.1, 1, "synthetic", va="center", fontsize=8); ax.text(10.1, 4, "authentic", va="center", fontsize=8)
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    ax.set_title("Regressor family selected by TPE + Hyperband: local (blue) on synthetic, global (grey) on authentic")
    save(fig, "fig3_family_selection")


# ---------------------------------------------------------------- fig4
def fig4():
    s = pd.read_csv(T / "shapley_gain.csv"); s = s[s.value == "test"]
    m = s.pivot(index="encoder", columns="ds", values="mean").reindex(index=E, columns=DS)
    lo = s.pivot(index="encoder", columns="ds", values="lo95").reindex(index=E, columns=DS)
    hi = s.pivot(index="encoder", columns="ds", values="hi95").reindex(index=E, columns=DS)
    b = pd.read_csv(T / "shapley_baseline.csv"); b = b[b.value == "test"].set_index("ds").reindex(DS)
    fig, ax = plt.subplots(figsize=(7.6, 3.6)); vmax = np.abs(m.values).max()
    im = ax.imshow(m.values, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(len(E)):
        for j in range(len(DS)):
            sig = lo.values[i, j] > 0 or hi.values[i, j] < 0
            ax.text(j, i, f"{m.values[i, j]:+.3f}" + ("" if sig else "\n(n.s.)"), ha="center", va="center", fontsize=7.5,
                    color="white" if abs(m.values[i, j]) > 0.6 * vmax else DARK)
    ax.set_xticks(range(len(DS))); ax.set_xticklabels([f"{NAME[d]}\ngain {b.loc[d, 'fusion_gain']:+.3f}" for d in DS], fontsize=8)
    ax.set_yticks(range(len(E))); ax.set_yticklabels([ENAME[e] for e in E]); ax.axvline(2.5, color=DARK, lw=1)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02).set_label("Shapley share (SROCC)")
    ax.set_title("Exact Shapley attribution of the fusion gain over the mean single encoder (test; mean of 10 splits)")
    save(fig, "fig4_shapley_fusion_gain")


# ---------------------------------------------------------------- fig5
def fig5():
    it = pd.read_csv(T / "shapley_interactions_gain.csv"); it = it[it.value == "test"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.9))
    for ax, (grp, sel) in zip(axes, [("synthetic", SYN), ("authentic", set(DS) - SYN)]):
        M = np.full((len(E), len(E)), np.nan)
        g = it[it.ds.isin(sel)].groupby("pair")["mean"].mean()
        for p, v in g.items():
            a, c = p.split("x"); i, j = E.index(a), E.index(c); M[i, j] = M[j, i] = v
        vmax = np.nanmax(np.abs(M)); im = ax.imshow(M, cmap="RdBu", vmin=-vmax, vmax=vmax)
        for i in range(len(E)):
            for j in range(len(E)):
                if i != j: ax.text(j, i, f"{M[i, j]:+.3f}", ha="center", va="center", fontsize=7, color="white" if abs(M[i, j]) > 0.6 * vmax else DARK)
        ax.set_xticks(range(len(E))); ax.set_xticklabels([ENAME[e] for e in E], rotation=40, ha="right"); ax.set_yticks(range(len(E))); ax.set_yticklabels([ENAME[e] for e in E])
        ax.set_title(f"{grp} datasets (mean)"); fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    fig.suptitle("Pairwise Shapley interactions on the fusion gain: red = redundancy, blue = synergy", fontsize=10)
    fig.tight_layout(); save(fig, "fig5_shapley_interactions")


# ---------------------------------------------------------------- fig6
def fig6():
    c = pd.read_csv(T / "cross_transfer.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (m, title) in zip(axes, [("full", "full CASH (stage 3)"), ("global_eqw", "global kernels, equal weights (stage 3b)"), ("ridge_concat", "ridge on concatenated features")]):
        p = c[c.method == m].pivot(index="src", columns="tgt", values="srocc").reindex(index=DS, columns=DS)
        im = ax.imshow(p.values, cmap="Blues", vmin=0.1, vmax=0.95)
        for i in range(len(DS)):
            for j in range(len(DS)):
                if i != j: ax.text(j, i, f"{p.values[i, j]:.2f}", ha="center", va="center", fontsize=7.5, color="white" if p.values[i, j] > 0.7 else DARK)
        ax.set_xticks(range(len(DS))); ax.set_xticklabels([NAME[d] for d in DS], rotation=40, ha="right"); ax.set_yticks(range(len(DS))); ax.set_yticklabels([NAME[d] for d in DS])
        ax.set_title(f"{title}\nmean {np.nanmean(p.values):.3f}"); ax.set_xlabel("target"); ax.set_ylabel("source")
    fig.suptitle("Cross-dataset SROCC (tuned on source by 5-fold grouped CV, scored on all target images)", fontsize=10)
    fig.tight_layout(); save(fig, "fig6_cross_dataset_matrices")


# ---------------------------------------------------------------- fig7
def fig7():
    c = pd.read_csv(T / "cross_transfer.csv")
    pairs = [("A->A", "authentic -> authentic"), ("A->S", "authentic -> synthetic"), ("S->A", "synthetic -> authentic"), ("S->S", "synthetic -> synthetic")]
    meth = [("full", "full CASH", GREY, "o", True), ("global", "global kernels, tuned weights", GREY, "s", False),
            ("eqw", "all families, equal weights", GREY, "o", False), ("global_eqw", "global kernels, equal weights", ACC, "o", True)]
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    rows = [(k, lab, c[c.pair == k]) for k, lab in pairs] + [("ALL", "all 30 pairs", c)]
    for i, (k, lab, d) in enumerate(rows):
        mu = d.groupby("method").srocc.mean()
        ax.plot([mu["ridge_concat"]] * 2, [i - 0.28, i + 0.28], color=DARK, lw=2)
        for m, ml, col, mk, filled in meth:
            ax.scatter(mu[m], i, s=46 if m in ("full", "global_eqw") else 30, marker=mk, color=col if filled else "none", edgecolor=col, lw=1.2, zorder=3)
        ax.text(mu["full"], i + 0.33, f"{mu['full']:.3f}", ha="center", fontsize=7, color=GREY)
        ax.text(mu["global_eqw"], i - 0.42, f"{mu['global_eqw']:.3f}", ha="center", fontsize=7, color=ACC)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[1] for r in rows]); ax.invert_yaxis(); ax.set_xlabel("mean cross-dataset SROCC")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker=mk, ls="", color=col, markerfacecolor=col if f else "none", label=ml) for m, ml, col, mk, f in meth] + [Line2D([], [], color=DARK, lw=2, label="ridge (concat)")]
    ax.legend(handles=h, frameon=False, fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.45, -0.18)); ax.grid(axis="x", color="#eceef1", lw=0.6)
    ax.set_title("Equal fusion weights close the cross-dataset gap to ridge")
    save(fig, "fig7_transfer_decomposition")


# ---------------------------------------------------------------- fig8
KADID = {"k01": "Gaussian blur", "k02": "lens blur", "k03": "motion blur", "k04": "colour diffusion", "k05": "colour shift", "k06": "colour quantisation",
         "k07": "colour saturation 1", "k08": "colour saturation 2", "k09": "JPEG2000", "k10": "JPEG", "k11": "white noise", "k12": "white noise (colour)",
         "k13": "impulse noise", "k14": "multiplicative noise", "k15": "denoise", "k16": "brighten", "k17": "darken", "k18": "mean shift", "k19": "jitter",
         "k20": "non-eccentricity patch", "k21": "pixelate", "k22": "quantisation", "k23": "colour block", "k24": "high sharpen", "k25": "contrast change"}
TID = {"t01": "additive Gaussian noise", "t02": "noise in colour comps", "t03": "spatially correlated noise", "t04": "masked noise", "t05": "high-freq noise",
       "t06": "impulse noise", "t07": "quantisation noise", "t08": "Gaussian blur", "t09": "denoising", "t10": "JPEG", "t11": "JPEG2000",
       "t12": "JPEG transm. errors", "t13": "JPEG2000 transm. errors", "t14": "non-eccentricity pattern", "t15": "local block-wise", "t16": "mean shift",
       "t17": "contrast change", "t18": "colour saturation", "t19": "multiplicative noise", "t20": "comfort noise", "t21": "lossy compr. of noisy",
       "t22": "colour quant. + dither", "t23": "chromatic aberrations", "t24": "sparse sampling"}


def fig8():
    p = pd.read_csv(T / "per_distortion.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2))
    for ax, ds, names in [(axes[0], "kadid10k", KADID), (axes[1], "tid2013", TID)]:
        d = p[p.ds == ds].copy(); d["name"] = d.dist_type.map(names).fillna(d.dist_type); d = d.sort_values("srocc_median")
        cols = [ACC if v < 0.75 else LGREY for v in d.srocc_median]
        ax.barh(d.name, d.srocc_median, color=cols); ax.set_xlim(0, 1); ax.set_title(NAME[ds]); ax.set_xlabel("within-type SROCC (median over test splits)")
        for y, v in enumerate(d.srocc_median): ax.text(v + 0.01, y, f"{v:.2f}", va="center", fontsize=6.5)
        ax.tick_params(axis="y", labelsize=7)
    fig.suptitle("Per-distortion difficulty: content-preserving photometric/local changes fail on both datasets (highlighted < 0.75)", fontsize=10)
    fig.tight_layout(); save(fig, "fig8_per_distortion")


# ---------------------------------------------------------------- fig9
def fig9():
    w = pd.read_csv(T / "encoder_weights.csv", header=[0, 1], index_col=0)
    mean = pd.DataFrame({e: w[(f"weff_{e}", "mean")] for e in E}).reindex(DS)
    fig, ax = plt.subplots(figsize=(7, 3)); left = np.zeros(len(DS))
    cols = {"arniqa": "#173f6b", "contrique": "#2b6cb0", "resnet50": "#c9ccd1", "dinov2": "#a7abb2", "clip": "#e0a03a", "mae": "#8a8f98"}
    for e in E:
        v = mean[e].values; ax.barh(range(len(DS)), v, left=left, color=cols[e], label=ENAME[e], edgecolor="white", lw=0.6); left += v
    ax.set_yticks(range(len(DS))); ax.set_yticklabels([NAME[d] for d in DS]); ax.invert_yaxis(); ax.set_xlim(0, 1)
    ax.set_xlabel("effective fusion weight  w_e^2 / sum w^2  (mean over 10 nested splits)")
    ax.legend(frameon=False, fontsize=7.5, ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    ax.set_title("Fusion weights chosen by the search: IQA encoders dominate on synthetic data")
    save(fig, "fig9_encoder_weights")


if __name__ == "__main__":
    for f in [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9]: f()
