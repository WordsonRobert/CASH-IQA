#!/usr/bin/env python3
"""
Does attribution without re-tuning (stage 5) reproduce the exact re-tuned Shapley attribution (stage 2)?

Three games per dataset, each in the fusion-gain form of shapley_fusion_gain.py (g(S) = v(S) - c,
c = mean single-encoder score, values averaged over the 10 outer splits):
  exact        stage 2: every subset re-tuned by tpe_hb (budget 800), test SROCC       <- reference
  fixed_test   stage 5: stage-1 configuration restricted to S, no re-tuning, test SROCC
  fixed_inner  stage 5: same, inner-CV SROCC (uses no test data at all)

Agreement measures (per dataset, against `exact`):
  rho_phi     Spearman correlation of the 6 Shapley values
  rho_int     Spearman correlation of the 15 pairwise interaction indices
  rho_v       Spearman correlation of the 63 subset values v(S)
  max_dphi    largest absolute difference of a Shapley value
  sign_agree  encoders whose Shapley value has the same sign
  top2_same   the two highest-ranked encoders coincide (as a set)

Pre-registered criterion: `fixed_test` agrees with `exact` when rho_phi >= 0.9 on at least 5 of 6 datasets.

Descriptive outputs (added after the run; they do not enter the criterion):
  scale       least-squares slope of phi_fixed on phi_exact through the origin (1 = same magnitude)
  c / gain    mean single-encoder score and fusion gain v(all) - c in each game
  single_gap  v_exact({e}) - v_fixed({e}) per encoder and its mean: the cost of not re-tuning a single encoder
  rho CI      95% bootstrap interval of rho_phi (2000 resamples of the outer splits)
  per-split   median / minimum of rho_phi computed split by split (no averaging over splits)

    python noretune_shapley_check.py      # writes ../results/tables/noretune_shapley_{values,agreement}.csv
"""
import json, glob, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parent))
from shapley_fusion_gain import game, shapley, interactions, E, DS

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"; T = R / "tables"
RHO_MIN, MIN_DATASETS = 0.9, 5


def load_exact(ds):
    out = {}
    for fn in sorted(glob.glob(str(R / "stage2_shapley" / ds / "split*.jsonl"))):
        recs = {}
        for line in open(fn):
            try: r = json.loads(line)
            except Exception: continue
            recs[tuple(r["subset"])] = {"test": r["test"]["srocc"], "inner": r["inner"], "seconds": r.get("seconds", 0.0)}
        if len(recs) == 2 ** len(E) - 1:
            out[int(Path(fn).stem[5:])] = recs
    return out


def load_fixed(ds):
    out, meta = {}, {}
    for fn in sorted(glob.glob(str(R / "stage5_noretune_shapley" / ds / "split*.json"))):
        d = json.loads(open(fn).read()); k = int(Path(fn).stem[5:])
        out[k] = {tuple(r["subset"]): {"test": r["test"]["srocc"], "inner": r["inner"]} for r in d["records"]}
        meta[k] = d
    return out, meta


def mean_game(per_split, key):
    phis, ints, vs, cs = [], [], [], []
    for recs in per_split.values():
        g, c = game(recs, key); phis.append(shapley(g)); ints.append(interactions(g)); cs.append(c)
        vs.append({frozenset(s): r[key] for s, r in recs.items()})
    phi = {e: float(np.mean([p[e] for p in phis])) for e in E}
    inter = {p: float(np.mean([i[p] for i in ints])) for p in ints[0]}
    v = {S: float(np.mean([x[S] for x in vs])) for S in vs[0]}
    return phi, inter, v, float(np.mean(cs))


def descriptive(ds, ks, ex, fx, ref_phi, ref_v, ref_c, g_test, g_inner, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed); full = frozenset(E)
    a = np.array([ref_phi[e] for e in E]); b = np.array([g_test[0][e] for e in E]); c = np.array([g_inner[0][e] for e in E])
    per_e = [shapley(game(ex[k], "test")[0]) for k in ks]; per_f = [shapley(game(fx[k], "test")[0]) for k in ks]
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(ks), len(ks))
        boot.append(spearmanr([np.mean([per_e[i][e] for i in idx]) for e in E], [np.mean([per_f[i][e] for i in idx]) for e in E])[0])
    split_rho = [spearmanr([per_e[i][e] for e in E], [per_f[i][e] for e in E])[0] for i in range(len(ks))]
    gaps = {e: float(np.mean([ex[k][(e,)]["test"] - fx[k][(e,)]["test"] for k in ks])) for e in E}
    single_gap = np.mean(list(gaps.values()))
    return dict(ds=ds, c_exact=ref_c, c_fixed=g_test[3], gain_exact=ref_v[full] - ref_c, gain_fixed=g_test[2][full] - g_test[3],
                single_gap=single_gap, scale_test=float(a @ b / (a @ a)), scale_inner=float(a @ c / (a @ a)),
                rho_phi_lo95=float(np.nanpercentile(boot, 2.5)), rho_phi_hi95=float(np.nanpercentile(boot, 97.5)),
                rho_phi_split_median=float(np.median(split_rho)), rho_phi_split_min=float(np.min(split_rho)),
                **{f"single_gap_{e}": gaps[e] for e in E})


def main():
    rows_v, rows_a, checks, rows_x = [], [], [], []
    secs_exact = secs_fixed = 0.0
    for ds in DS:
        ex = load_exact(ds); fx, meta = load_fixed(ds)
        ks = sorted(set(ex) & set(fx))
        if not ks:
            print(f"{ds}: no overlapping splits yet, skipped"); continue
        ex = {k: ex[k] for k in ks}; fx = {k: fx[k] for k in ks}
        secs_exact += sum(r["seconds"] for recs in ex.values() for r in recs.values())
        secs_fixed += sum(meta[k]["seconds"] for k in ks)
        for k in ks:
            checks.append(dict(ds=ds, split=k, family=meta[k]["base_family"],
                               full_test_diff=meta[k]["check_full_test_diff"], full_inner_diff=meta[k]["check_full_inner_diff"]))
        ref_phi, ref_int, ref_v, ref_c = mean_game(ex, "test")
        g_test, g_inner = mean_game(fx, "test"), mean_game(fx, "inner")
        games = {"exact": (ref_phi, ref_int, ref_v), "fixed_test": g_test[:3], "fixed_inner": g_inner[:3]}
        rows_x.append(descriptive(ds, ks, ex, fx, ref_phi, ref_v, ref_c, g_test, g_inner))
        for name, (phi, inter, v) in games.items():
            for e in E:
                rows_v.append(dict(ds=ds, game=name, encoder=e, phi=phi[e]))
            if name == "exact": continue
            a = np.array([ref_phi[e] for e in E]); b = np.array([phi[e] for e in E])
            pi = list(ref_int); Ss = list(ref_v)
            top = lambda d: set(sorted(E, key=lambda e: -d[e])[:2])
            rows_a.append(dict(ds=ds, game=name, n_splits=len(ks),
                               rho_phi=spearmanr(a, b)[0],
                               rho_int=spearmanr([ref_int[p] for p in pi], [inter[p] for p in pi])[0],
                               rho_v=spearmanr([ref_v[S] for S in Ss], [v[S] for S in Ss])[0],
                               max_dphi=float(np.abs(a - b).max()),
                               sign_agree=int((np.sign(a) == np.sign(b)).sum()),
                               top2_same=top(ref_phi) == top(phi)))
    if not rows_a:
        sys.exit("no stage-5 results found")
    V, A, C = pd.DataFrame(rows_v), pd.DataFrame(rows_a), pd.DataFrame(checks)
    T.mkdir(parents=True, exist_ok=True)
    V.to_csv(T / "noretune_shapley_values.csv", index=False); A.to_csv(T / "noretune_shapley_agreement.csv", index=False)
    pd.set_option("display.width", 200)
    print("check: fixed game on the full encoder set vs stage-1 tpe_hb (must be ~0):")
    print(f"  max |test diff| = {C.full_test_diff.abs().max():.2e}   max |inner diff| = {C.full_inner_diff.abs().max():.2e}"
          f"   over {len(C)} splits")
    print("\nfusion-gain Shapley values (mean over splits):")
    print(V.pivot_table(index=["encoder"], columns=["ds", "game"], values="phi").reindex(E).round(3).to_string())
    print("\nagreement with the exact re-tuned game:")
    print(A.round(3).to_string(index=False))
    for g in ["fixed_test", "fixed_inner"]:
        a = A[A.game == g]; n_ok = int((a.rho_phi >= RHO_MIN).sum())
        verdict = "AGREES" if (n_ok >= MIN_DATASETS and len(a) == len(DS)) else ("DOES NOT AGREE" if len(a) == len(DS) else "INCOMPLETE")
        print(f"\n{g}: rho_phi >= {RHO_MIN} on {n_ok}/{len(a)} datasets  ->  {verdict} (criterion: >= {MIN_DATASETS} of {len(DS)})")
    if secs_fixed > 0:
        print(f"\ncompute: exact searches {secs_exact/3600:.2f} worker-h vs fixed game {secs_fixed/3600:.3f} worker-h "
              f"(x{secs_exact/secs_fixed:.0f})")
    X = pd.DataFrame(rows_x); X.to_csv(T / "noretune_shapley_scale.csv", index=False)
    print("\ndescriptive (not part of the criterion): magnitude, baseline and split-level agreement")
    gcols = [c for c in X if c.startswith("single_gap_")]
    print(X.drop(columns=gcols).round(3).to_string(index=False))
    print("\nsingle-encoder test SROCC lost by not re-tuning, v_exact({e}) - v_fixed({e}) (mean over splits):")
    print(X.set_index("ds")[gcols].rename(columns=lambda c: c[len("single_gap_"):]).round(3).to_string())
    print(f"\nwrote {T}/noretune_shapley_values.csv, noretune_shapley_agreement.csv, noretune_shapley_scale.csv")


if __name__ == "__main__":
    main()
