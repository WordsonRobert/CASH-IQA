# 03 · Analysis (shipped results → paper tables and figures)

These scripts run on CPU in seconds. They need only `../results/`, not the features or the data.

| Script | Reads | Writes | Purpose |
|---|---|---|---|
| `sanity_checks.py` | `results/stage1_automl`, `stage2_shapley`, `stage3*`, `stage5*`, `stage6*` | stdout (exit 1 on any problem) | Completeness (10 splits, 63 subsets, 6 + 18 transfer files, 60 stage-5 files, 6 + 60 stage-6 files); no NaN or zero scores; stage-2 budget accounting; stage-2 budget adequacy (budget 800 vs stage-1 1500 on the all-encoder subset); inner-vs-test gap; exact reproduction of stages 1 and 3 by stages 5 and 6 at their reference setting |
| `shapley_fusion_gain.py` | `results/stage2_shapley` | `tables/shapley_gain.csv`, `shapley_interactions_gain.csv`, `shapley_baseline.csv` | Exact Shapley values and Grabisch interaction indices of the **fusion-gain game**, with 95% bootstrap CIs (5000 resamples over splits), for both the test game and the inner-CV game |
| `compare_published.py` | `published_baselines.csv`, `tables/headline_*.csv` | `tables/comparison_published.csv` | Our numbers next to published ones, grouped by setting |
| `published_baselines.csv` | – | – | 16 published methods with venue, setting, protocol, aggregation (median or mean), per-dataset SROCC, and the source URL of every number |
| `noretune_shapley_check.py` | `results/stage2_shapley`, `stage5_noretune_shapley` | `tables/noretune_shapley_values.csv`, `noretune_shapley_agreement.csv`, `noretune_shapley_scale.csv` | Fusion-gain Shapley values of three games (exact, without re-tuning on test, without re-tuning on inner CV); rank agreement of values, interactions and subset values; the **pre-registered verdict** (ρ_φ ≥ 0.9 on ≥ 5 of 6 datasets); descriptive magnitude ratio, baselines, bootstrap and per-split agreement |
| `transfer_shrinkage.py` | `results/stage3*`, `stage6_transfer_shrinkage` | `tables/transfer_shrinkage_{pairs,summary,curve,alpha_lodo,by_source,by_target}.csv`, `shrinkage_indomain.csv`, `shrinkage_indomain_tests.csv` | Leave-one-dataset-out choice of α per target; transfer by domain pair against full CASH, stage 3b and ridge; the **pre-registered verdict** (above full CASH and within 0.010 of ridge); in-domain cost of shrinking with paired tests |
| `make_figures.py` | `results/tables/*.csv`, `results/stage1_automl` | `figures/fig1…fig11.{pdf,png}` | All paper figures |

## Why a second Shapley script?

`02_automl_pipeline/stage4_analysis.py` uses the textbook game: v(S) is the test SROCC of the model re-tuned
on subset S, and v(∅) = 0. Because every single encoder already scores 0.6–0.9, that convention has two
problems:

- Most of each φ is "credit for existing", about 1/6 of the total.
- Every pairwise interaction is strongly negative, because two encoders are never worth twice one.

`shapley_fusion_gain.py` re-baselines the game per split:

```
g(S) = v(S) − c,      c = mean_e v({e}),      g(∅) = 0
```

Then Σ_e φ_e = v(all) − c, which is the gain of fusion over an average single encoder. Within a split, every φ
moves by the same constant (−c/n), so rankings are unchanged, and interactions become readable:

- I_ij < 0 means redundancy;
- I_ij > 0 means synergy.

Both versions are shipped (`shapley.csv`, `shapley_interactions.csv` vs `shapley_gain.csv`,
`shapley_interactions_gain.csv`). The paper uses the fusion-gain version.

## Stages 5 and 6: what was fixed before the run

Both follow-up analyses had their decision rule written into the scripts before any result existed, and the
verdict lines are printed exactly as the rule dictates:

- **Stage 5:** the no-re-tuning game "agrees" with the exact one if the Spearman correlation of the six
  fusion-gain Shapley values is ≥ 0.9 on at least 5 of the 6 datasets. Result: AGREES (5/6 on test, 5/6 on
  inner CV).
- **Stage 6:** for target T, α*_T maximises the mean SROCC over the 20 transfer pairs among the other five
  datasets (ties to the larger α); shrinkage "closes the gap" if its mean over the 30 pairs is above full CASH
  and within 0.010 of ridge. Result: α* = 1 for every target, 0.709 vs 0.652 (full) and 0.742 (ridge):
  DOES NOT CLOSE THE GAP.

Everything printed below the line `descriptive (not part of the criterion)` was added afterwards to explain the
results; it changes no verdict.

## The published comparison: caveats that matter

- **SLIQUE reports means;** everything else reports medians. Our means are in `tables/headline_*.csv`.
- **Protocols differ.** CONTRIQUE, ARNIQA and SLIQUE use 70/10/20 and are matched by our holdout protocol;
  GRepQ, TRIQA and Gamma-T use 80/20. Our nested protocol (80/20 with selection on inner folds) is the stricter
  of the two.
- **Some sources are not peer-reviewed or not training-free:**
  - SHAMISA (arXiv 2026) reports CSIQ 0.981, which is above the fine-tuned ceiling.
  - "Regression-free kNN" fine-tunes its distortion module on IQA labels, so it is not training-free in our
    sense.
- **Ceiling vs label-free:** the fine-tuned rows are an upper reference, not competitors; the no-label rows
  are a lower reference.
