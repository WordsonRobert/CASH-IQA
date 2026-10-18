# 03 · Analysis (shipped results → paper tables and figures)

These scripts run on CPU in seconds. They need only `../results/`, not the features or the data.

| Script | Reads | Writes | Purpose |
|---|---|---|---|
| `sanity_checks.py` | `results/stage1_automl`, `stage2_shapley`, `stage3*` | stdout (exit 1 on any problem) | Completeness (10 splits, 63 subsets, 6 + 18 transfer files); no NaN or zero scores; stage-2 budget accounting; stage-2 budget adequacy (budget 800 vs stage-1 1500 on the all-encoder subset); inner-vs-test gap |
| `shapley_fusion_gain.py` | `results/stage2_shapley` | `tables/shapley_gain.csv`, `shapley_interactions_gain.csv`, `shapley_baseline.csv` | Exact Shapley values and Grabisch interaction indices of the **fusion-gain game**, with 95% bootstrap CIs (5000 resamples over splits), for both the test game and the inner-CV game |
| `compare_published.py` | `published_baselines.csv`, `tables/headline_*.csv` | `tables/comparison_published.csv` | Our numbers next to published ones, grouped by setting |
| `published_baselines.csv` | – | – | 16 published methods with venue, setting, protocol, aggregation (median or mean), per-dataset SROCC, and the source URL of every number |
| `make_figures.py` | `results/tables/*.csv`, `results/stage1_automl` | `figures/fig1…fig9.{pdf,png}` | All paper figures |

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
