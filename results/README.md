# results/

All raw outputs of the reported run, plus every derived table. Paths are produced by `02_automl_pipeline/`.

```
stage1_automl/{featureset}/{protocol}/{dataset}/split{k}/
        random.json  lhs_sh.json  tpe.json  tpe_hb.json  baselines.json     (featureset main)
        tpe_hb.json  baselines.json                                         (featuresets canon, arniqa_proj; nested only)
stage2_shapley/{dataset}/split{k}.jsonl          63 lines: one re-tuned encoder subset each
stage3_cross_dataset/{source}.json               full CASH + all baselines, tested on the 5 other datasets
stage3b_transfer_variants/{global|eqw|global_eqw}/{source}.json      (transfer decomposition)
stage5_noretune_shapley/{dataset}/split{k}.json  63 encoder subsets scored with the stage-1 model held fixed (no re-tuning)
stage6_transfer_shrinkage/cross/{source}.json    stage-3 model with weights shrunk toward equal, 11 alphas, 5 targets
stage6_transfer_shrinkage/indomain/{dataset}/split{k}.json   stage-1 model at 11 alphas: inner CV and test
work/{dataset}/splits.json                       the frozen splits (nested, holdout, cv_full) - regenerate bit-identically with stage0
work/plan.json                                   stage-0 GPU benchmark + run-time projection
logs/                                            stage0_prepare.log, stages1to4_run.log, stages5to6_run.log, smoke_test.log,
                                                 per_job_logs.tar.gz (stages 1, 2, 3, 5, 6)
report_stage4.md                                 auto-generated summary of the stage 1-3b tables
report_stage5.txt                                output of 03_analysis/noretune_shapley_check.py (verdict: AGREES)
report_stage6.txt                                output of 03_analysis/transfer_shrinkage.py (verdict: DOES NOT CLOSE THE GAP)
tables/                                          see below
```

**Counts:** stage 1 has 840 JSON files; stage 2 has 60 JSONL files (3,780 searches); stage 3 has 6 files;
stage 3b has 18 files; stage 5 has 60 files (3,780 subset evaluations, no search); stage 6 has 6 + 60 files.

## tables/

| File | Content | Produced by |
|---|---|---|
| `headline_nested.csv`, `headline_holdout.csv` | per (dataset, method): n, SROCC median / mean / std, PLCC median / mean, KRCC, RMSE | `stage4_analysis.py` |
| `significance_nested.csv`, `significance_holdout.csv` | `tpe_hb` vs each baseline: mean paired difference, wins / losses, Wilcoxon p, Holm-adjusted p (across the 6 datasets) | `stage4_analysis.py` |
| `automl_benchmark.csv` | per strategy: final test / inner median, search seconds, number of full evaluations, budget fraction to reach 99% of the final inner score | `stage4_analysis.py` |
| `automl_anytime.csv` | median test and inner score of the incumbent at 2, 5, 10, 20, 30, 50, 75 and 100% of the budget | `stage4_analysis.py` |
| `family_selection_counts.csv` | family selected by `tpe_hb`, counted over the 10 splits | `stage4_analysis.py` |
| `family_selection.csv` | best configuration *per family* (random search, selected by inner CV): inner and test median | `stage4_analysis.py` |
| `encoder_weights.csv` | effective kernel weight w_e² / Σ w² of the selected models: mean and std over splits | `stage4_analysis.py` |
| `featureset_ablation.csv` | main vs canonical-view vs ARNIQA-projector features (`tpe_hb`, nested, median) | `stage4_analysis.py` |
| `shapley.csv`, `shapley_interactions.csv`, `subsets.csv` | raw-game Shapley values (v(∅) = 0), interactions, the median score of every subset | `stage4_analysis.py` |
| `shapley_gain.csv`, `shapley_interactions_gain.csv`, `shapley_baseline.csv` | fusion-gain game (**used in the paper**) | `03_analysis/shapley_fusion_gain.py` |
| `cross_dataset.csv` | source × target SROCC / PLCC for full CASH and every baseline | `stage4_analysis.py` |
| `cross_transfer.csv` | full CASH, the 3 variants, and ridge_concat per (source, target), with the domain pair | `stage3b_summary.py` |
| `per_distortion.csv` | within-distortion-type SROCC (median over splits) on KADID-10k, CSIQ and TID2013 | `stage4_analysis.py` (needs `preds.npz`) |
| `per_reference.csv` | per reference image: mean whole-split SROCC of the splits that held it out (a coarse indicator of hard content) | `stage4_analysis.py` (needs `preds.npz`) |
| `comparison_published.csv` | our numbers next to the published ones | `03_analysis/compare_published.py` |
| `noretune_shapley_values.csv` | fusion-gain Shapley value per (dataset, game, encoder); games `exact` (stage 2), `fixed_test`, `fixed_inner` (stage 5) | `03_analysis/noretune_shapley_check.py` |
| `noretune_shapley_agreement.csv` | per (dataset, game): Spearman of Shapley values / interactions / subset values with the exact game, max abs difference, sign and top-2 agreement | `03_analysis/noretune_shapley_check.py` |
| `noretune_shapley_scale.csv` | magnitude ratio, single-encoder baselines and fusion gains of both games, bootstrap CI and per-split spread of the rank agreement | `03_analysis/noretune_shapley_check.py` |
| `transfer_shrinkage_pairs.csv` | per (source, target): full CASH, LODO shrinkage and its alpha, alpha = 1, stage-3b variants, ridge | `03_analysis/transfer_shrinkage.py` |
| `transfer_shrinkage_summary.csv`, `_by_source.csv`, `_by_target.csv` | the same averaged by domain pair, by source, by target | `03_analysis/transfer_shrinkage.py` |
| `transfer_shrinkage_curve.csv` | mean transfer SROCC at every alpha (descriptive; not used for selection) | `03_analysis/transfer_shrinkage.py` |
| `transfer_shrinkage_alpha_lodo.csv` | per target: the leave-one-dataset-out selection scores at every alpha and the chosen alpha | `03_analysis/transfer_shrinkage.py` |
| `shrinkage_indomain.csv`, `shrinkage_indomain_tests.csv` | in-domain test / inner SROCC at every alpha per split; alpha 0 vs 1 paired tests and the alpha inner CV prefers | `03_analysis/transfer_shrinkage.py` |

**Method names:**

- `random`, `lhs_sh`, `tpe`, `tpe_hb`: the CASH strategies (`tpe_hb` is primary).
- `ridge_{encoder}`, `ridge_concat`: grid-tuned ridge baselines.
- `knn_mean_concat`, `local_krr_default`: the untuned retrieval baselines.
- Stage 3: `automl` is the full CASH model.
