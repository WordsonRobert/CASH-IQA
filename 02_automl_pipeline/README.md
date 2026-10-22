# 02 · AutoML pipeline (the experiment)

This folder holds the complete experiment, as it was run for the paper. Every setting is in `config.yaml`, and
the frozen splits are in `../results/work/`.

## Method in brief

Each encoder block is L2-normalised, so the cosine similarity of the weighted concatenation
`[w_1 x_1 | … | w_6 x_6]` decomposes **exactly** into per-encoder similarities:

```
S(x, x') = Σ_e w_e² S_e(x, x') / Σ_e w_e²          S_e = cosine Gram of encoder e
```

The consequences:

- `core.py::Bank` computes one N × N Gram matrix per encoder once per job. Every weight vector the search tries
  is then a weighted sum (`Bank.fused`), with no feature recomputation.
- The distance used by local models is `d = 2 − 2S`, the squared Euclidean distance of the normalised
  concatenation.

**Regressor families (`core.py::predict`):**

| Family | Prediction for a query x with neighbourhood N_K(x) (top-K by S) |
|---|---|
| `nw` | Nadaraya–Watson: Σ_i a_i y_i / Σ_i a_i, with a_i = exp(−γ d_i) |
| `local_krr_lin` | μ + k_qᵀ (K_N + λI)⁻¹ (y_N − μ), with the linear kernel S; μ = mean(y_N) if `center`, else 0 |
| `local_krr_rbf` | the same with the RBF kernel exp(−γ d) |
| `lwlr` | locally weighted ridge on [x, 1] with weights exp(−γ d_i), solved in the exact K × K dual: (R(S_N + 1)R + λI) v = R y, with ŷ = Σ (S_q,i + 1) r_i v_i and r = exp(−γ d / 2) |
| `global_krr_lin` | kernel ridge over the whole gallery with the linear kernel S + 1 (bias), on centred y; fp64 Cholesky with an lstsq fallback |
| `global_krr_rbf` | the same with exp(−γ d) |

`global_krr_lin` with one-hot weights is exactly ridge regression on one encoder's features. With equal weights
it is ridge on the concatenation (normalised per block). The baselines use exactly that.

**CASH search space (`core.py::Space`).** The search draws:

- the family;
- w_e ∈ [0, 1] for each of the six encoders (none fixed);
- K ∈ [5, 200] (log-integer, local families only);
- log10 γ ∈ [−2, 2];
- log10 λ ∈ [−4, 2];
- `center` ∈ {F, T} (local KRR only).

Parameters a family ignores are dropped (`Space.clean`), so identical models get identical cache keys.

**Search strategies (`core.py::search`).** Every strategy has the same budget, counted in **fold-evaluations**:
one evaluation of one configuration on one inner fold. `Ledger` enforces the budget and caches (config, fold)
scores, so a re-evaluation is free.

| Strategy | Implementation |
|---|---|
| `random` | uniform draws, each scored on all 5 inner folds (`run_random`) |
| `lhs_sh` | `run_lhs_sh`: a Latin hypercube of n0 = 900 configurations, then successive halving over fidelity = number of inner folds, rungs (1, 2, 5), η = 3. The first full-fidelity result arrives only after about 80% of the budget. |
| `tpe` | Optuna multivariate grouped TPE, 20 start-up trials, every trial scored on all 5 folds |
| `tpe_hb` | TPE plus Hyperband pruning (min resource 1 fold, max 5, reduction factor 3); fold order permuted per seed. **The primary strategy, fixed before the run.** |

**Leak-proofing (`core.py::Plan`).**

- A `Plan` holds the search folds, which are the only data the search can touch. It also holds the final
  `(test, gallery)` pair, reachable only through `Plan.final()`.
- `final()` asserts that `freeze()` was called. After freezing, `eval_fold()` raises an error.
- Splits are built by `make_splits` and checked by `check_splits`, which asserts that no group crosses
  outer, inner or holdout boundaries.
- Seeds come from `job_seed(...)` (sha256), so every job is reproducible on its own.

## Stages

| Stage | Script | One job = | What it does | Output (`../results/…`) | Time (A100-80GB, 6 workers) |
|---|---|---|---|---|---|
| 0 | `stage0_prepare.py` | – | Validates every H5 against `meta.csv`, freezes splits (refuses to overwrite differing ones), benchmarks each family, projects run time | `work/{ds}/splits.json`, `work/plan.json`, `logs/stage0_prepare.log` | ~10 min |
| 1 | `stage1_automl_benchmark.py` | (featureset, protocol, dataset, split) | 4 strategies × {nested (budget 1500), holdout (budget 300)} plus 11 grid-tuned baselines; feature ablations `canon` and `arniqa_proj` with `tpe_hb` only. Records the anytime curve (test score of past incumbents, computed after freezing) and, for `random`, the best configuration per family | `stage1_automl/{fs}/{protocol}/{ds}/split{k}/{strategy}.json`, `baselines.json` | 2.5 h |
| 2 | `stage2_shapley_attribution.py` | (dataset, split) | For all 63 non-empty encoder subsets, a fresh `tpe_hb` search (budget 800) restricted to that subset, nested protocol | `stage2_shapley/{ds}/split{k}.jsonl` (63 lines) | 9.5 h |
| 3 | `stage3_cross_dataset.py` | source dataset | Search on 5 grouped folds over the *whole* source set (budget 1500); the full source set is the gallery; scored on all images of the other five datasets; baselines selected the same way | `stage3_cross_dataset/{src}.json` | 5 min |
| 3b | `stage3b_transfer_variants.py` (diagnostic) | (variant, source) | Same as stage 3, with restricted spaces: `global` (global families only), `eqw` (weights fixed equal), `global_eqw` (both) | `stage3b_transfer_variants/{variant}/{src}.json` | 11 min |
| – | `stage3b_summary.py` | – | Stage 3 vs 3b by domain pair | `tables/cross_transfer.csv` | s |
| 4 | `stage4_analysis.py` | – | Headline tables, Wilcoxon + Holm, AutoML benchmark and anytime curves, family selection, encoder weights, feature ablation, raw-game Shapley values, cross-dataset matrices, per-distortion and per-reference failures | `tables/*.csv`, `report_stage4.md` | s |
| 5 | `stage5_noretune_shapley.py` | (dataset, split) | Attribution **without re-tuning**: the stage-1 `tpe_hb` configuration of the split is held fixed and, for each of the 63 subsets, the weights of the encoders outside the subset are set to 0. Inner-CV scores (5 folds) are computed before `freeze()`, test scores after; nothing is selected. Check: the full subset reproduces the stage-1 test and inner scores exactly | `stage5_noretune_shapley/{ds}/split{k}.json` | 5.5 min |
| 6 | `stage6_transfer_shrinkage.py` | source (`cross`) or (dataset, split) (`indomain`) | Shrinks the selected effective weights toward equal, ω_α = (1−α) ω + α/6 with w_e = √ω_α,e, for α ∈ {0, 0.1, …, 1}; family and hyperparameters unchanged. `cross`: the stage-3 model of each source scored on all five targets. `indomain`: the stage-1 model scored on inner folds and on test. Check: α = 0 reproduces stages 3 and 1 exactly | `stage6_transfer_shrinkage/cross/{src}.json`, `…/indomain/{ds}/split{k}.json` | 2.5 min |

**Baselines (`core.py::baseline_configs`).** Every baseline is tuned on the *same* inner folds by mean SROCC and
scored once on test:

- `ridge_{encoder}` (6) and `ridge_concat`: `global_krr_lin`, with λ ∈ {10⁻⁴, 10^−3.5, …, 10²} (13 values).
- `knn_mean_concat`: plain K-NN mean, K ∈ {5, 10, 20, 40, 80, 160}.
- `local_krr_default`: local RBF-KRR with K = 50, γ = 1, λ = 0.1, centred, equal weights. This is a sensible
  untuned retrieval model.

## Run

```bash
pip install -r requirements.txt
bash smoke_test.sh            # CSIQ + LIVEC, 2 splits, tiny budgets: every stage end to end in ./_smoke (5-10 min)
bash run_all.sh               # full experiment: stage 0 -> 1 -> 2 -> 3 -> 3b -> 4 -> 5 -> 6 -> 03_analysis   (~12 h, A100-80GB)
bash status.sh                # progress; any stage can be killed and rerun (finished jobs are skipped)
python run_jobs.py --stage 2 --gpus 0,1 --workers-per-gpu 8   # stages are 1, 2, 3, 4 (= 3b), 5, 6
```

`RUN_ON_COLAB.md` gives the procedure for running the full experiment on a Colab A100 runtime, with results
mirrored to Google Drive so that an interrupted session can be resumed.

## Output schemas

- **Stage-1 strategy file:** `job`, `strategy`, `budget`, `fold_evals_used`, `n_full_evals`, `search_seconds`,
  `selected` (configuration), `inner` (mean inner-CV SROCC), `test` (`srocc`, `krcc`, `plcc` (4-parameter
  logistic), `plcc_raw`, `rmse`, `logistic_ok`), `trajectory` (incumbents with fold-eval counts),
  `anytime` (`frac`, `inner`, test metrics), `per_family` (random search only), `n_test`, `env`.
- **Stage-1 baselines file:** `baselines.{name}` = {`inner`, `cfg`, `test`}.
- **Stage-2 line:** `subset`, `inner`, `test`, `selected`, `fold_evals` (= 800), `seconds`.
- **Stage-3 / 3b file:** `selected`, `inner`, `targets.{tgt}`, which holds metrics for `automl` and every
  baseline (stage 3) or for the single variant (stage 3b).
- **Stage-5 file:** `base_cfg` (the stage-1 configuration), `stage1_test_srocc`, `stage1_inner`,
  `check_full_test_diff`, `check_full_inner_diff` (both 0 up to rounding), `n_fold_evals` (= 315), `seconds`, and
  `records`: one entry per subset with `subset`, `cfg`, `inner`, `inner_folds`, `test`.
- **Stage-6 `cross` file:** `selected` (stage-3 configuration), `alphas`, `omega.{alpha}` (effective weights),
  `targets.{tgt}.{alpha}` (metrics), `checks.{tgt}` (α = 0 minus stage 3; 0), `seconds`.
- **Stage-6 `indomain` file:** `selected` (stage-1 configuration), `alphas`, `inner.{alpha}`, `test.{alpha}`,
  `check_alpha0_test_diff` (0), `seconds`.

Per-split test predictions (`preds.npz`) are not shipped because of their size. The failure tables derived from
them (`tables/per_distortion.csv`, `tables/per_reference.csv`) are shipped, and `stage4_analysis.py` keeps them
when no `preds.npz` is found.
