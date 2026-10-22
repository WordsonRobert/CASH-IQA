# CASH-IQA: Training-Free Blind Image Quality Assessment via Automated Fusion of Frozen Encoders

CASH-IQA predicts perceptual image quality without training a single network weight.

**How it works:**

1. Six frozen pretrained encoders, one per pretraining paradigm, are fused through an exact kernel-sum
   identity.
2. A closed-form regressor predicts the quality score. It is either a local retrieval model or global kernel
   ridge.
3. The choice of regressor family, the fusion weights and every hyperparameter is made jointly by AutoML: a
   **CASH** problem (Combined Algorithm Selection and Hyperparameter optimisation).
4. All selection happens strictly inside reference-grouped nested cross-validation.

**What this repository contains:**

- the feature-extraction code, with a fidelity check against published encoder results;
- the complete experimental pipeline;
- every raw result (per split, per search strategy, per encoder subset);
- the analysis code that produces all tables and figures.

```
                         ┌──────────── frozen encoders (one per pretraining paradigm) ─────────────┐
 image ──► views ──────► │ ARNIQA · CONTRIQUE (IQA self-sup.) │ ResNet-50 (supervised) │ DINOv2 │ CLIP │ MAE │
 (5 crops × 2 scales)    └────────────────────────────────────────────────────────────────────────┘
                                         │  per-encoder L2-normalised features (no fitted transform)
                                         ▼
           per-encoder cosine Gram matrices S_e  ──►  fused kernel  S = Σ_e w_e² S_e / Σ_e w_e²
                                         │
                                         ▼
         regressor family ∈ { Nadaraya–Watson, local KRR (lin/RBF), LWLR,  global KRR (lin/RBF) }
                                         │
         CASH search over (family, w_1..w_6, K, γ, λ, centring)  — TPE + Hyperband, inner grouped CV only
                                         ▼
                                predicted quality score
```

---

## Results

### In-domain accuracy

The protocol is nested: 10 outer 80/20 splits, each with 5 grouped inner folds. Values are median test SROCC
and PLCC. Δ is the mean paired gain over ridge regression on the concatenated features of all six encoders,
which is the standard frozen-encoder protocol. Wins count splits out of 10. The p-values come from a paired
Wilcoxon test, Holm-corrected across datasets; 0.012 is the smallest value attainable with 10 splits.

| Dataset | SROCC | PLCC | Ridge (concat) | Δ | Wins | Holm p |
|---|---|---|---|---|---|---|
| KADID-10k | **0.953** | 0.957 | 0.934 | +0.019 | 10/10 | 0.012 |
| CSIQ | **0.965** | 0.975 | 0.941 | +0.021 | 9/10 | 0.012 |
| TID2013 | **0.922** | 0.938 | 0.895 | +0.045 | 10/10 | 0.012 |
| KonIQ-10k | **0.936** | 0.947 | 0.930 | +0.005 | 10/10 | 0.012 |
| LIVE Challenge | **0.897** | 0.923 | 0.894 | +0.007 | 8/10 | 0.014 |
| SPAQ | **0.923** | 0.927 | 0.921 | +0.002 | 10/10 | 0.012 |

### Comparison with published frozen-encoder methods

Full table, with protocol, aggregation and source for every number: [`results/tables/comparison_published.csv`](results/tables/comparison_published.csv).

| Dataset | Standing | Ours | Closest published |
|---|---|---|---|
| KonIQ-10k | best | 0.936 | 0.916 (SLIQUE) |
| SPAQ | best | 0.923–0.924 | 0.921 (SLIQUE) |
| TID2013 (nested) | best | 0.922 | 0.904 (SHAMISA) |
| LIVE Challenge | tied | 0.897 | 0.897 (SLIQUE) |
| CSIQ | close | 0.965 (nested), 0.958 (70/10/20) | 0.962 (ARNIQA, 70/10/20); 0.981 (SHAMISA preprint) |
| KADID-10k | close | 0.953 | 0.957 (SLIQUE, a mean) |

The gap to per-dataset fine-tuned models is 0.005–0.022 SROCC, with zero trained parameters.

### Main findings

1. **The search space matters; the optimiser hardly does.**
   - Random search, LHS + successive halving, TPE and TPE + Hyperband end within 0.005 SROCC at an equal
     budget ([fig. 2](figures/fig2_anytime_curves.pdf)).
   - The regressor family the search selects depends on the distortion regime. On synthetic distortions it
     picks local retrieval models (24 of 30 splits); on authentic photographs it picks global kernel ridge (30
     of 30) ([fig. 3](figures/fig3_family_selection.pdf)).
2. **Exact Shapley attribution of pretraining paradigms**, computed over all 63 encoder subsets, each re-tuned.
   - On synthetic distortions, IQA self-supervision carries the fusion gain. On TID2013, ARNIQA scores +0.093;
     its own ImageNet initialisation, ResNet-50, scores −0.020.
   - On authentic photographs, CLIP carries it ([fig. 4](figures/fig4_shapley_fusion_gain.pdf)).
   - ARNIQA × CONTRIQUE is the strongest redundancy. Distortion-aware × semantic pairs are synergistic
     ([fig. 5](figures/fig5_shapley_interactions.pdf)).
3. **The attribution ranking does not need re-tuning; its magnitudes do.**
   - Scoring every subset with the selected model held fixed (inactive encoders' weights set to zero, nothing
     re-searched) costs 0.47 instead of 56.4 GPU worker-hours (×121).
   - Its Shapley ranking agrees with the exact re-tuned one (Spearman ≥ 0.9) on 5 of 6 datasets, a criterion
     fixed before the run, and the two most valuable encoders are identical on all six. The version computed on
     inner-CV scores, which uses no test data, agrees equally well.
   - Magnitudes are inflated 1.1–1.9×, most on synthetic data: weak encoders scored alone with hyperparameters
     tuned for the fused kernel do much worse than when re-tuned ([fig. 10](figures/fig10_noretune_vs_exact_shapley.pdf)).
4. **Transfer is limited by source-specific fusion weights and by hyperparameters tuned jointly with them.**
   - In cross-dataset tests, re-searching with equal weights recovers 83% of the deficit relative to ridge, and
     equal weights with global kernels match ridge (0.739 vs 0.742 mean SROCC over 30 source–target pairs)
     ([fig. 7](figures/fig7_transfer_decomposition.pdf)).
   - Shrinking an already-tuned model's weights toward equal, with the amount chosen leave-one-dataset-out,
     selects full equal weights for every target but recovers only 64% (0.709); it does not meet the
     pre-registered criterion of matching ridge within 0.010 ([fig. 11](figures/fig11_transfer_shrinkage.pdf)).
   - In-domain, shrinking costs up to 0.06 SROCC (TID2013), and inner CV itself prefers the tuned weights on
     58 of 60 splits.
   - *Practical rule:* tune the weights when the deployment domain is known; when it is not, search with equal
     weights and global kernels from the start rather than shrinking a tuned model afterwards.
5. **Failures are systematic.** Mean shift, colour saturation, block/patch substitution and contrast change
   are the hardest distortions on both KADID-10k and TID2013
   ([fig. 8](figures/fig8_per_distortion.pdf)). All of them preserve image content, which similarity-based
   prediction cannot resolve.

---

## Repository structure

```
CASH-IQA/
├── README.md
├── requirements.txt                 dependencies for 02 + 03 (01 has its own, GPU)
│
├── 01_feature_extraction/           images -> features/{dataset}/{encoder}.h5          (GPU, ~6 h on a laptop GPU)
│   ├── iqa_datasets.py              loaders and standard protocols for the six datasets
│   ├── preflight.py                 checks packages, GPU, every dataset and image, weights, each encoder end to end
│   ├── extract_features.py          the six encoders and their exact view protocols
│   ├── verify_features.py           ridge linear-probe fidelity check vs published ARNIQA / CONTRIQUE numbers
│   ├── run_extraction.sh            extraction + verification, resumable
│   └── results/                     linear_probe_fidelity.csv, extraction_manifest.json, extraction_log.txt
│
├── features/                        {dataset}/meta.csv (in git) + {dataset}/{encoder}.h5 (not in git, ~4.8 GB)
│
├── 02_automl_pipeline/              the experiment                                      (GPU, ~12 h on one A100-80GB)
│   ├── config.yaml                  every setting of the reported experiments
│   ├── core.py                      data, splits, fused-kernel bank, regressors, CASH space, search strategies, metrics
│   ├── stage0_prepare.py            feature checks, frozen splits, GPU benchmark, run-time projection
│   ├── stage1_automl_benchmark.py   4 CASH strategies × 2 protocols × 6 datasets × 10 splits + baselines + ablations
│   ├── stage2_shapley_attribution.py   all 63 encoder subsets re-tuned per split (exact Shapley game)
│   ├── stage3_cross_dataset.py      tune on one dataset, test on each of the other five
│   ├── stage3b_transfer_variants.py transfer decomposition: global-only / equal-weight / both
│   ├── stage3b_summary.py           -> results/tables/cross_transfer.csv
│   ├── stage4_analysis.py           statistics -> results/tables/*.csv + results/report_stage4.md
│   ├── stage5_noretune_shapley.py   every encoder subset scored with the selected model held fixed (no re-tuning)
│   ├── stage6_transfer_shrinkage.py tuned fusion weights shrunk toward equal: cross-dataset + in-domain
│   ├── run_jobs.py                  parallel, resumable GPU job launcher
│   ├── run_all.sh · status.sh · smoke_test.sh
│   └── RUN_ON_COLAB.md              running the full experiment on a Colab A100
│
├── 03_analysis/                     post-processing of the shipped results              (CPU, seconds)
│   ├── sanity_checks.py             completeness, budget accounting, budget adequacy, NaNs
│   ├── shapley_fusion_gain.py       Shapley values / interactions of the fusion gain, bootstrap CIs
│   ├── compare_published.py         comparison with published methods (+ published_baselines.csv with sources)
│   ├── noretune_shapley_check.py    stage 5 vs exact Shapley: agreement, magnitude, pre-registered verdict
│   ├── transfer_shrinkage.py        stage 6: leave-one-dataset-out shrinkage, in-domain cost, pre-registered verdict
│   └── make_figures.py              figures/fig1..fig11
│
├── results/
│   ├── stage1_automl/               {featureset}/{protocol}/{dataset}/split{k}/{strategy|baselines}.json   (840 files)
│   ├── stage2_shapley/              {dataset}/split{k}.jsonl, 63 re-tuned subsets per file               (60 files)
│   ├── stage3_cross_dataset/        {source}.json                                                         (6 files)
│   ├── stage3b_transfer_variants/   {variant}/{source}.json                                               (18 files)
│   ├── stage5_noretune_shapley/     {dataset}/split{k}.json, 63 subsets without re-tuning per file          (60 files)
│   ├── stage6_transfer_shrinkage/   cross/{source}.json + indomain/{dataset}/split{k}.json, 11 alphas       (66 files)
│   ├── tables/                      every table (CSV)
│   ├── work/                        frozen splits (splits.json per dataset) + plan.json (GPU benchmark)
│   ├── logs/                        stage-0 log, run logs, smoke-test log, per-job logs (tar.gz)
│   ├── report_stage4.md             auto-generated summary of the stage 1-3b tables
│   └── report_stage5.txt · report_stage6.txt   output of the two stage-5/6 analyses (with their verdicts)
│
└── figures/                         fig1..fig11 (.pdf and .png)
```

Each folder has its own README describing its contents and outputs.

---

## Reproducing

### From the shipped results

This needs a CPU only and no data. It regenerates every table and figure, and the regenerated tables are
identical to the shipped ones.

```bash
pip install -r requirements.txt
python 03_analysis/sanity_checks.py          # PROBLEMS: none
cd 02_automl_pipeline && python stage4_analysis.py && python stage3b_summary.py && cd ..
python 03_analysis/shapley_fusion_gain.py
python 03_analysis/compare_published.py
python 03_analysis/noretune_shapley_check.py
python 03_analysis/transfer_shrinkage.py
python 03_analysis/make_figures.py
```

### From scratch

This needs a GPU.

1. Obtain KADID-10k, CSIQ, TID2013, KonIQ-10k, LIVE Challenge and SPAQ. The expected layout is in
   [`01_feature_extraction/README.md`](01_feature_extraction/README.md).
2. Clone [ARNIQA](https://github.com/miccunifi/ARNIQA) and [CONTRIQUE](https://github.com/pavancm/CONTRIQUE)
   into `third_party/`, and place the CONTRIQUE checkpoint in `third_party/CONTRIQUE/models/`. The ARNIQA
   weights download automatically.
3. Run:

```bash
cd 01_feature_extraction && pip install -r requirements.txt && python preflight.py && bash run_extraction.sh
cd ../02_automl_pipeline && bash smoke_test.sh && bash run_all.sh
```

`stage0_prepare.py` refuses to overwrite the frozen splits in `results/work/`. Delete them to draw new ones.

---

## Evaluation protocol

- **Splits.** Synthetic datasets are split by reference image, so no content appears on both sides. Authentic
  datasets are split by image.
- **Nested protocol (primary).**
  - 10 outer 80/20 splits; all model selection happens in 5 grouped inner folds of the outer-train part.
  - The test part is reachable only through `Plan.final()`, which raises an error until the search has been
    frozen.
- **Holdout protocol (secondary).** 10 × 70/10/20: selection on the validation part, then a refit on train +
  validation. This matches how CONTRIQUE, ARNIQA and SLIQUE report results.
- **Search.**
  - Every strategy gets the same budget of 1,500 fold-evaluations.
  - Baselines are tuned on the same inner folds.
  - The primary strategy (TPE + Hyperband) was fixed in advance.
- **Statistics.** Medians over splits; paired Wilcoxon tests over splits, Holm-corrected across datasets; PLCC
  after a 4-parameter logistic fit.
- **Follow-up experiments (stages 5 and 6).** Their success criteria were written into the scripts before they
  were run, and both are reported as they came out: stage 5 met its criterion, stage 6 did not. Neither
  re-selects anything on test data; both reproduce the earlier stages exactly at their reference setting
  (checked by `sanity_checks.py`).

## Citation

Paper in preparation. If you use this code or these results, please cite this repository.
