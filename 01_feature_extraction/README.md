# 01 · Feature extraction (images → frozen-encoder features)

This step turns every image of the six datasets into one feature vector per encoder. Nothing is learned here.

- **No splits at extraction time.** Features are computed for all images of a dataset at once.
- **No fitted transform.** There is no PCA and no whitening; the only operation is a per-row L2 normalisation.

Every split and fold is therefore decided later, in `02_automl_pipeline/`, and this step cannot leak test
information.

## Encoders: one per pre-training paradigm

| Key | Model | Paradigm | Representation stored as `main` | Dim |
|---|---|---|---|---|
| `arniqa` | ARNIQA (WACV'24), ResNet-50 | IQA self-supervised (synthetic degradation manifold) | backbone `f` (2048, L2) mean over 5 crops, for [full \| half] scale | 4096 |
| `contrique` | CONTRIQUE (TIP'22), ResNet-50 | IQA self-supervised (contrastive, synthetic + authentic) | backbone `h` (2048, L2) of the **whole** image, for [full \| half] scale | 4096 |
| `resnet50` | torchvision ResNet-50 IMAGENET1K_V1 | supervised classification; **ARNIQA's exact initialisation** | avgpool (2048, L2), ARNIQA views | 4096 |
| `dinov2` | DINOv2 ViT-L/14 | self-distillation | [L2(CLS) \| L2(mean patch)] per view, ARNIQA views | 4096 |
| `clip` | OpenAI CLIP ViT-L/14 | language-supervised contrastive | projected image embedding (768, L2), ARNIQA views | 1536 |
| `mae` | MAE ViT-L/16 (timm) | masked reconstruction | [L2(CLS) \| L2(mean patch)] per view, ARNIQA views | 4096 |

**ARNIQA views.** These are the official ARNIQA test protocol:

- 5 crops of 224 × 224 (centre and four corners), taken at full resolution and at half resolution (bicubic).
- Features are averaged over the crops, L2-normalised per scale, then concatenated as [full | half].
- Images smaller than 224 px are zero-padded, exactly as `TF.crop` does in ARNIQA.

Every non-IQA encoder sees exactly these views. Between the IQA encoders and the generic encoders, the network
is therefore the only thing that changes.

**Extra variants (stored in the same H5, used by the ablation in stage 1):**

- `canon`, for `dinov2`, `clip` and `mae`: the encoder's canonical single view. That is resize to 256 and
  centre-crop 224 (CLIP: resize 224 and centre-crop 224), with the model's own normalisation.
- `proj`, for `arniqa`: the 128-d projector output per scale (256-d over [full | half]).

**Dataset-specific protocol:**

- SPAQ images are resized so the shorter side is 512 px (bicubic) before any encoder sees them (SPAQ / ARNIQA
  protocol).
- CONTRIQUE uses no ImageNet normalisation, as in the official code.
- Crops run under fp16 autocast; CONTRIQUE's full-image views run in fp32.

## Files

| File | Role |
|---|---|
| `iqa_datasets.py` | One loader per dataset, returning `key, path, score, score_type, q, ref, dist_type, dist_level, synthetic`. `q` is always higher-is-better (`q = −DMOS` for CSIQ). `ref` is the split group: the reference image for synthetic sets and the image itself for authentic sets. |
| `preflight.py` | Checks packages, the GPU, every annotation file and image, the weights, and runs each encoder end to end on one image. Run this first. |
| `extract_features.py` | The six encoders and their view protocols; writes `features/{ds}/{enc}.h5` (`keys`, `main`, optionally `canon` / `proj`) and `features/{ds}/meta.csv`. Resumable. |
| `verify_features.py` | Fidelity check: ridge linear probe (λ ∈ logspace(−3, 3, 13) selected on validation) over 10 × 70/10/20 reference-disjoint splits; compares with published ARNIQA / CONTRIQUE numbers. |
| `run_extraction.sh` | Extraction followed by verification, logged. |
| `results/linear_probe_fidelity.csv` | Output of `verify_features.py` for every (dataset, encoder, variant). |
| `results/extraction_manifest.json` | Per (dataset, encoder): dims, seconds, weight file sha256, library versions. |
| `results/extraction_log.txt` | Full extraction log (paths anonymised). |

## Fidelity check (median test SROCC, ridge probe, 10 × 70/10/20)

| Dataset | ARNIQA ours / published | CONTRIQUE ours / published |
|---|---|---|
| KADID-10k | 0.920 / 0.908 | 0.932 / 0.934 |
| CSIQ | 0.954 / 0.962 | 0.921 / 0.942 |
| TID2013 | 0.881 / 0.880 | **0.735** / 0.843 |
| KonIQ-10k | 0.867 / – | 0.882 / 0.894 |
| LIVE Challenge | 0.752 / – | 0.815 / 0.845 |
| SPAQ | 0.897 / 0.905 | 0.897 / 0.914 |

- **ARNIQA** reproduces its paper to within 0.012 on all four datasets that paper reports.
- **CONTRIQUE** reproduces within 0.03 everywhere except TID2013, where this probe gives 0.735 against the
  published 0.843. Inside the main pipeline, the tuned CONTRIQUE ridge baseline reaches 0.804 (holdout) and
  0.866 (nested).

For this reason, the primary comparisons are made against ridge on the same features, under identical splits.
Published numbers are reported as secondary context.

## Setup

```
data/                                   ($IQA_DATA_ROOT, default <repo>/data)
├── kadid10k/   dmos.csv, images/
├── csiq/       csiq.csv, dst_imgs/
├── TID2013/    mos_with_names.txt, distorted_images/
├── koniq10k/   koniq10k.csv, images/ (1024x768 release)
├── livec/      Data/AllImages_release.mat, Data/AllMOS_release.mat, Images/
└── spaq/       "MOS and Image attribute scores.xlsx", TestImage/
third_party/
├── ARNIQA/     git clone https://github.com/miccunifi/ARNIQA      (weights download automatically)
└── CONTRIQUE/  git clone https://github.com/pavancm/CONTRIQUE     + models/CONTRIQUE_checkpoint25.tar
```

```bash
pip install -r requirements.txt          # + CLIP: pip install git+https://github.com/openai/CLIP.git
python preflight.py                      # must end with "ALL PASS"
bash run_extraction.sh                   # about 6 h on a laptop RTX 4050 for 6 datasets x 6 encoders; resumable
```

The full feature set is about 4.8 GB of H5 files. They are not tracked by git; only `features/{ds}/meta.csv` is.
Dataset sizes: KADID-10k 10,125 · CSIQ 866 · TID2013 3,000 · KonIQ-10k 10,073 · LIVE Challenge 1,162 (the 7
training images t1–t7 excluded) · SPAQ 11,125. LIVE IQA has a loader but is excluded, because no licence was
available.
