# features/

`features/` is the output of `01_feature_extraction/` and the input of `02_automl_pipeline/`.

```
features/{dataset}/meta.csv        in git: one row per image, in the row order of every H5 file
features/{dataset}/{encoder}.h5    NOT in git (~4.8 GB in total): datasets `keys`, `main` [, `canon` | `proj`]
```

**`meta.csv` columns:**

| Column | Meaning |
|---|---|
| `key` | image key; equals the H5 `keys` row by row (checked in `core.py::load_features`) |
| `score`, `score_type` | score as released (`mos` or `dmos`) |
| `q` | regression target, higher is better (`−dmos` for CSIQ) |
| `ref` | split group: reference image (synthetic sets) or the image itself (authentic sets) |
| `dist_type`, `dist_level` | distortion type and level (synthetic sets only), used by the failure analysis |
| `synthetic` | bool |
| `width`, `height` | image size as seen by the encoders |

To get the H5 files, run `01_feature_extraction/run_extraction.sh`, or copy an existing extraction into this
folder.
