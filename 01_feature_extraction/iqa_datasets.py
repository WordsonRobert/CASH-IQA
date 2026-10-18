"""
CASH-IQA -- dataset registry (shared by local extraction and server pipeline).

Every loader returns a DataFrame with one row per *distorted* image:
    key        unique relative image key (stable across encoders)
    path       absolute path on disk (local only)
    score      score exactly as released
    score_type 'mos' (higher=better) or 'dmos' (lower=better)
    q          quality, always higher=better  (q = score for MOS, q = -score for DMOS)
    ref        group id for splitting (reference image for synthetic sets, key itself for authentic)
    dist_type  distortion type (synthetic only, else '')
    dist_level distortion level (synthetic only, else -1)
    synthetic  bool

Protocol notes (deliberate, documented choices):
  * LIVE Challenge: the 7 'trainingImages' (t1..t7) listed first in AllImages_release.mat are
    EXCLUDED (standard 1162-image protocol, e.g. HyperIQA / DBCNN).
  * LIVE IQA (release 2): reference copies (orgs==1) are excluded -> 779 distorted images.
  * SPAQ: images are resized so that the shorter side is 512 px (bicubic) before any encoder
    (SPAQ paper / ARNIQA protocol). Done in the extractor, not here.
  * CSIQ / LIVE are DMOS (lower = better); q = -dmos. Correlations are reported on q, which gives
    the same |SROCC| as the raw-score convention used in the literature.
"""
import os, re
from pathlib import Path
import numpy as np
import pandas as pd

EXPECTED_N = {"kadid10k": 10125, "csiq": 866, "tid2013": 3000, "live": 779,
              "koniq10k": 10073, "livec": 1162, "spaq": 11125}
SYNTHETIC = {"kadid10k", "csiq", "tid2013", "live"}
ALL_DATASETS = ["kadid10k", "csiq", "tid2013", "koniq10k", "livec", "spaq"]   # LIVE IQA: no licence -> excluded
OPTIONAL_DATASETS = ["live"]                                                  # loader kept; pass --datasets live if access is granted
CHECK_FILES = True   # set False to parse annotations without images present


def _index_dir(d, recursive=False):
    """case-insensitive filename -> path index (skips macOS '._' junk)."""
    idx = {}
    if recursive:
        for dp, _, fns in os.walk(d):
            if "__MACOSX" in dp:
                continue
            for fn in fns:
                if not fn.startswith("._"):
                    idx.setdefault(fn.lower(), os.path.join(dp, fn))
    else:
        for e in os.scandir(d):
            if e.is_file() and not e.name.startswith("._"):
                idx.setdefault(e.name.lower(), e.path)
    return idx


def _finish(df, name):
    df = df.reset_index(drop=True)
    assert df["key"].is_unique, f"{name}: duplicate keys"
    assert np.isfinite(df["score"].values).all(), f"{name}: non-finite scores"
    df["q"] = np.where(df["score_type"] == "dmos", -df["score"].values, df["score"].values)
    df["synthetic"] = name in SYNTHETIC
    if "dist_type" not in df:
        df["dist_type"] = ""
    if "dist_level" not in df:
        df["dist_level"] = -1
    exp = EXPECTED_N[name]
    if len(df) != exp:
        raise RuntimeError(f"{name}: expected {exp} images, found {len(df)}")
    if CHECK_FILES and "path" in df:
        miss = [p for p in df["path"] if not os.path.isfile(p)]
        if miss:
            raise FileNotFoundError(f"{name}: {len(miss)} image files missing, e.g. {miss[:3]}")
    return df[["key", "path", "score", "score_type", "q", "ref", "dist_type", "dist_level", "synthetic"]]


def load_kadid10k(root):
    d = Path(root) / "kadid10k"
    t = pd.read_csv(d / "dmos.csv")
    idx = _index_dir(d / "images")
    rows = []
    for _, r in t.iterrows():
        k = r["dist_img"].strip()
        m = re.match(r"(I\d+)_(\d+)_(\d+)\.png$", k)
        rows.append(dict(key=k, path=idx.get(k.lower(), str(d / "images" / k)), score=float(r["dmos"]),
                         score_type="mos", ref=m.group(1), dist_type=f"k{int(m.group(2)):02d}",
                         dist_level=int(m.group(3))))
    return _finish(pd.DataFrame(rows), "kadid10k")   # KADID 'dmos' column is a MOS in [1,5], higher=better


def load_csiq(root):
    d = Path(root) / "csiq"
    t = pd.read_csv(d / "csiq.csv")
    TYPE = {"noise": ("awgn", "AWGN"), "blur": ("blur", "BLUR"), "jpeg": ("jpeg", "JPEG"),
            "jpeg 2000": ("jpeg2000", "jpeg2000"), "fnoise": ("fnoise", "fnoise"),
            "contrast": ("contrast", "contrast")}
    rows = []
    for _, r in t.iterrows():
        if pd.isna(r["dmos"]):
            continue
        ref = str(r["image"]).strip(); typ = str(r["dst_type"]).strip(); lev = int(r["dst_lev"])
        folder, seg = TYPE[typ]
        k = f"{folder}/{ref}.{seg}.{lev}.png"
        rows.append(dict(key=k, path=str(d / "dst_imgs" / k), score=float(r["dmos"]), score_type="dmos",
                         ref=ref, dist_type=folder, dist_level=lev))
    return _finish(pd.DataFrame(rows), "csiq")


def load_tid2013(root):
    d = Path(root) / "TID2013"
    idx = _index_dir(d / "distorted_images")
    rows = []
    with open(d / "mos_with_names.txt") as f:
        for line in f:
            p = line.split()
            if len(p) < 2:
                continue
            k = p[1].strip().lower()
            m = re.match(r"(i\d+)_(\d+)_(\d+)\.bmp$", k)
            rows.append(dict(key=k, path=idx.get(k, str(d / "distorted_images" / k)), score=float(p[0]),
                             score_type="mos", ref=m.group(1), dist_type=f"t{int(m.group(2)):02d}",
                             dist_level=int(m.group(3))))
    return _finish(pd.DataFrame(rows), "tid2013")


def load_live(root):
    import scipy.io as sio
    d = Path(root) / "live"
    need = ["dmos_realigned.mat", "refnames_all.mat", "jp2k", "jpeg", "wn", "gblur", "fastfading"]
    miss = [n for n in need if not (d / n).exists()]
    if miss:
        raise FileNotFoundError(
            f"LIVE IQA incomplete in {d}: missing {miss}. Re-extract databaserelease2.zip so that "
            f"{d}/{{jp2k,jpeg,wn,gblur,fastfading}}/imgN.bmp, dmos_realigned.mat and refnames_all.mat exist.")
    dm = sio.loadmat(d / "dmos_realigned.mat")
    dmos = dm["dmos_new"].astype(np.float64).flatten(); orgs = dm["orgs"].flatten()
    refs = [str(x[0]) for x in sio.loadmat(d / "refnames_all.mat")["refnames_all"].flatten()]
    folders = [("jp2k", 227), ("jpeg", 233), ("wn", 174), ("gblur", 174), ("fastfading", 174)]
    keys, types = [], []
    for fo, n in folders:
        for i in range(1, n + 1):
            keys.append(f"{fo}/img{i}.bmp"); types.append(fo)
    assert len(keys) == len(dmos) == len(orgs) == len(refs) == 982, "LIVE annotation length mismatch"
    rows = [dict(key=k, path=str(d / k), score=float(s), score_type="dmos", ref=os.path.splitext(r)[0],
                 dist_type=t, dist_level=-1)
            for k, s, o, r, t in zip(keys, dmos, orgs, refs, types) if o == 0]
    return _finish(pd.DataFrame(rows), "live")


def load_koniq10k(root):
    d = Path(root) / "koniq10k"
    t = pd.read_csv(d / "koniq10k.csv")
    idx = _index_dir(d / "images")
    rows = [dict(key=str(r["image_name"]).strip(), path=idx.get(str(r["image_name"]).strip().lower(), ""),
                 score=float(r["MOS"]), score_type="mos", ref=str(r["image_name"]).strip())
            for _, r in t.iterrows()]
    return _finish(pd.DataFrame(rows), "koniq10k")


def load_livec(root):
    import scipy.io as sio
    d = Path(root) / "livec"
    names = [str(x[0]) for x in sio.loadmat(d / "Data" / "AllImages_release.mat")["AllImages_release"].flatten()]
    mos = sio.loadmat(d / "Data" / "AllMOS_release.mat")["AllMOS_release"].flatten().astype(np.float64)
    assert len(names) == len(mos) == 1169
    idx = _index_dir(d / "Images", recursive=False)          # top-level Images/ only (trainingImages/ excluded)
    names, mos = names[7:], mos[7:]                         # drop t1..t7 training images
    assert not any(n.startswith("t") and n[1:].split(".")[0].isdigit() for n in names)
    rows = [dict(key=n, path=idx.get(n.lower(), ""), score=float(s), score_type="mos", ref=n)
            for n, s in zip(names, mos)]
    return _finish(pd.DataFrame(rows), "livec")


def load_spaq(root):
    d = Path(root) / "spaq"
    t = pd.read_excel(d / "MOS and Image attribute scores.xlsx", engine="openpyxl", usecols=["Image name", "MOS"])
    idx = _index_dir(d / "TestImage")
    rows = [dict(key=str(r["Image name"]).strip(), path=idx.get(str(r["Image name"]).strip().lower(), ""),
                 score=float(r["MOS"]), score_type="mos", ref=str(r["Image name"]).strip())
            for _, r in t.iterrows()]
    return _finish(pd.DataFrame(rows), "spaq")


LOADERS = {"kadid10k": load_kadid10k, "csiq": load_csiq, "tid2013": load_tid2013, "live": load_live,
           "koniq10k": load_koniq10k, "livec": load_livec, "spaq": load_spaq}


def load_dataset(name, data_root, check_files=True):
    global CHECK_FILES
    CHECK_FILES = check_files
    return LOADERS[name](data_root)
