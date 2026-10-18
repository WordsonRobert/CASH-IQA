#!/usr/bin/env python3
"""
CASH-IQA -- local feature extraction
=======================================
One H5 per (dataset, encoder) holding features for ALL images of the dataset.
There is NO train/test split at extraction time and NO fitted transform (no PCA):
every split / fold is decided later on the server, so nothing here can leak.

Encoder "basis" (one encoder per pre-training paradigm):
    arniqa      IQA self-supervised, synthetic degradation manifold  (ResNet-50, WACV'24)
    contrique   IQA self-supervised, contrastive synthetic+authentic (ResNet-50, TIP'22)
    resnet50    supervised ImageNet-1k classification (IMAGENET1K_V1 -- the exact init of ARNIQA)
    dinov2      self-distillation                                    (ViT-L/14)
    clip        language-supervised contrastive                      (ViT-L/14, OpenAI)
    mae         masked reconstruction                                (ViT-L/16, MAE pre-train)

View protocols (fixed a priori, never tuned on test data):
    arniqa      official ARNIQA test protocol: 5 crops (centre + 4 corners, 224) at full scale and at
                half scale (bicubic /2); backbone feature f (2048, L2-normalised) averaged over crops;
                [full | half] -> 4096.  ('proj' = ARNIQA's 128-d projector embedding per scale, [full|half] = 256-d; kept for the ablation)
    contrique   official CONTRIQUE protocol: whole image at full and half scale, no ImageNet
                normalisation; L2-normalised backbone h (2048) per scale -> 4096.
    resnet50, dinov2, clip, mae
                'main'  = identical views to ARNIQA (5 crops x 2 scales at native resolution), so the
                          encoder is the only thing that changes. ViTs: [CLS | mean patch] per view
                          (CLIP: projected image embedding), L2-normalised, averaged over crops.
                'canon' = the encoder's own canonical single view (resize + centre crop 224), kept for
                          the "native resolution vs canonical view" ablation.
    SPAQ        every encoder sees the image resized so the shorter side is 512 px (bicubic).

Output layout:
    {out}/{dataset}/meta.csv                 key, score, score_type, q, ref, dist_type, dist_level, width, height
    {out}/{dataset}/{encoder}.h5             datasets: keys, main [, canon | proj]; attrs: provenance
    {out}/manifest.json                      versions, checkpoints (sha256), timings

Usage:
    python extract_features.py                                  # everything
    python extract_features.py --datasets csiq livec --encoders arniqa contrique
    python verify_features.py                                   # sanity: linear probes vs published numbers
"""
import argparse, os, sys, json, time, hashlib, platform, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
import torchvision.transforms.functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parent))
from iqa_datasets import load_dataset, ALL_DATASETS, OPTIONAL_DATASETS

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[1]                      # repository root
ALL_ENCODERS = ["arniqa", "contrique", "resnet50", "dinov2", "clip", "mae"]
IMNET_MEAN, IMNET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
CLIP_MEAN, CLIP_STD = [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]


# ----------------------------------------------------------------------------- utils
def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def load_rgb(path, dataset):
    img = Image.open(path).convert("RGB")
    if dataset == "spaq":                              # shorter side -> 512, bicubic (ARNIQA/SPAQ protocol)
        w, h = img.size
        if w < h:
            nw, nh = 512, int(512 / (w / h))
        else:
            nh, nw = 512, int(512 * (w / h))
        img = img.resize((nw, nh), Image.BICUBIC)
    return img


def center_corners_crop(img, s=224):
    """Exact ARNIQA crop (TF.crop zero-pads when the image is smaller than the crop)."""
    w, h = img.size
    cx, cy = w // 2, h // 2
    return [TF.crop(img, cy - s // 2, cx - s // 2, s, s), TF.crop(img, 0, 0, s, s),
            TF.crop(img, h - s, 0, s, s), TF.crop(img, 0, w - s, s, s), TF.crop(img, h - s, w - s, s, s)]


def half(img):
    w, h = img.size
    return img.resize((w // 2, h // 2), Image.BICUBIC)


class CropViews(torch.utils.data.Dataset):
    """10 views (5 crops x {full, half}) + optional canonical view."""
    def __init__(self, paths, dataset, mean, std, canon_tf=None):
        self.paths, self.ds, self.canon_tf = paths, dataset, canon_tf
        self.norm = T.Normalize(mean, std)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = load_rgb(self.paths[i], self.ds)
        v = [TF.to_tensor(c) for c in center_corners_crop(img)] + [TF.to_tensor(c) for c in center_corners_crop(half(img))]
        out = {"i": i, "views": self.norm(torch.stack(v)), "size": torch.tensor(img.size)}
        if self.canon_tf is not None:
            out["canon"] = self.canon_tf(img)
        return out


class FullViews(torch.utils.data.Dataset):
    """CONTRIQUE: whole image at full and half scale, ToTensor only."""
    def __init__(self, paths, dataset):
        self.paths, self.ds = paths, dataset

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = load_rgb(self.paths[i], self.ds)
        return {"i": i, "full": TF.to_tensor(img), "half": TF.to_tensor(half(img)), "size": torch.tensor(img.size)}


def l2(x):
    return F.normalize(x.float(), dim=-1)


# ----------------------------------------------------------------------------- encoders
class Encoder:
    name = ""; views = "crops"; mean = IMNET_MEAN; std = IMNET_STD; amp = True
    canon_tf = None
    info = {}

    def per_view(self, x):            # x: [V,3,224,224] -> dict(name -> [V,D]) already L2-normalised
        raise NotImplementedError


class ARNIQAEnc(Encoder):
    name = "arniqa"

    def __init__(self, dev, arniqa_dir):
        sys.path.insert(0, str(arniqa_dir))
        from models.resnet import ResNet
        self.m = ResNet(embedding_dim=128, pretrained=False, use_norm=True)
        w = Path(arniqa_dir) / "weights" / "ARNIQA.pth"
        if not w.exists():
            w.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file("https://github.com/miccunifi/ARNIQA/releases/download/weights/ARNIQA.pth", str(w))
        self.m.load_state_dict(torch.load(w, map_location="cpu"))
        self.m = self.m.to(dev).eval()
        self.info = {"weights": str(w), "sha256": sha256(w)}

    def per_view(self, x):
        f, g = self.m(x)                          # f: 2048 L2-normalised backbone, g: 128 L2-normalised projector
        return {"main": l2(f), "proj": l2(g)}


class ResNet50Enc(Encoder):
    name = "resnet50"

    def __init__(self, dev):
        import torchvision
        m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        self.m = nn.Sequential(*list(m.children())[:-1]).to(dev).eval()
        self.info = {"weights": "torchvision ResNet50_Weights.IMAGENET1K_V1"}

    def per_view(self, x):
        return {"main": l2(self.m(x).flatten(1))}


class DINOv2Enc(Encoder):
    name = "dinov2"

    def __init__(self, dev):
        self.m = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14").to(dev).eval()
        self.canon_tf = T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224),
                                   T.ToTensor(), T.Normalize(IMNET_MEAN, IMNET_STD)])
        self.info = {"weights": "torch.hub facebookresearch/dinov2 dinov2_vitl14"}

    def tokens(self, x):
        o = self.m.forward_features(x)
        return o["x_norm_clstoken"], o["x_norm_patchtokens"].mean(1)

    def per_view(self, x):
        c, p = self.tokens(x)
        return {"main": torch.cat([l2(c), l2(p)], -1)}


class MAEEnc(DINOv2Enc):
    name = "mae"

    def __init__(self, dev):
        import timm
        self.m = timm.create_model("vit_large_patch16_224.mae", pretrained=True, num_classes=0).to(dev).eval()
        self.canon_tf = T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224),
                                   T.ToTensor(), T.Normalize(IMNET_MEAN, IMNET_STD)])
        self.info = {"weights": "timm vit_large_patch16_224.mae", "timm": timm.__version__}

    def tokens(self, x):
        t = self.m.forward_features(x)            # [B, 1+196, 1024] after final norm
        return t[:, 0], t[:, 1:].mean(1)


class CLIPEnc(Encoder):
    name = "clip"; mean = CLIP_MEAN; std = CLIP_STD

    def __init__(self, dev):
        try:
            import clip
            self.m, _ = clip.load("ViT-L/14", device=dev)
            src = "openai/clip ViT-L/14"
        except ImportError:
            import open_clip
            self.m, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai", device=dev)
            src = "open_clip ViT-L-14 (openai weights)"
        self.m = self.m.float().eval()
        self.canon_tf = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224),
                                   T.ToTensor(), T.Normalize(CLIP_MEAN, CLIP_STD)])   # == CLIP preprocess
        self.info = {"weights": src}

    def per_view(self, x):
        return {"main": l2(self.m.encode_image(x))}


class CONTRIQUEEnc(Encoder):
    name = "contrique"; views = "full"; amp = False

    def __init__(self, dev, contrique_dir):
        sys.path.insert(0, str(contrique_dir)); sys.path.insert(0, str(Path(contrique_dir) / "modules"))
        from modules.CONTRIQUE_model import CONTRIQUE_model
        from modules.network import get_network
        ck = Path(contrique_dir) / "models" / "CONTRIQUE_checkpoint25.tar"
        enc = get_network("resnet50", pretrained=False)
        self.m = CONTRIQUE_model(argparse.Namespace(device=torch.device(dev)), enc, 2048)
        st = torch.load(ck, map_location="cpu")
        self.m.load_state_dict({k.replace("module.", ""): v for k, v in st.items()}, strict=True)
        self.m = self.m.to(dev).eval()
        self.info = {"weights": str(ck), "sha256": sha256(ck)}

    def full(self, a, b):
        o = self.m(a, b)
        return {"main": torch.cat([l2(o[4]), l2(o[5])], -1)}


PROTO = {
    "arniqa": "ARNIQA official: 5 crops(224) x {full, half(bicubic)}; backbone f(2048, L2) mean over crops; [full|half]. proj = 128-d projector (ablation).",
    "contrique": "CONTRIQUE official: whole image at full and half scale, ToTensor only; L2(h) 2048 per scale; [full|half].",
    "resnet50": "ImageNet-1k V1 ResNet-50 avgpool(2048, L2); ARNIQA views (5 crops x 2 scales); [full|half].",
    "dinov2": "DINOv2 ViT-L/14 [L2(CLS)|L2(mean patch)] per view; main = ARNIQA views [full|half]; canon = resize256+cc224.",
    "clip": "CLIP ViT-L/14 projected embedding (L2) per view; main = ARNIQA views [full|half]; canon = CLIP preprocess.",
    "mae": "MAE ViT-L/16 [L2(CLS)|L2(mean patch)] per view; main = ARNIQA views [full|half]; canon = resize256+cc224.",
}


def build(name, dev, a):
    if name == "arniqa": return ARNIQAEnc(dev, a.arniqa_dir)
    if name == "contrique": return CONTRIQUEEnc(dev, a.contrique_dir)
    if name == "resnet50": return ResNet50Enc(dev)
    if name == "dinov2": return DINOv2Enc(dev)
    if name == "clip": return CLIPEnc(dev)
    if name == "mae": return MAEEnc(dev)
    raise KeyError(name)


# ----------------------------------------------------------------------------- extraction loops
@torch.no_grad()
def run_crops(enc, paths, dataset, dev, batch, workers):
    ds = CropViews(paths, dataset, enc.mean, enc.std, enc.canon_tf)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, num_workers=workers, pin_memory=(dev == "cuda"))
    out, sizes = {}, np.zeros((len(paths), 2), np.int32)
    for b in dl:
        idx = b["i"].numpy(); B = len(idx)
        v = b["views"].to(dev, non_blocking=True).flatten(0, 1)         # [B*10,3,224,224]
        with torch.autocast("cuda", dtype=torch.float16, enabled=(dev == "cuda" and enc.amp)):
            f = enc.per_view(v)
            fc = enc.per_view(b["canon"].to(dev)) if "canon" in b else None
        for k, x in f.items():
            x = x.float().view(B, 2, 5, -1)                              # [B, scale, crop, D]
            x = torch.cat([l2(x[:, 0].mean(1)), l2(x[:, 1].mean(1))], -1) # mean over crops, per-scale L2
            out.setdefault(k, np.zeros((len(paths), x.shape[1]), np.float32))[idx] = x.cpu().numpy()
        if fc is not None:
            x = fc["main"].float()
            out.setdefault("canon", np.zeros((len(paths), x.shape[1]), np.float32))[idx] = l2(x).cpu().numpy()
        sizes[idx] = b["size"].numpy()
    return out, sizes


@torch.no_grad()
def run_full(enc, paths, dataset, dev, workers):
    dl = torch.utils.data.DataLoader(FullViews(paths, dataset), batch_size=1, num_workers=workers)
    out, sizes = None, np.zeros((len(paths), 2), np.int32)
    for b in dl:
        i = int(b["i"][0])
        x = enc.full(b["full"].to(dev), b["half"].to(dev))["main"].float()
        if out is None:
            out = np.zeros((len(paths), x.shape[1]), np.float32)
        out[i] = x[0].cpu().numpy(); sizes[i] = b["size"][0].numpy()
    return {"main": out}, sizes


def save_h5(path, keys, feats, attrs):
    tmp = str(path) + ".tmp"
    with h5py.File(tmp, "w") as f:
        f.create_dataset("keys", data=np.array([k.encode() for k in keys]))
        for k, v in feats.items():
            assert np.isfinite(v).all(), f"non-finite features in {k}"
            nrm = np.linalg.norm(v, axis=1)
            assert (nrm > 0.5).all(), f"zero feature rows in {k} (an image failed?)"
            f.create_dataset(k, data=v, compression="lzf")
        for k, v in attrs.items():
            f.attrs[k] = json.dumps(v, default=str)   # always JSON text (portable)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=os.environ.get("IQA_DATA_ROOT", str(REPO / "data")))
    ap.add_argument("--out", default=str(REPO / "features"))
    ap.add_argument("--arniqa_dir", default=str(REPO / "third_party" / "ARNIQA"))
    ap.add_argument("--contrique_dir", default=str(REPO / "third_party" / "CONTRIQUE"))
    ap.add_argument("--datasets", nargs="+", default=ALL_DATASETS, choices=ALL_DATASETS + OPTIONAL_DATASETS)
    ap.add_argument("--encoders", nargs="+", default=ALL_ENCODERS, choices=ALL_ENCODERS)
    ap.add_argument("--batch", type=int, default=16, help="images per batch (x10 views)")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 2))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true", help="run even if no GPU is visible (very slow)")
    ap.add_argument("--limit", type=int, default=0, help="debug only: first N images (writes nothing to the real out dir unless you point --out there)")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu" and not a.allow_cpu:
        raise SystemExit("No CUDA GPU visible to PyTorch -- refusing to extract on CPU (it would take days).\n"
                         "Check: python -c \"import torch; print(torch.cuda.is_available(), torch.version.cuda)\"  and  nvidia-smi\n"
                         "(pass --allow-cpu to override)")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    print(f"device={dev} {torch.cuda.get_device_name(0) if dev == 'cuda' else ''} | out={out}")

    man_p = out / "manifest.json"
    man = json.loads(man_p.read_text()) if man_p.exists() else {"runs": []}
    man["env"] = {"python": platform.python_version(), "torch": torch.__version__,
                  "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu", "host": platform.node()}

    metas = {}
    for ds in a.datasets:                                   # parse + validate annotations first (fail fast)
        try:
            metas[ds] = load_dataset(ds, a.data_root)
            if a.limit:
                metas[ds] = metas[ds].iloc[:a.limit].reset_index(drop=True)
            print(f"  [ok] {ds}: {len(metas[ds])} images, {metas[ds].ref.nunique()} groups")
        except Exception as e:
            print(f"  [SKIP] {ds}: {e}")
    for ename in a.encoders:
        todo = [ds for ds in metas if a.force or not (out / ds / f"{ename}.h5").exists()]
        if not todo:
            print(f"[{ename}] all done"); continue
        enc = build(ename, dev, a)
        for ds in todo:
            meta = metas[ds]; (out / ds).mkdir(exist_ok=True)
            t0 = time.time(); bs = a.batch
            while True:
                try:
                    if enc.views == "full":
                        feats, sizes = run_full(enc, meta["path"].tolist(), ds, dev, a.workers)
                    else:
                        feats, sizes = run_crops(enc, meta["path"].tolist(), ds, dev, bs, a.workers)
                    break
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache(); bs = max(1, bs // 2); print(f"    OOM -> batch {bs}")
            dt = time.time() - t0
            attrs = {"encoder": ename, "dataset": ds, "n": len(meta), "dims": {k: int(v.shape[1]) for k, v in feats.items()},
                     "protocol": PROTO[ename], "info": enc.info, "seconds": round(dt, 1),
                     "torch": torch.__version__}
            save_h5(out / ds / f"{ename}.h5", meta["key"].tolist(), feats, attrs)
            mp = out / ds / "meta.csv"
            if not mp.exists():
                m = meta.drop(columns=["path"]).copy(); m["width"], m["height"] = sizes[:, 0], sizes[:, 1]
                m.to_csv(mp, index=False)
            man["runs"].append({"dataset": ds, "encoder": ename, "dims": attrs["dims"], "seconds": attrs["seconds"],
                                "info": enc.info, "time": time.strftime("%Y-%m-%d %H:%M:%S")})
            man_p.write_text(json.dumps(man, indent=2))
            print(f"  [{ename}] {ds}: {attrs['dims']} in {dt/60:.1f} min")
        del enc; torch.cuda.empty_cache() if dev == "cuda" else None
    print("done. next: python verify_features.py")


if __name__ == "__main__":
    main()
