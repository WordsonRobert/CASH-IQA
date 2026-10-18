#!/usr/bin/env python3
"""
Preflight check -- run BEFORE the long extraction (takes 1-5 min, mostly model downloads).

  * python packages, GPU, free disk
  * every dataset's annotations parse and every image file exists (counts must match the standard protocol)
  * ARNIQA / CONTRIQUE local weights present
  * each of the 6 encoders builds (downloads DINOv2 / CLIP / MAE / ResNet-50 weights once) and runs
    one real image through its exact extraction path

Prints a PASS/FAIL table. Fix every FAIL before running run_extraction.sh.
"""
import os, sys, shutil, time, traceback, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

ok_all = True


def report(name, ok, msg=""):
    global ok_all
    ok_all &= ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<28} {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    REPO = Path(__file__).resolve().parents[1]                      # repository root
    ap.add_argument("--data_root", default=os.environ.get("IQA_DATA_ROOT", str(REPO / "data")))
    ap.add_argument("--out", default=str(REPO / "features"))
    ap.add_argument("--arniqa_dir", default=str(REPO / "third_party" / "ARNIQA"))
    ap.add_argument("--contrique_dir", default=str(REPO / "third_party" / "CONTRIQUE"))
    a = ap.parse_args()

    print("== packages")
    for mod in ["torch", "torchvision", "timm", "h5py", "numpy", "pandas", "scipy", "PIL", "openpyxl"]:
        try:
            m = __import__(mod); report(mod, True, getattr(m, "__version__", ""))
        except Exception as e:
            report(mod, False, f"pip install {mod if mod != 'PIL' else 'pillow'}")
    try:
        import clip; report("clip (openai)", True)
    except Exception:
        try:
            import open_clip; report("clip (open_clip)", True, open_clip.__version__)
        except Exception:
            report("clip", False, "pip install git+https://github.com/openai/CLIP.git   (or: pip install open_clip_torch)")

    print("== hardware")
    import torch
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        report("GPU", True, f"{p.name} {p.total_memory/1e9:.1f} GB")
        if p.total_memory < 7e9:
            print("        (small GPU: use --batch 4 if you see OOM; the script also halves batch automatically)")
    else:
        report("GPU", False, "no CUDA GPU visible -- extraction on CPU would take days")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(a.out).free / 1e9
    report("disk free at --out", free > 8, f"{free:.0f} GB (need ~5 GB)")

    print("== datasets")
    from iqa_datasets import load_dataset, ALL_DATASETS
    metas = {}
    for ds in ALL_DATASETS:
        try:
            m = load_dataset(ds, a.data_root); metas[ds] = m
            report(ds, True, f"{len(m)} images, {m.ref.nunique()} groups")
        except Exception as e:
            report(ds, False, str(e)[:160])

    print("== local weights")
    w = Path(a.arniqa_dir) / "weights" / "ARNIQA.pth"
    report("ARNIQA.pth", w.exists(), str(w) if w.exists() else f"missing {w} (it will be downloaded automatically)")
    c = Path(a.contrique_dir) / "models" / "CONTRIQUE_checkpoint25.tar"
    report("CONTRIQUE_checkpoint25.tar", c.exists(), str(c))

    print("== encoders (build + one real image through the exact extraction path)")
    import extract_features as E
    if not metas:
        report("encoders", False, "no dataset available to test with"); return
    ds = "csiq" if "csiq" in metas else next(iter(metas)); paths = metas[ds]["path"].tolist()[:2]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for name in E.ALL_ENCODERS:
        t = time.time()
        try:
            enc = E.build(name, dev, a)
            if enc.views == "full":
                f, _ = E.run_full(enc, paths, ds, dev, 0)
            else:
                f, _ = E.run_crops(enc, paths, ds, dev, 2, 0)
            import numpy as np
            good = all(np.isfinite(v).all() and v.shape[0] == 2 for v in f.values())
            report(name, good, " ".join(f"{k}:{v.shape[1]}d" for k, v in f.items()) + f"  ({time.time()-t:.0f}s)")
            del enc
            if dev == "cuda": torch.cuda.empty_cache()
        except Exception as e:
            report(name, False, f"{type(e).__name__}: {str(e)[:150]}")
            traceback.print_exc(limit=2)

    print("\nALL PASS -> run:  bash run_extraction.sh" if ok_all else "\nFix the FAIL lines above, then rerun preflight.py")


if __name__ == "__main__":
    main()
