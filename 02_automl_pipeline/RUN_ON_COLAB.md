# Running the pipeline on Google Colab (A100)

The experiments were run on a single Colab A100-80GB runtime, with features and results on Google Drive:

| Stage | Wall-clock |
|---|---|
| Stage 1 | 2.48 h |
| Stage 2 (budget 800) | 9.48 h, in a fresh runtime |
| Stage 3 | 5 min |
| Stage 3b | 11 min |

Colab sessions end without warning, so the run is built around three properties:

1. **Every job writes its own output** (atomic JSON; stage 2 appends one JSONL line per subset), and finished
   jobs are skipped.
2. **Outputs are mirrored to Drive** while the run is going.
3. **`%%bash` cells buffer their output until the cell ends**, so the run is started in the background and
   watched from a small Python monitor cell.

Drive layout assumed below:

```
MyDrive/CASH-IQA/features.tar          (features/ with all {ds}/{enc}.h5 + meta.csv, ~4.8 GB)
MyDrive/CASH-IQA/out/                  (results mirror, written by cell 4)
```

### Cell 1: runtime check, Drive, code

```python
import subprocess, sys
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True)
assert gpu.returncode == 0, "No GPU: Runtime > Change runtime type > A100"
print(gpu.stdout)
from google.colab import drive; drive.mount("/content/drive")
!git clone https://github.com/<user>/CASH-IQA /content/CASH-IQA
!pip -q install -r /content/CASH-IQA/requirements.txt
```

### Cell 2: features to local disk, then restore earlier results (safe on a fresh run)

```bash
%%bash
set -e
cd /content/CASH-IQA
tar -xf /content/drive/MyDrive/CASH-IQA/features.tar -C .        # local SSD is much faster than Drive for H5 reads
mkdir -p /content/drive/MyDrive/CASH-IQA/out
rsync -a /content/drive/MyDrive/CASH-IQA/out/ results/           # resume: finished jobs will be skipped
# repair a stage-2 JSONL whose last line was cut by a disconnect (the loader also tolerates it)
python - <<'EOF'
import json, glob
for f in glob.glob("results/stage2_shapley/*/*.jsonl"):
    good = []
    for l in open(f):
        try: json.loads(l); good.append(l if l.endswith("\n") else l + "\n")
        except Exception: pass
    open(f, "w").writelines(good)
EOF
bash 02_automl_pipeline/status.sh || true
```

### Cell 3: start a stage in the background

```python
import subprocess
STAGE = "1"        # "1", then "2", "3", "4" (= 3b); stage 0 runs once first:  !cd /content/CASH-IQA/02_automl_pipeline && python stage0_prepare.py
cmd = f"cd /content/CASH-IQA/02_automl_pipeline && nohup python run_jobs.py --stage {STAGE} > /content/run_stage{STAGE}.log 2>&1 &"
subprocess.run(cmd, shell=True)
```

To run stages 2 and 3 back to back in one runtime:

```
cd 02_automl_pipeline && (python run_jobs.py --stage 2 && python run_jobs.py --stage 3) > /content/run_s23.log 2>&1 &
```

### Cell 4: monitor and Drive backup (re-run it any time; it refreshes itself)

```python
import subprocess, time
from IPython.display import clear_output
LOG = "/content/run_stage1.log"
while True:
    subprocess.run("rsync -a --exclude 'work/logs' /content/CASH-IQA/results/ /content/drive/MyDrive/CASH-IQA/out/", shell=True)
    st = subprocess.run("bash /content/CASH-IQA/02_automl_pipeline/status.sh", shell=True, capture_output=True, text=True).stdout
    tail = subprocess.run(f"tail -n 8 {LOG}", shell=True, capture_output=True, text=True).stdout
    clear_output(wait=True); print(time.strftime("%H:%M:%S")); print(st); print(tail)
    if "RUN: not running" in st: print("finished (or stopped) -- check the log"); break
    time.sleep(120)
```

### Cell 5: after the last stage

```bash
%%bash
cd /content/CASH-IQA/02_automl_pipeline
python stage3b_summary.py && python stage4_analysis.py
cd .. && python 03_analysis/sanity_checks.py && python 03_analysis/shapley_fusion_gain.py && python 03_analysis/make_figures.py
rsync -a results/ figures /content/drive/MyDrive/CASH-IQA/out/
```

**Notes:**

- A restarted runtime can come up CPU-only. Cell 1 stops in that case; switch the runtime type back to A100.
- Stage 3 and 3b need very little GPU time. A GPU is still recommended, because RBF-KRR over about 11k source
  images is slow on a CPU.
- Stage 2 is the long pole: 63 subsets × 10 splits × 6 datasets = 3,780 searches.
  - The budget was set to 800 fold-evaluations so that it fits a single 12 h session.
  - `03_analysis/sanity_checks.py` (check 3) verifies that budget 800 matches the stage-1 budget of 1500 on the
    all-encoder subset, to within 0.004 median SROCC.
