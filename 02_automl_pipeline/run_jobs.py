#!/usr/bin/env python3
"""
Parallel, resumable job launcher.

    python run_jobs.py --stage 1                       # all GPUs, workers_per_gpu from config.yaml
    python run_jobs.py --stage 2 --gpus 0,1 --workers-per-gpu 8
    python run_jobs.py --stage 1 --datasets csiq livec --dry

Every job writes its own output; finished jobs are skipped, so it is safe to Ctrl-C and rerun.
Biggest datasets are scheduled first. Logs: work/logs/stage{N}/{job}.log
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import load_config
import importlib

SIZE = {"spaq": 11125, "kadid10k": 10125, "koniq10k": 10073, "tid2013": 3000, "livec": 1162, "csiq": 866, "live": 779}
MOD = {1: "stage1_automl_benchmark", 2: "stage2_shapley_attribution", 3: "stage3_cross_dataset", 4: "stage3b_transfer_variants"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True, choices=[1, 2, 3, 4])
    ap.add_argument("--config", default=None)
    ap.add_argument("--gpus", default=None, help="comma list, default: all visible")
    ap.add_argument("--workers-per-gpu", type=int, default=None)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config); mod = importlib.import_module(MOD[a.stage])
    jobs = [j for j in mod.list_jobs(cfg) if not mod.is_done(cfg, j)]
    if a.datasets:
        jobs = [j for j in jobs if j.get("ds", j.get("src")) in a.datasets]
    jobs.sort(key=lambda j: -SIZE.get(j.get("ds", j.get("src")), 0))
    if a.gpus is None:
        import torch
        n = torch.cuda.device_count(); gpus = [str(i) for i in range(n)] or ["cpu"]
    else:
        gpus = a.gpus.split(",")
    wpg = a.workers_per_gpu or cfg["runtime"]["workers_per_gpu"]
    slots = [g for g in gpus for _ in range(wpg)]
    print(f"stage {a.stage}: {len(jobs)} jobs pending | slots={len(slots)} on gpus {gpus}")
    if a.dry:
        for j in jobs[:20]: print("  ", json.dumps(j))
        return
    logdir = Path(cfg["paths"]["work"]) / "logs" / f"stage{a.stage}"; logdir.mkdir(parents=True, exist_ok=True)
    script = str(Path(__file__).with_name(MOD[a.stage] + ".py"))
    running, failed, done, t0 = {}, [], 0, time.time(); queue = list(jobs); tries = {}
    free = list(range(len(slots)))
    while queue or running:
        while queue and free:
            s = free.pop(); j = queue.pop(0); jid = "_".join(str(v) for v in j.values())
            env = dict(os.environ, OMP_NUM_THREADS=str(cfg["runtime"]["torch_threads"]),
                       MKL_NUM_THREADS=str(cfg["runtime"]["torch_threads"]))
            if slots[s] != "cpu": env["CUDA_VISIBLE_DEVICES"] = slots[s]
            if a.config: env["CASHIQA_CONFIG"] = a.config
            log = open(logdir / f"{jid}.log", "a")
            p = subprocess.Popen([sys.executable, script, "--job", json.dumps(j)] + (["--config", a.config] if a.config else []),
                                 stdout=log, stderr=subprocess.STDOUT, env=env)
            running[p] = (s, j, jid, log)
        time.sleep(2)
        for p in list(running):
            if p.poll() is None: continue
            s, j, jid, log = running.pop(p); log.close(); free.append(s)
            if p.returncode == 0 and mod.is_done(cfg, j):
                done += 1
            else:
                tries[jid] = tries.get(jid, 0) + 1
                if tries[jid] < 2: queue.append(j)
                else: failed.append(jid)
            el = time.time() - t0; rate = done / max(el, 1)
            eta = (len(jobs) - done - len(failed)) / rate / 3600 if rate > 0 else float("nan")
            print(f"  [{time.strftime('%H:%M:%S')}] done {done}/{len(jobs)} failed {len(failed)} "
                  f"running {len(running)} | elapsed {el/3600:.2f} h | ETA {eta:.2f} h", flush=True)
    print(f"finished: {done} ok, {len(failed)} failed {failed[:10]} (see {logdir})")


if __name__ == "__main__":
    main()
