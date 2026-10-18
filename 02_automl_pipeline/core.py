"""
CASH-IQA -- shared core: data, splits, fused-kernel bank, regressors, search space, search strategies, metrics.

Key implementation facts (all exact, no approximation):
  * Every encoder block is L2-normalised, so the fused cosine similarity of the weighted
    concatenation is   S = sum_e w_e^2 S_e / sum_e w_e^2   (S_e = per-encoder cosine Gram).
    One Gram per encoder is computed once per job; any weight vector is a weighted sum.
  * LWLR is solved in its exact dual form (K x K instead of (D+1) x (D+1)).
  * The test part of a split is only reachable through Plan.final(), which refuses to run
    until the search has been frozen. Search code never sees test indices.
"""
import os, json, time, math, hashlib, itertools, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, kendalltau, pearsonr, rankdata
from scipy.optimize import curve_fit

FAMILIES_LOCAL = ("nw", "local_krr_lin", "local_krr_rbf", "lwlr")
FAMILIES_GLOBAL = ("global_krr_lin", "global_krr_rbf")


# =============================================================== config / io
def load_config(path=None):
    path = path or os.environ.get("CASHIQA_CONFIG", str(Path(__file__).with_name("config.yaml")))
    try:
        import yaml
        with open(path) as f:
            cfg = yaml.safe_load(f)
    except ImportError:
        raise SystemExit("pip install pyyaml")
    base = Path(path).resolve().parent                       # relative paths are relative to the config file
    for k, v in cfg["paths"].items():
        p = Path(os.path.expanduser(str(v)))
        cfg["paths"][k] = str(p if p.is_absolute() else (base / p).resolve())
    for k in ("work", "results"):
        Path(cfg["paths"][k]).mkdir(parents=True, exist_ok=True)
    cfg["_path"] = str(Path(path).resolve())
    return cfg


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def atomic_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + f".tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=_json_default)
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, torch.Tensor): return o.tolist()
    return str(o)


def job_seed(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


# =============================================================== data
def load_meta(cfg, ds):
    return pd.read_csv(Path(cfg["paths"]["features"]) / ds / "meta.csv")


def load_features(cfg, ds, encoders, featureset="main"):
    """returns dict enc -> float32 np array [N, D] (rows L2-normalised), verified against meta order."""
    import h5py
    meta = load_meta(cfg, ds)
    fsel = cfg["featuresets"][featureset] or {}
    out = {}
    for e in encoders:
        var = fsel.get(e, "main")
        with h5py.File(Path(cfg["paths"]["features"]) / ds / f"{e}.h5", "r") as f:
            keys = [k.decode() for k in f["keys"][:]]
            if keys != meta["key"].tolist():
                raise RuntimeError(f"{ds}/{e}: key order differs from meta.csv")
            X = f[var][:].astype(np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12      # per-sample, no fitted transform
        out[e] = X
    return meta, out


# =============================================================== splits
def _group_shuffle(groups, rng):
    u = np.array(sorted(set(groups))); rng.shuffle(u); return u


def make_splits(meta, cfg):
    """All splits for one dataset. Groups = reference image (synthetic) or image (authentic)."""
    P = cfg["protocol"]; g = meta["ref"].astype(str).values; N = len(g)
    seed = job_seed(P["seed"], "splits", len(g))
    out = {"nested": [], "holdout": [], "cv_full": None, "groups": int(len(set(g)))}
    for k in range(P["n_outer"]):
        rng = np.random.RandomState(seed + k)
        u = _group_shuffle(g, rng); nte = max(1, int(round(P["outer_test_frac"] * len(u))))
        te_g = set(u[:nte]); tr_g = u[nte:]
        fold_of_g = {gg: i % P["n_inner_folds"] for i, gg in enumerate(tr_g)}   # groups dealt round-robin after shuffle
        tr = np.array([i for i in range(N) if g[i] not in te_g]); te = np.array([i for i in range(N) if g[i] in te_g])
        folds = np.array([fold_of_g[g[i]] for i in tr])
        out["nested"].append({"train": tr.tolist(), "test": te.tolist(), "inner_fold": folds.tolist()})
    a, b, _ = P["holdout_fracs"]
    for k in range(P["n_holdout"]):
        rng = np.random.RandomState(seed + 1000 + k)
        u = _group_shuffle(g, rng); n1 = int(round(a * len(u))); n2 = int(round((a + b) * len(u)))
        s = {x: 0 for x in u[:n1]}; s.update({x: 1 for x in u[n1:n2]}); s.update({x: 2 for x in u[n2:]})
        lab = np.array([s[x] for x in g])
        out["holdout"].append({"train": np.where(lab == 0)[0].tolist(), "val": np.where(lab == 1)[0].tolist(),
                               "test": np.where(lab == 2)[0].tolist()})
    rng = np.random.RandomState(seed + 5000); u = _group_shuffle(g, rng)
    fo = {gg: i % P["n_inner_folds"] for i, gg in enumerate(u)}
    out["cv_full"] = [fo[x] for x in g]
    check_splits(out, g)
    return out


def check_splits(sp, g):
    g = np.asarray(g)
    for s in sp["nested"]:
        tr, te, fo = np.array(s["train"]), np.array(s["test"]), np.array(s["inner_fold"])
        assert len(set(tr) & set(te)) == 0 and len(tr) + len(te) == len(g)
        assert not (set(g[tr]) & set(g[te])), "group leak outer"
        for f in np.unique(fo):
            assert not (set(g[tr[fo == f]]) & set(g[tr[fo != f]])), "group leak inner"
    for s in sp["holdout"]:
        parts = [set(g[np.array(s[k])]) for k in ("train", "val", "test")]
        assert not (parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2]), "group leak holdout"
    fo = np.array(sp["cv_full"])
    for f in np.unique(fo):
        assert not (set(g[fo == f]) & set(g[fo != f])), "group leak cv_full"


def load_splits(cfg, ds):
    return json.loads((Path(cfg["paths"]["work"]) / ds / "splits.json").read_text())


# =============================================================== metrics
def _logistic4(x, b1, b2, b3, b4):
    return b2 + (b1 - b2) / (1 + np.exp(-(x - b3) / (np.abs(b4) + 1e-12)))


def metrics(y, p):
    y = np.asarray(y, np.float64); p = np.asarray(p, np.float64)
    if not np.isfinite(p).all() or np.std(p) < 1e-12:
        return {"srocc": 0.0, "krcc": 0.0, "plcc": 0.0, "plcc_raw": 0.0, "rmse": float(np.std(y)), "logistic_ok": False}
    s = spearmanr(y, p)[0]; k = kendalltau(y, p)[0]; pr = pearsonr(y, p)[0]
    ok = True
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            beta, _ = curve_fit(_logistic4, p, y, p0=[y.max(), y.min(), np.mean(p), np.std(p) / 4 + 1e-6], maxfev=20000)
        yl = _logistic4(p, *beta); pl = pearsonr(y, yl)[0]; rmse = float(np.sqrt(np.mean((y - yl) ** 2)))
        if not np.isfinite(pl) or pl < pr - 1e-6:          # VQEG fit must not be worse than linear
            raise RuntimeError
    except Exception:
        ok = False; pl = pr
        a, b = np.polyfit(p, y, 1); rmse = float(np.sqrt(np.mean((y - (a * p + b)) ** 2)))
    return {"srocc": float(s), "krcc": float(k), "plcc": float(pl), "plcc_raw": float(pr), "rmse": rmse, "logistic_ok": ok}


def _rank_t(x):
    """ordinal ranks on device (predictions practically never tie; y ranks are exact, see Fold)."""
    r = torch.empty_like(x); r[torch.argsort(x)] = torch.arange(len(x), device=x.device, dtype=x.dtype); return r


def srocc_t(ry, p):
    if not torch.isfinite(p).all() or p.std() < 1e-12:
        return 0.0
    rp = _rank_t(p.double()); ry = ry.double()
    rp = rp - rp.mean(); ryc = ry - ry.mean()
    return float((rp * ryc).sum() / torch.sqrt((rp * rp).sum() * (ryc * ryc).sum()))


# =============================================================== kernel bank + regressors
class Bank:
    """Per-encoder cosine Grams G[e] (N x N) of one dataset, on device."""

    def __init__(self, feats, encoders, dev):
        self.enc = list(encoders); self.dev = dev
        Xs = [torch.from_numpy(feats[e]).to(dev) for e in self.enc]
        self.G = torch.stack([X @ X.T for X in Xs])            # [E, N, N] float32
        self.N = self.G.shape[1]
        del Xs

    def wvec(self, cfg):
        w = torch.tensor([float(cfg["w"].get(e, 0.0)) for e in self.enc], device=self.dev, dtype=torch.float64) ** 2
        if float(w.sum()) <= 0:
            w = torch.ones_like(w)
        return (w / w.sum()).float()

    def fused(self, w, rows, cols):
        """sum_e w_e G[e][rows][:, cols]  without materialising all E blocks at once."""
        out = None
        for e, we in enumerate(w.tolist()):              # one host sync, not one per encoder
            if we == 0.0:
                continue
            blk = self.G[e].index_select(0, rows).index_select(1, cols) * we
            out = blk if out is None else out.add_(blk)
        return out


def predict(bank, cfg, q_sim, g_idx, y_g, chunk_elems=2e8):
    """
    bank   : Bank (gallery x gallery similarities come from bank.G via g_idx)
    cfg    : dict(family, w{enc}, K, gamma, lam, center)
    q_sim  : [nq, ng] fused query-gallery cosine similarity (float32, device)
    g_idx  : [ng] long tensor, gallery indices into bank.G
    y_g    : [ng] float64 tensor, gallery targets
    returns: [nq] float64 predictions
    """
    fam = cfg["family"]; gam = float(cfg.get("gamma", 1.0)); lam = float(cfg.get("lam", 1e-2))
    w = bank.wvec(cfg); nq, ng = q_sim.shape
    if fam in FAMILIES_GLOBAL:
        Kgg = bank.fused(w, g_idx, g_idx).double()
        Kq = q_sim.double()
        if fam == "global_krr_rbf":
            Kgg = torch.exp(-gam * (2 - 2 * Kgg).clamp_min(0)); Kq = torch.exp(-gam * (2 - 2 * Kq).clamp_min(0))
        else:
            Kgg = Kgg + 1.0; Kq = Kq + 1.0                         # linear kernel + bias feature
        ym = y_g.mean()
        Kgg.diagonal().add_(lam)
        L, info = torch.linalg.cholesky_ex(Kgg)
        if int(info) != 0:
            alpha = torch.linalg.lstsq(Kgg, (y_g - ym)[:, None]).solution[:, 0]
        else:
            alpha = torch.cholesky_solve((y_g - ym)[:, None], L)[:, 0]
        return Kq @ alpha + ym

    k = int(min(max(int(cfg["K"]), 2), ng))
    cq, nb = torch.topk(q_sim, k, dim=1)                          # [nq,k] similarities and gallery positions
    cq = cq.double(); yn = y_g[nb]                                # [nq,k]
    d = (2 - 2 * cq).clamp_min(0)
    if fam == "nw":
        logw = -gam * d; logw = logw - logw.max(1, keepdim=True).values      # stable softmax weights
        ww = torch.exp(logw); return (ww * yn).sum(1) / ww.sum(1)
    out = torch.empty(nq, dtype=torch.float64, device=q_sim.device)
    wl = w.tolist(); n_act = max(1, sum(x > 0 for x in wl))
    step = max(1, int(chunk_elems // (k * k * n_act)))
    gnb = g_idx[nb]                                               # global indices of neighbours
    eye = torch.eye(k, dtype=torch.float64, device=q_sim.device)
    for s in range(0, nq, step):
        e_ = min(nq, s + step); gi = gnb[s:e_]; c = e_ - s
        G = None
        for ei, we in enumerate(wl):
            if we == 0.0: continue
            Ge = bank.G[ei][gi[:, :, None], gi[:, None, :]] * we
            G = Ge if G is None else G.add_(Ge)
        G = G.double(); y = yn[s:e_]; cqs = cq[s:e_]
        if fam in ("local_krr_lin", "local_krr_rbf"):
            if fam == "local_krr_rbf":
                Km = torch.exp(-gam * (2 - 2 * G).clamp_min(0)); kq = torch.exp(-gam * d[s:e_])
            else:
                Km = G; kq = cqs
            mu = y.mean(1, keepdim=True) if cfg.get("center", False) else torch.zeros_like(y[:, :1])
            v = torch.linalg.solve(Km + lam * eye, (y - mu)[..., None])[..., 0]
            out[s:e_] = (kq * v).sum(1) + mu[:, 0]
        elif fam == "lwlr":                                       # exact dual of weighted ridge on [x, 1]
            r = torch.sqrt(torch.exp(-gam * d[s:e_]))
            A = r[:, :, None] * (G + 1.0) * r[:, None, :]
            v = torch.linalg.solve(A + lam * eye, (r * y)[..., None])[..., 0]
            out[s:e_] = ((cqs + 1.0) * r * v).sum(1)
        else:
            raise KeyError(fam)
    return out


# =============================================================== plans (leak-proof access to data)
class Plan:
    """
    search_folds : list of (query_idx, gallery_idx)  -- the ONLY data the search may touch
    final        : (test_idx, gallery_idx)            -- reachable only after freeze()
    """

    def __init__(self, bank, y, search_folds, final, sim_cache=True):
        self.bank, self.dev = bank, bank.dev
        self.y = torch.from_numpy(np.array(y, dtype=np.float64)).to(bank.dev)
        self.folds = [(torch.as_tensor(q, device=bank.dev), torch.as_tensor(g, device=bank.dev)) for q, g in search_folds]
        self.ry = [torch.from_numpy(rankdata(np.asarray(y)[np.asarray(q)])).to(bank.dev) for q, _ in search_folds]
        self._final = final; self.frozen = False
        self.n_fold_evals = 0; self.t_eval = 0.0
        self.chunk = 2e8

    def eval_fold(self, cfg, f):
        if self.frozen:
            raise RuntimeError("search is frozen")
        t = time.perf_counter()
        q, g = self.folds[f]
        sim = self.bank.fused(self.bank.wvec(cfg), q, g)
        p = predict(self.bank, cfg, sim, g, self.y[g], self.chunk)
        s = srocc_t(self.ry[f], p)
        self.n_fold_evals += 1; self.t_eval += time.perf_counter() - t
        return s if math.isfinite(s) else 0.0

    def freeze(self):
        self.frozen = True

    def final(self, cfg):
        assert self.frozen, "Plan.final() called before the search was frozen"
        te, g = self._final
        te_t, g_t = torch.as_tensor(te, device=self.dev), torch.as_tensor(g, device=self.dev)
        sim = self.bank.fused(self.bank.wvec(cfg), te_t, g_t)
        p = predict(self.bank, cfg, sim, g_t, self.y[g_t], self.chunk).cpu().numpy()
        return p, metrics(self.y[te_t].cpu().numpy(), p)


def nested_plan(bank, y, split, n_folds):
    tr = np.array(split["train"]); fo = np.array(split["inner_fold"])
    folds = [(tr[fo == f], tr[fo != f]) for f in range(n_folds)]
    return Plan(bank, y, folds, (np.array(split["test"]), tr))


def holdout_plan(bank, y, split, refit):
    tr, va, te = (np.array(split[k]) for k in ("train", "val", "test"))
    gal = np.concatenate([tr, va]) if refit else tr
    return Plan(bank, y, [(va, tr)], (te, gal))


# =============================================================== search space
class Space:
    def __init__(self, cfg, encoders, families=None):
        S = cfg["search_space"]; self.enc = list(encoders)
        self.fam = list(families or S["families"])
        self.wb = S["w"]; self.Kb = S["K"]; self.gb = S["log10_gamma"]; self.lb = S["log10_lambda"]
        self.dim = 1 + (len(self.enc) if len(self.enc) > 1 else 0) + 4

    def from_unit(self, u):
        u = list(u); i = 0
        fam = self.fam[min(int(u[i] * len(self.fam)), len(self.fam) - 1)]; i += 1
        if len(self.enc) > 1:
            w = {e: self.wb[0] + u[i + j] * (self.wb[1] - self.wb[0]) for j, e in enumerate(self.enc)}; i += len(self.enc)
        else:
            w = {self.enc[0]: 1.0}
        K = int(round(math.exp(math.log(self.Kb[0]) + u[i] * (math.log(self.Kb[1]) - math.log(self.Kb[0]))))); i += 1
        gam = 10 ** (self.gb[0] + u[i] * (self.gb[1] - self.gb[0])); i += 1
        lam = 10 ** (self.lb[0] + u[i] * (self.lb[1] - self.lb[0])); i += 1
        center = u[i] >= 0.5
        return self.clean({"family": fam, "w": w, "K": K, "gamma": gam, "lam": lam, "center": center})

    def clean(self, c):
        """drop parameters the family ignores (so identical models get identical configs)."""
        f = c["family"]; c = dict(c)
        if f in FAMILIES_GLOBAL: c.pop("K", None); c.pop("center", None)
        if f in ("nw", "lwlr"): c.pop("center", None)
        if f in ("local_krr_lin", "global_krr_lin"): c.pop("gamma", None)
        if f == "nw": c.pop("lam", None)
        return c

    def optuna(self, trial):
        fam = trial.suggest_categorical("family", self.fam)
        w = {e: trial.suggest_float(f"w_{e}", *self.wb) for e in self.enc} if len(self.enc) > 1 else {self.enc[0]: 1.0}
        c = {"family": fam, "w": w}
        if fam in FAMILIES_LOCAL:
            c["K"] = trial.suggest_int("K", self.Kb[0], self.Kb[1], log=True)
        if fam not in ("local_krr_lin", "global_krr_lin"):
            c["gamma"] = 10 ** trial.suggest_float("log10_gamma", *self.gb)
        if fam != "nw":
            c["lam"] = 10 ** trial.suggest_float("log10_lambda", *self.lb)
        if fam in ("local_krr_lin", "local_krr_rbf"):
            c["center"] = trial.suggest_categorical("center", [False, True])
        return self.clean(c)


def lhs(n, d, rng):
    u = np.zeros((n, d))
    for j in range(d):
        u[:, j] = (rng.permutation(n) + rng.uniform(size=n)) / n
    return u


# =============================================================== search strategies
class Ledger:
    """budget accounting + incumbent trajectory (full-fidelity scores only)."""

    def __init__(self, plan, budget):
        self.plan, self.budget = plan, budget
        self.best = (-np.inf, None); self.traj = []; self.n_full = 0; self.t0 = time.time()
        self.cache = {}                                           # (cfg_key, fold) -> score

    @staticmethod
    def key(c):
        return json.dumps(c, sort_keys=True, default=float)

    def left(self):
        return self.budget - self.plan.n_fold_evals

    def fold(self, c, f):
        k = (self.key(c), f)
        if k not in self.cache:
            self.cache[k] = self.plan.eval_fold(c, f)
        return self.cache[k]

    def full(self, c):
        """mean over all search folds; returns None if the budget cannot cover it."""
        nf = len(self.plan.folds); k = self.key(c)
        need = sum(1 for f in range(nf) if (k, f) not in self.cache)
        if need > self.left():
            return None
        s = float(np.mean([self.fold(c, f) for f in range(nf)]))
        self.done(c, s)
        return s

    def done(self, c, s):
        self.n_full += 1
        if s > self.best[0]:
            self.best = (s, c)
            self.traj.append({"fold_evals": self.plan.n_fold_evals, "seconds": round(time.time() - self.t0, 3),
                              "n_full": self.n_full, "inner": s, "cfg": c})


def run_random(space, ledger, rng):
    while ledger.left() >= len(ledger.plan.folds):
        ledger.full(space.from_unit(rng.uniform(size=space.dim)))


def run_lhs_sh(space, ledger, rng, eta=3):
    """Latin-hypercube design followed by successive halving; fidelity = number of inner folds."""
    nf = len(ledger.plan.folds)
    rungs = [1, 2, nf] if nf >= 3 else [nf]
    unit_cost = rungs[0] + sum((rungs[i] - rungs[i - 1]) / eta ** i for i in range(1, len(rungs)))
    n0 = max(eta ** (len(rungs) - 1), int(ledger.budget / unit_cost))
    pop = [space.from_unit(u) for u in lhs(n0, space.dim, rng)]
    order = rng.permutation(nf)
    for ri, r in enumerate(rungs):
        sc = []
        for c in pop:
            k = Ledger.key(c); need = sum(1 for f in order[:r] if (k, int(f)) not in ledger.cache)
            if need > ledger.left():
                break
            s = float(np.mean([ledger.fold(c, int(f)) for f in order[:r]]))
            if r == nf:
                ledger.done(c, s)
            sc.append(s)
        pop = pop[:len(sc)]
        if ri == len(rungs) - 1 or not pop:
            break
        keep = max(1, len(pop) // eta)
        pop = [pop[i] for i in np.argsort(sc)[::-1][:keep]]


def run_optuna(space, ledger, seed, hyperband):
    import optuna
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    nf = len(ledger.plan.folds)
    sampler = optuna.samplers.TPESampler(seed=seed % (2 ** 31), multivariate=True, group=True,
                                         n_startup_trials=20, constant_liar=False)
    pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource=nf, reduction_factor=3) if (hyperband and nf > 1) \
        else optuna.pruners.NopPruner()
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    order = list(np.random.RandomState(seed).permutation(nf))
    while ledger.left() >= 1:
        tr = study.ask(); c = space.optuna(tr)
        vals, pruned = [], False
        for step, f in enumerate(order):
            if ledger.left() < 1 and (Ledger.key(c), int(f)) not in ledger.cache:
                pruned = True; break
            vals.append(ledger.fold(c, int(f)))
            if hyperband and step < nf - 1:
                tr.report(float(np.mean(vals)), step + 1)
                if tr.should_prune():
                    pruned = True; break
        if pruned:
            study.tell(tr, state=optuna.trial.TrialState.PRUNED)
            if ledger.left() < 1: break
            continue
        s = float(np.mean(vals)); ledger.done(c, s); study.tell(tr, s)


def search(strategy, space, plan, budget, seed):
    led = Ledger(plan, budget); rng = np.random.RandomState(seed)
    t = time.time()
    if strategy == "random": run_random(space, led, rng)
    elif strategy == "lhs_sh": run_lhs_sh(space, led, rng)
    elif strategy == "tpe": run_optuna(space, led, seed, hyperband=False)
    elif strategy == "tpe_hb": run_optuna(space, led, seed, hyperband=True)
    else: raise KeyError(strategy)
    return led, time.time() - t


def per_family_best(led, families):
    """best full-fidelity config per family among everything the search evaluated (selection by inner CV)."""
    nf = len(led.plan.folds); best = {}
    full = {}
    for (k, f), s in led.cache.items():
        full.setdefault(k, {})[f] = s
    for k, d in full.items():
        if len(d) < nf: continue
        c = json.loads(k); s = float(np.mean(list(d.values())))
        if c["family"] not in best or s > best[c["family"]][0]:
            best[c["family"]] = (s, c)
    return best


# =============================================================== fixed baselines (grid + inner CV)
def baseline_configs(encoders):
    lam = [10 ** x for x in np.arange(-4, 2.01, 0.5)]
    B = {}
    for e in encoders:
        B[f"ridge_{e}"] = [{"family": "global_krr_lin", "w": {x: float(x == e) for x in encoders}, "lam": l} for l in lam]
    B["ridge_concat"] = [{"family": "global_krr_lin", "w": {x: 1.0 for x in encoders}, "lam": l} for l in lam]
    B["knn_mean_concat"] = [{"family": "nw", "w": {x: 1.0 for x in encoders}, "K": k, "gamma": 0.0}
                            for k in (5, 10, 20, 40, 80, 160)]
    B["local_krr_default"] = [{"family": "local_krr_rbf", "w": {x: 1.0 for x in encoders}, "K": 50, "gamma": 1.0, "lam": 0.1,
                         "center": True}]
    return B


def run_baselines(plan, encoders):
    """each baseline: pick its grid point by inner-CV mean SROCC, evaluate test once (after freeze)."""
    nf = len(plan.folds); chosen = {}
    for name, grid in baseline_configs(encoders).items():
        sc = [float(np.mean([plan.eval_fold(c, f) for f in range(nf)])) for c in grid]
        j = int(np.argmax(sc)); chosen[name] = (sc[j], grid[j])
    return chosen


def env_info():
    return {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "host": os.uname().nodename, "pid": os.getpid()}
