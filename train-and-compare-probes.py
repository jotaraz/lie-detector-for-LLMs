#!/usr/bin/env python3
"""Train and compare linear deception probes across models, layers and datasets.

Full specification: train-and-compare-probes-spec.md (this file implements it).

Usage (one invocation per probed model):
    python train-and-compare-probes.py train --model meta-llama/Llama-3.3-70B-Instruct \
        [--layers 0-59] [--device cuda] [--balance balanced] [--reg-coeff 10.0] \
        [--split-frac 0.8] [--backend torch] [--root autocomp2] \
        [--out results/probe-comparison] [--force] [--no-validate]

    python train-and-compare-probes.py plot [--out results/probe-comparison]

Probes: L2-regularized logistic regression (sklearn convention C = 1/reg_coeff,
no intercept) on StandardScaler-normalized per-token activations of the model
response. Positive class = deceptive. 7 probes per (model, layer): one per full
dataset, the three pairs, and the triple ("all3"). Each probe is evaluated on the
three full datasets' test splits and on three control datasets (true stories,
fake stories, evasive company assistant), where fiction on request is expected to
score as honest.
"""

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

# ----------------------------------------------------------------------------
# Configuration constants (see spec §2, §3, §6)
# ----------------------------------------------------------------------------

JUDGE = "gpt-5.4-mini"
SEED = 42                 # token-pool permutations (fresh RNG per pool)
VALIDATION_SEED = 1234    # row subsample for the torch-vs-sklearn check
VALIDATION_CAP = 20000
FPR_TARGETS = (0.05, 0.10)

KNOWN_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
]

# Full datasets: name -> list of (data dir, {class: accepted judge labels}).
# prompted_roleplay draws honest and deceptive conversations from two dirs.
FULL_DATASETS = {
    "unprompted_roleplay": [
        ("data-more-roleplay-explicit",
         {"honest": {"HONEST"}, "deceptive": {"DECEPTIVE"}}),
    ],
    "prompted_roleplay": [
        ("data-ir-honest-explicit", {"honest": {"HONEST"}}),
        ("data-ir-dishonest-explicit", {"deceptive": {"DECEPTIVE"}}),
    ],
    "company_assistant": [
        ("data-company-assistant-roleplay",
         {"honest": {"honest_authorized"}, "deceptive": {"deceptive"}}),
    ],
}
FULL_ORDER = list(FULL_DATASETS)

# Controls: name -> (data dir, accepted labels or None for "all conversations").
CONTROL_DATASETS = {
    "true_stories": ("data-true-stories", None),
    "fake_stories": ("data-fake-stories", None),
    "evasive_company_assistant": ("data-company-assistant-roleplay", {"evasive"}),
}

PROBE_CONFIGS = [
    ("unprompted_roleplay", ["unprompted_roleplay"]),
    ("prompted_roleplay", ["prompted_roleplay"]),
    ("company_assistant", ["company_assistant"]),
    ("unprompted+prompted", ["unprompted_roleplay", "prompted_roleplay"]),
    ("unprompted+company", ["unprompted_roleplay", "company_assistant"]),
    ("prompted+company", ["prompted_roleplay", "company_assistant"]),
    ("all3", list(FULL_ORDER)),
]
PROBE_TRAINSETS = dict(PROBE_CONFIGS)

CLS_CODE = {"honest": 0, "deceptive": 1, "control": 2}

# Defaults resolve relative to this script's directory (not the CWD), so the
# script can be invoked from anywhere; --root/--out override.
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = str(SCRIPT_DIR / "autocomp2")
DEFAULT_OUT = str(SCRIPT_DIR / "results" / "probe-comparison")

# ----------------------------------------------------------------------------
# Scored-file selection (spec §3): judge gpt-5.4-mini, timestamped only,
# latest timestamp wins.
# ----------------------------------------------------------------------------

SCORED_RE = re.compile(
    r"^scored-(?:[A-Za-z0-9]+-)?(?P<ts>\d{8}-\d{6})-(?P<model>.+)\.json$")


def _judge_re(model_fname: str) -> re.Pattern:
    return re.compile(
        rf"^{re.escape(model_fname)}-judge-{re.escape(JUDGE)}"
        rf"(?:-[0-9a-f]{{8}})?-(?P<ts>\d{{8}}-\d{{6}})\.json$")


def find_scored_file(dir_path: Path, dirname: str, model_fname: str):
    """Return (path, {conversation k: label}) for the selected scored file."""
    candidates = []  # (ts, path)
    if dirname.startswith("data-more-roleplay"):
        for f in os.listdir(dir_path):
            m = SCORED_RE.match(f)
            if m and m.group("model") == model_fname:
                candidates.append((m.group("ts"), dir_path / f))
    else:
        pat = _judge_re(model_fname)
        for f in os.listdir(dir_path):
            m = pat.match(f)
            if m:
                candidates.append((m.group("ts"), dir_path / f))
    chosen, chosen_ts = None, None
    for ts, path in sorted(candidates):
        with open(path) as fh:
            d = json.load(fh)
        if d.get("judge_model") != JUDGE:
            continue
        if chosen_ts is None or ts > chosen_ts:
            chosen, chosen_ts, chosen_data = path, ts, d
    if chosen is None:
        raise FileNotFoundError(
            f"no timestamped scored file with judge_model={JUDGE!r} for "
            f"{model_fname} in {dir_path}")
    records = chosen_data.get("verdicts") or chosen_data.get("scores")
    if records is None:
        raise ValueError(f"{chosen}: neither 'verdicts' nor 'scores' present")
    id_key = "index" if "verdicts" in chosen_data else "k"
    label_by_k = {r[id_key]: r.get("label") for r in records}
    return chosen, label_by_k


# ----------------------------------------------------------------------------
# Per-directory data: metadata, labels, mmap'd activations (spec §4, §11)
# ----------------------------------------------------------------------------

@dataclass
class DirData:
    name: str
    path: Path
    meta: dict
    k_list: list
    N: int
    m: int
    layers: list
    pt_path: Path
    scored_path: Path | None = None
    label_by_k: dict | None = None
    _acts: torch.Tensor | None = field(default=None, repr=False)
    canonical_valid: torch.Tensor | None = field(default=None, repr=False)

    def acts(self) -> torch.Tensor:
        if self._acts is None:
            obj = torch.load(self.pt_path, map_location="cpu",
                             weights_only=True, mmap=True)
            t = obj["autocompletion_activations"]
            # N and m align tokens with conversations and labels — a mismatch
            # there means the json and pt describe different data: hard fail.
            if t.shape[0] != self.N or t.shape[2] != self.m:
                raise ValueError(
                    f"{self.pt_path}: activation shape {tuple(t.shape)} does "
                    f"not match metadata (N={self.N}, m={self.m})")
            # The layer axis may be larger than the (stale) json says, e.g.
            # activations regenerated with all layers without updating the
            # json. The json's `layers` list is our only map from absolute
            # layer number to tensor index, so on mismatch we assume the pt
            # holds the contiguous first layers 0..L-1 (the regeneration
            # scheme). If a regenerated pt ever holds a non-contiguous layer
            # subset, its json MUST be correct — hence the loud warning.
            if t.shape[3] != len(self.layers):
                assumed = list(range(t.shape[3]))
                print(f"  WARNING: {self.name}: pt has {t.shape[3]} layers but "
                      f"json says {self.layers} — assuming layers "
                      f"{assumed[0]}..{assumed[-1]} (stale json?)")
                self.layers = assumed
            self._acts = t
        return self._acts

    def layer_slice(self, layer: int, device: str) -> torch.Tensor:
        """Materialize layer slice as flat (N*m, d) fp32 on `device`.

        Also maintains/checks the canonical token-validity mask (all-zero rows
        are padding; physically identical across layers)."""
        pos = self.layers.index(layer)
        x = self.acts()[:, 0, :, pos, :].to(device=device, dtype=torch.float32)
        valid = (x != 0).any(dim=2).cpu()
        if self.canonical_valid is None:
            self.canonical_valid = valid
        elif not torch.equal(valid, self.canonical_valid):
            n_diff = int((valid != self.canonical_valid).sum())
            print(f"  WARNING: {self.name} layer {layer}: validity mask differs "
                  f"from canonical at {n_diff} positions; using canonical")
        return x.reshape(-1, x.shape[-1])


def load_dir(root: Path, dirname: str, model_fname: str,
             need_labels: bool) -> DirData:
    dir_path = root / dirname
    meta_path = dir_path / f"{model_fname}.json"
    pt_path = dir_path / f"{model_fname}.pt"
    for p in (meta_path, pt_path):
        if not p.exists():
            raise FileNotFoundError(f"missing required file: {p}")
    with open(meta_path) as fh:
        meta = json.load(fh)
    convs = meta["conversations"]
    dd = DirData(
        name=dirname, path=dir_path, meta=meta,
        k_list=[c["k"] for c in convs], N=len(convs), m=meta["m"],
        layers=list(meta["layers"]), pt_path=pt_path)
    if need_labels:
        dd.scored_path, dd.label_by_k = find_scored_file(
            dir_path, dirname, model_fname)
        if len(dd.label_by_k) < dd.N:
            raise ValueError(
                f"{dd.scored_path}: only {len(dd.label_by_k)} records for "
                f"{dd.N} conversations — selected scored file does not cover "
                f"the dataset (spec §3 guard)")
    return dd


# ----------------------------------------------------------------------------
# Splits and token pools (spec §5, §6)
# ----------------------------------------------------------------------------

@dataclass
class ClassSplit:
    dirname: str
    train: list  # conversation positions (indices into the dir's conv list)
    test: list


def build_splits(dirs: dict, split_frac: float) -> dict:
    """splits[dataset][class] = ClassSplit. Deterministic first-N% per class."""
    splits = {}
    for ds, sources in FULL_DATASETS.items():
        splits[ds] = {}
        for dirname, clsmap in sources:
            dd = dirs[dirname]
            for cls, labels in clsmap.items():
                sel = [i for i, k in enumerate(dd.k_list)
                       if dd.label_by_k.get(k) in labels]
                if len(sel) < 2:
                    raise ValueError(
                        f"{ds}/{cls}: only {len(sel)} conversations with labels "
                        f"{labels} in {dirname}")
                n_train = int(len(sel) * split_frac)
                n_train = max(1, min(n_train, len(sel) - 1))
                splits[ds][cls] = ClassSplit(
                    dirname=dirname, train=sel[:n_train], test=sel[n_train:])
    return splits


@dataclass
class EvalSet:
    name: str
    kind: str                      # "full" | "control"
    chunks: list                   # [(dirname, LongTensor flat indices)]
    counts: list                   # tokens per conversation, in order
    meta: list                     # [(k, dirname, cls, label)] in order


def _conv_token_idx(dd: DirData, conv_pos: int) -> torch.Tensor:
    valid = dd.canonical_valid[conv_pos]
    return (torch.nonzero(valid, as_tuple=False).squeeze(1)
            + conv_pos * dd.m).to(torch.long)


def _pack_chunks(entries):
    """entries: [(dirname, idx, k, cls, label)] -> EvalSet pieces.
    Groups contiguous same-dir runs into single index tensors."""
    chunks, counts, meta = [], [], []
    cur_dir, cur_idx = None, []
    for dirname, idx, k, cls, label in entries:
        if dirname != cur_dir:
            if cur_idx:
                chunks.append((cur_dir, torch.cat(cur_idx)))
            cur_dir, cur_idx = dirname, []
        cur_idx.append(idx)
        counts.append(len(idx))
        meta.append((k, dirname, cls, label))
    if cur_idx:
        chunks.append((cur_dir, torch.cat(cur_idx)))
    return chunks, counts, meta


def build_token_structs(dirs: dict, splits: dict):
    """Build train pools (permuted, layer-independent) and eval sets.

    Requires canonical_valid on every dir (i.e., one layer_slice() call done).
    Returns (train_pools, eval_sets):
      train_pools[(ds, cls)] = (dirname, LongTensor of flat indices, permuted)
      eval_sets[name] = EvalSet
    """
    train_pools, eval_sets = {}, {}
    n_skipped = 0

    for ds in FULL_ORDER:
        for cls in ("honest", "deceptive"):
            sp = splits[ds][cls]
            dd = dirs[sp.dirname]
            parts = [_conv_token_idx(dd, i) for i in sp.train]
            parts = [p for p in parts if len(p) > 0]
            if not parts:
                raise ValueError(f"{ds}/{cls}: no valid train tokens")
            pool = torch.cat(parts)
            # Fresh RNG per pool, seed 42: selection independent of execution
            # order and identical across layers (spec §6).
            perm = np.random.default_rng(SEED).permutation(len(pool))
            train_pools[(ds, cls)] = (
                sp.dirname, pool[torch.as_tensor(perm, dtype=torch.long)])

        entries = []
        for cls in ("honest", "deceptive"):
            sp = splits[ds][cls]
            dd = dirs[sp.dirname]
            for i in sp.test:
                idx = _conv_token_idx(dd, i)
                if len(idx) == 0:
                    n_skipped += 1
                    continue
                entries.append((sp.dirname, idx, dd.k_list[i], cls, cls))
        chunks, counts, meta = _pack_chunks(entries)
        eval_sets[ds] = EvalSet(ds, "full", chunks, counts, meta)

    for name, (dirname, labels) in CONTROL_DATASETS.items():
        dd = dirs[dirname]
        entries = []
        for i, k in enumerate(dd.k_list):
            label = dd.label_by_k.get(k) if dd.label_by_k else "unscored"
            if labels is not None and label not in labels:
                continue
            idx = _conv_token_idx(dd, i)
            if len(idx) == 0:
                n_skipped += 1
                continue
            entries.append((dirname, idx, k, "control", label))
        if not entries:
            raise ValueError(f"control {name}: no conversations selected")
        chunks, counts, meta = _pack_chunks(entries)
        eval_sets[name] = EvalSet(name, "control", chunks, counts, meta)

    if n_skipped:
        print(f"  note: skipped {n_skipped} conversations with zero valid tokens")
    return train_pools, eval_sets


def assemble_train(ds_list, balance, train_pools, X_by_dir, device):
    """Gather the (balanced) training matrix for one probe config.

    Returns (X fp32 on device, y ±1 fp32 on device, take: {pool: n})."""
    items = [(ds, cls) for ds in ds_list for cls in ("honest", "deceptive")]
    sizes = {it: len(train_pools[it][1]) for it in items}
    if balance == "balanced":
        n = min(sizes.values())
        take = {it: n for it in items}
    else:
        take = {}
        for cls in ("honest", "deceptive"):
            n_cls = min(sizes[(ds, cls)] for ds in ds_list)
            for ds in ds_list:
                take[(ds, cls)] = n_cls
    parts, ys = [], []
    for it in items:
        dirname, pool = train_pools[it]
        sel = pool[:take[it]].to(device)
        parts.append(X_by_dir[dirname].index_select(0, sel))
        ys.append(torch.full((take[it],), 1.0 if it[1] == "deceptive" else -1.0))
    X = torch.cat(parts)
    y = torch.cat(ys).to(device)
    return X, y, {f"{ds}/{cls}": int(v) for (ds, cls), v in take.items()}


def gather_eval(es: EvalSet, X_by_dir, device) -> torch.Tensor:
    parts = [X_by_dir[dirname].index_select(0, idx.to(device))
             for dirname, idx in es.chunks]
    return torch.cat(parts)


# ----------------------------------------------------------------------------
# Probe fitting (spec §7): objective 0.5 w·w + C Σ log(1+exp(-y x·w)),
# matching sklearn LogisticRegression(C=C, fit_intercept=False).
# ----------------------------------------------------------------------------

def fit_logreg_torch(X: torch.Tensor, y: torch.Tensor, C: float,
                     max_iter: int = 500) -> torch.Tensor:
    w = torch.zeros(X.shape[1], device=X.device, dtype=torch.float32,
                    requires_grad=True)
    opt = torch.optim.LBFGS(
        [w], lr=1.0, max_iter=max_iter, history_size=20,
        tolerance_grad=1e-7, tolerance_change=1e-10,
        line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = X.mv(w)
        loss = 0.5 * w.dot(w) \
            + C * torch.nn.functional.softplus(-y * z).sum()
        loss.backward()
        return loss

    def grad_inf_norm():
        with torch.no_grad():
            z = X.mv(w)
            g = w + C * (X.t().mv(-y * torch.sigmoid(-y * z)))
            return float(g.abs().max())

    ginf = math.inf
    for _ in range(3):
        opt.step(closure)
        ginf = grad_inf_norm()
        if ginf < 1e-4:
            break
    if ginf > 1e-2:
        print(f"  WARNING: LBFGS not fully converged (grad inf-norm {ginf:.2e})")
    return w.detach()


def fit_logreg_sklearn(X: torch.Tensor, y: torch.Tensor, C: float) -> torch.Tensor:
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=C, fit_intercept=False,
                            solver="lbfgs", max_iter=5000, tol=1e-8)
    lr.fit(X.cpu().numpy(), y.cpu().numpy().astype(np.int64))
    assert list(lr.classes_) == [-1, 1]  # coef_ is wrt class +1 = deceptive
    return torch.as_tensor(lr.coef_[0], dtype=torch.float32, device=X.device)


def validate_against_sklearn(Xn, y, C, out_dir: Path):
    """One-off torch-vs-sklearn check on a capped subsample (spec §7)."""
    n = Xn.shape[0]
    cap = min(VALIDATION_CAP, n)
    idx = np.random.default_rng(VALIDATION_SEED).choice(n, cap, replace=False)
    idx_t = torch.as_tensor(idx, dtype=torch.long, device=Xn.device)
    Xs, ys = Xn[idx_t], y[idx_t]
    w_t = fit_logreg_torch(Xs, ys, C)
    w_sk = fit_logreg_sklearn(Xs, ys, C)
    cos = float(torch.dot(w_t, w_sk)
                / (w_t.norm() * w_sk.norm() + 1e-30))
    rel = float((w_t - w_sk).norm() / (w_sk.norm() + 1e-30))
    ok = cos > 0.995
    msg = (f"torch-vs-sklearn validation: n={cap}, cosine={cos:.6f}, "
           f"rel weight diff={rel:.2e} -> {'OK' if ok else 'MISMATCH'}")
    print(f"  {msg}")
    if not ok:
        print("  WARNING: torch fit deviates from sklearn — investigate before "
              "trusting the grid!")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "validation.json", "w") as fh:
        json.dump({"n": cap, "C": C, "cosine": cos,
                   "rel_weight_diff": rel, "ok": ok}, fh, indent=1)
    return ok


# ----------------------------------------------------------------------------
# Metrics (spec §8)
# ----------------------------------------------------------------------------

def roc_metrics(scores: np.ndarray, labels01: np.ndarray) -> dict:
    from sklearn.metrics import roc_auc_score, roc_curve
    auroc = float(roc_auc_score(labels01, scores))
    fpr, tpr, thr = roc_curve(labels01, scores)
    rec = {}
    for t in FPR_TARGETS:
        rec[t] = float(tpr[fpr <= t].max()) if (fpr <= t).any() else 0.0
    return {
        "auroc": auroc,
        "recall_at_5fpr": rec[0.05],
        "recall_at_10fpr": rec[0.10],
        "roc": {
            "fpr": [float(v) for v in fpr],
            "tpr": [float(v) for v in tpr],
            "thresholds": [None if not np.isfinite(v) else float(v)
                           for v in thr],
        },
    }


# ----------------------------------------------------------------------------
# Train command (spec §10, §11)
# ----------------------------------------------------------------------------

def parse_layer_spec(spec: str) -> set:
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return out


def print_manifest(dirs: dict, splits: dict, layers: list, model_fname: str):
    print(f"\n=== manifest for {model_fname} ===")
    for dd in dirs.values():
        size_gb = dd.pt_path.stat().st_size / 1e9
        scored = dd.scored_path.name if dd.scored_path else "-"
        print(f"  {dd.name}: N={dd.N} m={dd.m} L={len(dd.layers)} "
              f"pt={size_gb:.1f}GB scored={scored}")
    for ds in FULL_ORDER:
        for cls in ("honest", "deceptive"):
            sp = splits[ds][cls]
            print(f"  {ds}/{cls}: {len(sp.train)} train / {len(sp.test)} test "
                  f"conversations ({sp.dirname})")
    print(f"  layers to process ({len(layers)}): {layers}")
    print("=" * 40)


def block_complete(block_dir: Path) -> bool:
    return (block_dir / "block_summary.json").exists()


def cmd_train(args):
    t_start = time.time()
    model_fname = args.model.replace("/", "--")
    root = Path(args.root)
    out_root = Path(args.out)
    model_out = out_root / model_fname
    C = 1.0 / args.reg_coeff
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        device = "cpu"

    # --- resolve all inputs up front; fail before any compute (spec §10) ---
    needed = {}
    for sources in FULL_DATASETS.values():
        for dirname, _ in sources:
            needed[dirname] = True
    for dirname, labels in CONTROL_DATASETS.values():
        needed.setdefault(dirname, labels is not None)
        if labels is not None:
            needed[dirname] = True
    dirs = {dn: load_dir(root, dn, model_fname, need_labels)
            for dn, need_labels in needed.items()}

    # Open all activation files now (cheap with mmap): checks mmap works,
    # validates shapes against metadata, and reconciles stale layer lists —
    # must happen before the layer intersection below.
    for dd in dirs.values():
        dd.acts()

    splits = build_splits(dirs, args.split_frac)

    layer_sets = [set(dd.layers) for dd in dirs.values()]
    inter = sorted(set.intersection(*layer_sets))
    max_layers = max(len(s) for s in layer_sets)
    if len(inter) < max_layers:
        print(f"WARNING: layer intersection has {len(inter)} layers but some "
              f"dataset carries {max_layers} — stale activation file? "
              f"(spec §2)")
    if args.layers:
        requested = parse_layer_spec(args.layers)
        missing = requested - set(inter)
        if missing:
            print(f"WARNING: requested layers not available everywhere, "
                  f"skipped: {sorted(missing)}")
        layers = sorted(requested & set(inter))
    else:
        layers = inter
    if not layers:
        sys.exit("no layers to process")

    print_manifest(dirs, splits, layers, model_fname)

    token_structs = None     # built lazily at the first processed layer
    validated = args.no_validate or args.backend == "sklearn"

    for layer in layers:
        block_dir = model_out / f"layer{layer}"
        if block_complete(block_dir) and not args.force:
            print(f"layer {layer}: complete, skipping (use --force to redo)")
            continue
        t0 = time.time()
        block_dir.mkdir(parents=True, exist_ok=True)

        X_by_dir = {dn: dd.layer_slice(layer, device)
                    for dn, dd in dirs.items()}
        if token_structs is None:
            token_structs = build_token_structs(dirs, splits)
        train_pools, eval_sets = token_structs

        X_eval = {name: gather_eval(es, X_by_dir, device)
                  for name, es in eval_sets.items()}

        rows, token_store = [], {}
        for probe_name, ds_list in PROBE_CONFIGS:
            probe_dir = block_dir / f"probe_{probe_name}"
            probe_dir.mkdir(parents=True, exist_ok=True)

            X, y, take = assemble_train(
                ds_list, args.balance, train_pools, X_by_dir, device)
            mu = X.mean(dim=0)
            sigma = X.std(dim=0, unbiased=False).clamp_min(1e-8)
            X.sub_(mu).div_(sigma)  # in-place; X is a fresh gather

            fit = fit_logreg_torch if args.backend == "torch" \
                else fit_logreg_sklearn
            w = fit(X, y, C)

            if not validated:
                validate_against_sklearn(X, y, C, out_root)
                validated = True
            del X
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

            torch.save({
                "w": w.cpu(), "scaler_mean": mu.cpu(),
                "scaler_scale": sigma.cpu(),
                "config": {
                    "model": args.model, "layer": layer,
                    "probe": probe_name, "train_datasets": ds_list,
                    "balance": args.balance, "take": take,
                    "reg_coeff": args.reg_coeff, "C": C,
                    "split_frac": args.split_frac, "seed": SEED,
                    "backend": args.backend, "judge": JUDGE,
                    "scored_files": {dd.name: str(dd.scored_path)
                                     for dd in dirs.values()
                                     if dd.scored_path},
                },
            }, probe_dir / "probe.pt")

            # score(x) = ((x - mu)/sigma)·w = x·(w/sigma) - mu·(w/sigma)
            weff = w / sigma
            b = -float(torch.dot(mu, weff))
            token_store[probe_name] = {}

            for name, es in eval_sets.items():
                tok = X_eval[name].mv(weff) + b
                conv_scores = torch.stack(
                    [s.mean() for s in torch.split(tok, es.counts)])
                cs = conv_scores.cpu().numpy().astype(np.float64)

                conv_json = [
                    {"k": int(k), "dir": dirname, "cls": cls, "label": label,
                     "n_tokens": int(n), "score": float(s)}
                    for (k, dirname, cls, label), n, s
                    in zip(es.meta, es.counts, cs)]
                result = {
                    "model": args.model, "layer": layer, "probe": probe_name,
                    "eval_dataset": name, "kind": es.kind,
                    "conversations": conv_json,
                }
                row = {"model": args.model, "layer": layer,
                       "probe": probe_name, "eval_dataset": name,
                       "kind": es.kind, "n_conversations": len(cs)}
                if es.kind == "full":
                    cls_arr = np.array(
                        [1 if m_[2] == "deceptive" else 0 for m_ in es.meta])
                    metrics = roc_metrics(cs, cls_arr)
                    result.update(metrics)
                    row.update({
                        "n_honest": int((cls_arr == 0).sum()),
                        "n_deceptive": int((cls_arr == 1).sum()),
                        "auroc": metrics["auroc"],
                        "recall_at_5fpr": metrics["recall_at_5fpr"],
                        "recall_at_10fpr": metrics["recall_at_10fpr"],
                        "mean_score_honest": float(cs[cls_arr == 0].mean()),
                        "mean_score_deceptive": float(cs[cls_arr == 1].mean()),
                    })
                else:
                    row.update({
                        "mean_score_all": float(cs.mean()),
                        "std_score_all": float(cs.std()),
                    })
                rows.append(row)
                with open(probe_dir / f"eval_{name}.json", "w") as fh:
                    json.dump(result, fh)

                token_store[probe_name][name] = {
                    "scores": tok.to(torch.float16).cpu(),
                    "conv_k": torch.as_tensor(
                        np.repeat([m_[0] for m_ in es.meta], es.counts),
                        dtype=torch.int32),
                    "cls": torch.as_tensor(
                        np.repeat([CLS_CODE[m_[2]] for m_ in es.meta],
                                  es.counts), dtype=torch.int8),
                }
                del tok

        torch.save(token_store, block_dir / "token_scores.pt")
        with open(block_dir / "block_summary.json", "w") as fh:
            json.dump(rows, fh, indent=1)

        del X_by_dir, X_eval
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"layer {layer}: done in {time.time() - t0:.1f}s")

    write_summary_csv(out_root)
    print(f"\ntotal: {(time.time() - t_start) / 60:.1f} min; "
          f"summary at {out_root / 'summary.csv'}")


# ----------------------------------------------------------------------------
# Summary + plot command (spec §9)
# ----------------------------------------------------------------------------

SUMMARY_COLS = ["model", "layer", "probe", "eval_dataset", "kind",
                "n_conversations", "n_honest", "n_deceptive", "auroc",
                "recall_at_5fpr", "recall_at_10fpr", "mean_score_honest",
                "mean_score_deceptive", "mean_score_all", "std_score_all"]


def collect_rows(out_root: Path) -> list:
    rows = []
    for f in sorted(out_root.glob("*/layer*/block_summary.json")):
        with open(f) as fh:
            rows.extend(json.load(fh))
    return rows


def write_summary_csv(out_root: Path) -> list:
    rows = collect_rows(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "summary.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_COLS,
                                extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return rows


def best_layer(rows, model, probe):
    """Best layer by AUROC on the probe's own test split(s); combined probes:
    mean AUROC over their constituent datasets' test splits (spec §9)."""
    train_ds = PROBE_TRAINSETS[probe]
    per_layer = {}
    for r in rows:
        if (r["model"] == model and r["probe"] == probe
                and r["kind"] == "full" and r["eval_dataset"] in train_ds):
            per_layer.setdefault(r["layer"], []).append(r["auroc"])
    scores = {lay: float(np.mean(v)) for lay, v in per_layer.items()
              if len(v) == len(train_ds)}
    if not scores:
        return None, None
    lay = max(scores, key=scores.get)
    return lay, scores[lay]


def plot_eval(eval_json: dict, png_prefix: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = eval_json["model"]
    title = (f"{model}  layer {eval_json['layer']}  "
             f"probe={eval_json['probe']}  eval={eval_json['eval_dataset']}")
    convs = eval_json["conversations"]

    if eval_json["kind"] == "full":
        fpr, tpr = eval_json["roc"]["fpr"], eval_json["roc"]["tpr"]
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(fpr, tpr)
        ax.plot([0, 1], [0, 1], "k--", lw=0.5)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR (recall)")
        ax.set_title(f"{title}\nAUROC={eval_json['auroc']:.3f}  "
                     f"R@5%={eval_json['recall_at_5fpr']:.3f}  "
                     f"R@10%={eval_json['recall_at_10fpr']:.3f}", fontsize=8)
        fig.tight_layout()
        fig.savefig(png_prefix.with_name(png_prefix.name + "_roc.png"), dpi=150)
        plt.close(fig)

        hon = [c["score"] for c in convs if c["cls"] == "honest"]
        dec = [c["score"] for c in convs if c["cls"] == "deceptive"]
        fig, ax = plt.subplots(figsize=(6, 4))
        bins = np.histogram_bin_edges(hon + dec, bins=40)
        ax.hist(hon, bins=bins, alpha=0.6, label=f"honest (n={len(hon)})",
                density=True)
        ax.hist(dec, bins=bins, alpha=0.6, label=f"deceptive (n={len(dec)})",
                density=True)
        ax.set_xlabel("mean deception score (logit)")
        ax.set_ylabel("density")
        ax.legend()
        ax.set_title(title, fontsize=8)
        fig.tight_layout()
        fig.savefig(png_prefix.with_name(png_prefix.name + "_hist.png"), dpi=150)
        plt.close(fig)
    else:
        scores = [c["score"] for c in convs]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(scores, bins=40, alpha=0.8, density=True,
                label=f"all (n={len(scores)})")
        ax.set_xlabel("mean deception score (logit)")
        ax.set_ylabel("density")
        ax.legend()
        ax.set_title(title + "  [control: expected honest]", fontsize=8)
        fig.tight_layout()
        fig.savefig(png_prefix.with_name(png_prefix.name + "_hist.png"), dpi=150)
        plt.close(fig)


def cmd_plot(args):
    out_root = Path(args.out)
    rows = write_summary_csv(out_root)
    if not rows:
        sys.exit(f"no results found under {out_root}")
    models = sorted({r["model"] for r in rows})
    eval_names = FULL_ORDER + list(CONTROL_DATASETS)
    n_pngs = 0
    for model in models:
        model_fname = model.replace("/", "--")
        for probe_name, _ in PROBE_CONFIGS:
            lay, score = best_layer(rows, model, probe_name)
            if lay is None:
                continue
            print(f"{model}  probe={probe_name}: best layer {lay} "
                  f"(selection AUROC {score:.3f})")
            plot_dir = out_root / "plots" / model_fname / probe_name
            plot_dir.mkdir(parents=True, exist_ok=True)
            probe_dir = (out_root / model_fname / f"layer{lay}"
                         / f"probe_{probe_name}")
            for name in eval_names:
                ej = probe_dir / f"eval_{name}.json"
                if not ej.exists():
                    print(f"  missing {ej}, skipping")
                    continue
                with open(ej) as fh:
                    eval_json = json.load(fh)
                plot_eval(eval_json, plot_dir / f"layer{lay}_{name}")
                n_pngs += 1
    print(f"\nwrote plots for {n_pngs} evaluations under {out_root / 'plots'}; "
          f"summary.csv regenerated")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train", help="train and evaluate the probe grid "
                                      "for one model")
    pt.add_argument("--model", required=True,
                    help=f"probed model, e.g. one of {KNOWN_MODELS}")
    pt.add_argument("--layers", default=None,
                    help="layer subset, e.g. '0-29' or '8,12,16,24'")
    pt.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    pt.add_argument("--balance", choices=["balanced", "unbalanced"],
                    default="balanced")
    pt.add_argument("--reg-coeff", type=float, default=10.0,
                    help="repe convention: sklearn C = 1/reg_coeff")
    pt.add_argument("--split-frac", type=float, default=0.8)
    pt.add_argument("--backend", choices=["torch", "sklearn"], default="torch")
    pt.add_argument("--root", default=DEFAULT_ROOT,
                    help="autocomp2 data root (default: next to this script)")
    pt.add_argument("--out", default=DEFAULT_OUT,
                    help="results dir (default: results/probe-comparison "
                         "next to this script)")
    pt.add_argument("--force", action="store_true",
                    help="recompute blocks even if outputs exist")
    pt.add_argument("--no-validate", action="store_true",
                    help="skip the one-off torch-vs-sklearn check")
    pt.set_defaults(func=cmd_train)

    pp = sub.add_parser("plot", help="regenerate summary.csv and render PNGs "
                                     "for the best layer per (model, probe)")
    pp.add_argument("--out", default=DEFAULT_OUT,
                    help="results dir (default: results/probe-comparison "
                         "next to this script)")
    pp.set_defaults(func=cmd_plot)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
