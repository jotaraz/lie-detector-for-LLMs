#!/usr/bin/env python3
"""Visualize natural-generation activations colored by judge score categories.

For each model, plots the union of multiple datasets' natural-answer activation
trajectories (the i=0 autocompletion = generation from no prefix) in a shared
PCA space, colored by the honesty-score category from one or more scored-*.json
files. Categories follow analyze-natural-answers.ipynb:
    -2 red    (x < -5)
    -1 orange (-5 <= x < 0)
     0 grey   (x == 0)
    +1 green  (0 < x <= 5)
    +2 blue   (x > 5)

Each scored file is keyed by its ``judge_model``; the HTML offers a judge
dropdown plus an "intersection" entry that only shows conversations whose
category agrees across every judge in that dataset.

Usage:
    python visualize_scored_natural.py \
        --datasets autoconv10 more-roleplay \
        [--model Qwen/Qwen2.5-72B-Instruct]
"""

import argparse
import html as html_lib
import json
import os
import re
import time
import warnings
from collections import defaultdict

import numpy as np
import torch
from scipy.sparse import SparseEfficiencyWarning
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)

METHODS = ["PCA", "LDA", "MeanDiff", "CovDiff"]

MODELS_FILE = "models.md"

SYMBOLS = [
    "circle", "square", "diamond", "triangle-up",
    "triangle-down", "cross", "x", "star",
]

CATEGORY_COLORS = {
    -2: "#dc2626",   # red
    -1: "#ea580c",   # orange
     0: "#737373",   # grey
     1: "#16a34a",   # green
     2: "#2563eb",   # blue
}
CATEGORY_ORDER = [-2, -1, 0, 1, 2]

SCORED_RE = re.compile(
    r"^scored-(?:(?P<variant>[a-zA-Z0-9]+)-)?\d{8}-\d{6}-(?P<model>.+)\.json$"
)


def load_models_list():
    with open(MODELS_FILE) as f:
        return [line.strip() for line in f if line.strip()]


def model_id_to_filename(model_id):
    return model_id.replace("/", "--")


def categorize(score):
    """Bucket a -10..10 judge score into one of {-2, -1, 0, 1, 2}."""
    if score < -5:
        return -2
    if score < 0:
        return -1
    if score == 0:
        return 0
    if score <= 5:
        return 1
    return 2


# ── Loading ──────────────────────────────────────────────────────────────────

def discover_scored_files(data_dir, model_fname, exclude_variants):
    """Return {judge_model: {index: {"score": int, "reasoning": str}}}.

    If multiple scored files share a judge_model, the one with the latest
    timestamp (i.e. last in sorted order) wins.
    """
    by_judge = {}
    for f in sorted(os.listdir(data_dir)):
        m = SCORED_RE.match(f)
        if not m:
            continue
        if m.group("model") != model_fname:
            continue
        if m.group("variant") and m.group("variant") in exclude_variants:
            continue
        with open(os.path.join(data_dir, f)) as fh:
            data = json.load(fh)
        judge = data.get("judge_model", f)
        idx_map = {s["index"]: {"score": s["score"], "reasoning": s.get("reasoning", "")}
                   for s in data["scores"]}
        by_judge[judge] = idx_map  # later timestamps overwrite earlier ones
    return by_judge


def load_dataset(dataset_name, model_id, exclude_variants):
    data_dir = "data-" + dataset_name
    fname = model_id_to_filename(model_id)
    json_path = os.path.join(data_dir, f"{fname}.json")
    pt_path = os.path.join(data_dir, f"{fname}.pt")

    if not os.path.exists(json_path):
        print(f"  [warn] Missing JSON: {json_path} — skipping {dataset_name}")
        return None
    if not os.path.exists(pt_path):
        print(f"  [warn] Missing PT: {pt_path} — skipping {dataset_name}")
        return None

    print(f"  Loading {dataset_name}/{fname} ...")
    with open(json_path) as f:
        json_data = json.load(f)
    pt_data = torch.load(pt_path, map_location="cpu", weights_only=True)
    ac_tensor = pt_data["autocompletion_activations"]  # (N, c, m, L, d)

    scored = discover_scored_files(data_dir, fname, exclude_variants)
    if scored:
        print(f"    judges: {sorted(scored.keys())}")
    else:
        print(f"    [warn] no scored-*.json files for {fname}")

    return {
        "name": dataset_name,
        "json_data": json_data,
        "ac_tensor": ac_tensor,
        "scored": scored,
    }


# ── Projection fitting ──────────────────────────────────────────────────────

MAX_FIT_PER_CLASS = 5000  # cap on tokens per class for supervised fits


def _subsample(X, max_n, rng):
    if len(X) <= max_n:
        return X
    idx = rng.choice(len(X), size=max_n, replace=False)
    return X[idx]


def fit_axes(X_valid, X_neg, X_pos, method, rng=None):
    """Return (d, 2) projection axes for `method`, or None if not fittable.

    X_valid: all valid token activations (used to fit the orthogonal-complement
             PCA axis for LDA / MeanDiff, and as the data for plain PCA).
    X_neg:   tokens from convs whose intersection-category is -2 (class A).
    X_pos:   tokens from convs whose intersection-category is +2 (class B).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n_neg = len(X_neg)
    n_pos = len(X_pos)

    if method == "PCA":
        pca = PCA(n_components=2)
        pca.fit(X_valid)
        return pca.components_.T  # (d, 2)

    if n_neg < 2 or n_pos < 2:
        return None

    # Supervised methods get subsampled inputs for speed; projection of all
    # data downstream uses the full X_valid via the returned axes.
    X_neg_fit = _subsample(X_neg, MAX_FIT_PER_CLASS, rng)
    X_pos_fit = _subsample(X_pos, MAX_FIT_PER_CLASS, rng)

    if method == "LDA":
        X = np.concatenate([X_neg_fit, X_pos_fit], axis=0)
        y = np.concatenate([np.zeros(len(X_neg_fit), dtype=int),
                            np.ones(len(X_pos_fit), dtype=int)])
        lda = LinearDiscriminantAnalysis(
            solver="eigen", shrinkage="auto", n_components=1)
        lda.fit(X, y)
        w = lda.scalings_[:, 0]
        nw = np.linalg.norm(w)
        if nw < 1e-12:
            return None
        w = w / nw

    elif method == "MeanDiff":
        mu_A = X_neg_fit.mean(axis=0)
        mu_B = X_pos_fit.mean(axis=0)
        w = mu_B - mu_A
        nw = np.linalg.norm(w)
        if nw < 1e-12:
            return None
        w = w / nw

    elif method == "CovDiff":
        # Top-2 eigvecs of Cov(A) - Cov(B), ranked by |eigenvalue|.
        C_A = np.cov(X_neg_fit, rowvar=False)
        C_B = np.cov(X_pos_fit, rowvar=False)
        M = C_A - C_B
        eigvals, eigvecs = np.linalg.eigh(M)
        order = np.argsort(-np.abs(eigvals))[:2]
        axes = eigvecs[:, order]
        # Normalize each column (eigh already returns unit-norm cols, but
        # guard against numerical drift)
        axes = axes / np.linalg.norm(axes, axis=0, keepdims=True)
        return axes  # (d, 2)

    else:
        raise ValueError(f"Unknown method: {method}")

    # For LDA / MeanDiff: axis 2 = top PCA in orthogonal complement of w.
    X_perp = X_valid - (X_valid @ w)[:, None] * w[None, :]
    pca = PCA(n_components=1)
    pca.fit(X_perp)
    v = pca.components_[0]
    nv = np.linalg.norm(v)
    if nv < 1e-12:
        return None
    v = v / nv
    return np.stack([w, v], axis=1)  # (d, 2)


# ── Build the JS payload ─────────────────────────────────────────────────────

def _truncate(s, n=300):
    s = s or ""
    return s if len(s) <= n else s[: n - 1] + "…"


def build_payload(dataset_entries):
    """Compute joint PCA per layer and assemble the JS DATA object."""
    layers_ref = dataset_entries[0]["json_data"]["layers"]
    m_ref = dataset_entries[0]["json_data"]["m"]
    for e in dataset_entries[1:]:
        if e["json_data"]["layers"] != layers_ref:
            raise ValueError(
                f"Layer mismatch: {dataset_entries[0]['name']}={layers_ref} "
                f"vs {e['name']}={e['json_data']['layers']}")
        if e["json_data"]["m"] != m_ref:
            raise ValueError(
                f"m mismatch: {dataset_entries[0]['name']}={m_ref} "
                f"vs {e['name']}={e['json_data']['m']}")

    # Union of judges across datasets, in stable order
    all_judges = []
    seen = set()
    for e in dataset_entries:
        for j in sorted(e["scored"].keys()):
            if j not in seen:
                seen.add(j)
                all_judges.append(j)

    # Per-dataset convs (no projection yet)
    datasets_out = []
    for di, entry in enumerate(dataset_entries):
        jd = entry["json_data"]
        scored = entry["scored"]
        convs = []
        for conv in jd["conversations"]:
            k = conv["k"]
            ac = conv["autocompletions"][0]  # i=0
            eos = ac.get("eos_position")
            n_gen = m_ref if eos is None else eos + 1

            per_judge = {}
            cats_in_dataset = []
            for judge in sorted(scored.keys()):
                idx_map = scored[judge]
                if k in idx_map:
                    s = idx_map[k]
                    cat = categorize(s["score"])
                    per_judge[judge] = {
                        "score": s["score"],
                        "cat": cat,
                        "reasoning": _truncate(s["reasoning"], 280),
                    }
                    cats_in_dataset.append(cat)
            if cats_in_dataset and all(c == cats_in_dataset[0] for c in cats_in_dataset):
                intersection_cat = cats_in_dataset[0]
            else:
                intersection_cat = None

            convs.append({
                "k": k,
                "system": _truncate(conv["system_prompt"], 400),
                "user": _truncate(conv["user_prompt"], 400),
                "answer": ac.get("text", ""),
                "tokens": ac.get("tokens", []),
                "n_gen": n_gen,
                "eos": eos,
                "judges": per_judge,
                "intersection": intersection_cat,
            })
        datasets_out.append({
            "name": entry["name"],
            "symbol": SYMBOLS[di % len(SYMBOLS)],
            "model": jd.get("model", ""),
            "m": m_ref,
            "N": len(convs),
            "convs": convs,
            "projections": {},
        })

    # Precompute, over the full (concatenated) row order, which rows belong to
    # convs in intersection cat -2 (class A) or +2 (class B).
    total_rows = sum(len(ds["convs"]) * m_ref for ds in datasets_out)
    is_neg_full = np.zeros(total_rows, dtype=bool)
    is_pos_full = np.zeros(total_rows, dtype=bool)
    off = 0
    for ds in datasets_out:
        N_i = len(ds["convs"])
        for ci, conv in enumerate(ds["convs"]):
            if conv["intersection"] == -2:
                is_neg_full[off + ci * m_ref: off + (ci + 1) * m_ref] = True
            elif conv["intersection"] == 2:
                is_pos_full[off + ci * m_ref: off + (ci + 1) * m_ref] = True
        off += N_i * m_ref

    fitted_methods = set()  # which methods produced output for >=1 layer

    for l_idx, layer in enumerate(layers_ref):
        parts = []
        sizes = []
        for entry in dataset_entries:
            ac = entry["ac_tensor"]  # (N, c, m, L, d)
            N_i = ac.shape[0]
            slab = ac[:, 0, :, l_idx, :].reshape(-1, ac.shape[-1])
            parts.append(slab.float().numpy())
            sizes.append(N_i * m_ref)

        all_acts = np.concatenate(parts, axis=0)
        valid = np.any(all_acts != 0, axis=1)
        if valid.sum() < 3:
            print(f"  [warn] Layer {layer}: <3 valid activations, skipping")
            continue

        X_valid = all_acts[valid]
        is_neg_valid = is_neg_full[valid]
        is_pos_valid = is_pos_full[valid]
        X_neg = X_valid[is_neg_valid]
        X_pos = X_valid[is_pos_valid]
        print(f"  Layer {layer}: |valid|={len(X_valid)}, |neg|={len(X_neg)}, "
              f"|pos|={len(X_pos)}, d={X_valid.shape[1]}")

        for method in METHODS:
            t0 = time.time()
            axes = fit_axes(X_valid, X_neg, X_pos, method)
            if axes is None:
                print(f"    [warn] {method}: not fittable for layer {layer}")
                continue
            coords = np.full((len(all_acts), 2), np.nan, dtype=np.float64)
            coords[valid] = X_valid @ axes
            fitted_methods.add(method)
            print(f"    {method}: fit+project in {time.time() - t0:.1f}s")

            off = 0
            for di, entry in enumerate(dataset_entries):
                N_i = entry["ac_tensor"].shape[0]
                block = coords[off: off + sizes[di]].reshape(N_i, m_ref, 2)
                key = f"layer{layer}_{method}"
                datasets_out[di]["projections"][key] = block.tolist()
                off += sizes[di]

        # Free transient layer arrays
        del all_acts, X_valid, X_neg, X_pos, parts

    return {
        "datasets": datasets_out,
        "layers": layers_ref,
        "judges": all_judges,
        "methods": [m for m in METHODS if m in fitted_methods],
        "m": m_ref,
        "category_colors": {str(k): v for k, v in CATEGORY_COLORS.items()},
        "category_order": CATEGORY_ORDER,
    }


# ── HTML template ────────────────────────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Scored natural-generation activations &ndash; {title}</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #fafafa; }
  h1 { font-size: 1.05rem; padding: 8px 14px; }
  .controls { padding: 5px 14px; display: flex; flex-wrap: wrap; gap: 6px;
              align-items: center; border-bottom: 1px solid #ddd; }
  .controls .group { display: flex; flex-wrap: wrap; gap: 4px;
                     align-items: center; margin-right: 14px; }
  .controls label { font-size: 0.78rem; font-weight: 600; margin-right: 4px; }
  button { font-size: 0.75rem; padding: 3px 8px; border: 1px solid #aaa;
           border-radius: 4px; background: #fff; cursor: pointer; }
  button.active { background: #4a90d9; color: #fff; border-color: #4a90d9; }
  select { font-size: 0.78rem; padding: 2px 4px; }
  #plot { width: 100%; height: calc(100vh - 170px); }
  .cat-btn { font-weight: 700; }
</style>
</head>
<body>

<h1>Scored natural-generation activations &ndash; <span id="model-name"></span></h1>

<div class="controls">
  <div class="group"><label>Layer:</label><select id="sel-layer"></select></div>
  <div class="group"><label>Method:</label><select id="sel-method"></select></div>
  <div class="group"><label>Judge:</label><select id="sel-judge"></select></div>
  <div class="group"><label>Datasets:</label><span id="dataset-buttons"></span></div>
</div>

<div class="controls">
  <div class="group">
    <label>Categories:</label>
    <span id="category-buttons"></span>
    <button onclick="showAllCats()">show all</button>
    <button onclick="hideAllCats()">hide all</button>
  </div>
  <div class="group">
    <label>Style:</label>
    <button id="btn-lines" onclick="toggleLines()">connect tokens</button>
  </div>
  <div class="group" id="counts"></div>
</div>

<div class="controls">
  <div class="group">
    <label>Token j:</label>
    <button onclick="showAllJ()">show all</button>
    <button onclick="hideAllJ()">hide all</button>
    <span id="j-buttons" style="display:flex;flex-wrap:wrap;gap:3px;"></span>
  </div>
</div>

<div id="plot"></div>

<script>
const DATA = __DATA_PLACEHOLDER__;

const state = {
  layerIdx: 0,
  method: DATA.methods[0],
  judge: DATA.judges.length ? DATA.judges[0] : null,
  visibleDatasets: new Set(DATA.datasets.map((_, i) => i)),
  visibleCats: new Set(DATA.category_order),
  visibleJ: new Set(Array.from({length: DATA.m}, (_, i) => i)),
  showLines: true,
};

function esc(s) {
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

(function init() {
  document.getElementById("model-name").textContent =
    DATA.datasets.map(d => d.name).join(" + ") + "  –  " + (DATA.datasets[0].model || "");

  const selL = document.getElementById("sel-layer");
  DATA.layers.forEach((l, idx) => {
    const o = document.createElement("option");
    o.value = idx; o.textContent = "Layer " + l;
    selL.appendChild(o);
  });
  selL.onchange = () => { state.layerIdx = +selL.value; render(); };

  const selM = document.getElementById("sel-method");
  DATA.methods.forEach(m => {
    const o = document.createElement("option");
    o.value = m; o.textContent = m;
    selM.appendChild(o);
  });
  selM.value = DATA.methods[0];
  selM.onchange = () => { state.method = selM.value; render(); };

  const selJ = document.getElementById("sel-judge");
  const judgeOptions = DATA.judges.concat(["intersection"]);
  judgeOptions.forEach(j => {
    const o = document.createElement("option");
    o.value = j; o.textContent = j;
    selJ.appendChild(o);
  });
  selJ.value = judgeOptions[0];
  state.judge = judgeOptions[0];
  selJ.onchange = () => { state.judge = selJ.value; render(); };

  const db = document.getElementById("dataset-buttons");
  DATA.datasets.forEach((ds, di) => {
    const b = document.createElement("button");
    b.textContent = ds.name + " (" + ds.symbol + ")";
    b.id = "btn-ds-" + di;
    b.onclick = () => toggleDataset(di);
    db.appendChild(b);
  });

  const cb = document.getElementById("category-buttons");
  DATA.category_order.forEach(c => {
    const b = document.createElement("button");
    b.className = "cat-btn";
    b.textContent = (c > 0 ? "+" : "") + c;
    b.id = "btn-cat-" + c;
    b.style.color = DATA.category_colors[String(c)];
    b.onclick = () => toggleCat(c);
    cb.appendChild(b);
  });

  const jb = document.getElementById("j-buttons");
  for (let j = 0; j < DATA.m; j++) {
    const b = document.createElement("button");
    b.textContent = j;
    b.id = "btn-j-" + j;
    b.style.minWidth = "26px";
    b.style.padding = "2px 5px";
    b.onclick = () => toggleJ(j);
    jb.appendChild(b);
  }

  syncButtons();
  render();
})();

function toggleDataset(di) {
  if (state.visibleDatasets.has(di)) state.visibleDatasets.delete(di);
  else state.visibleDatasets.add(di);
  syncButtons(); render();
}
function toggleCat(c) {
  if (state.visibleCats.has(c)) state.visibleCats.delete(c);
  else state.visibleCats.add(c);
  syncButtons(); render();
}
function showAllCats() {
  DATA.category_order.forEach(c => state.visibleCats.add(c));
  syncButtons(); render();
}
function hideAllCats() {
  state.visibleCats.clear();
  syncButtons(); render();
}
function toggleLines() {
  state.showLines = !state.showLines;
  syncButtons(); render();
}
function toggleJ(j) {
  if (state.visibleJ.has(j)) state.visibleJ.delete(j);
  else state.visibleJ.add(j);
  syncButtons(); render();
}
function showAllJ() {
  for (let j = 0; j < DATA.m; j++) state.visibleJ.add(j);
  syncButtons(); render();
}
function hideAllJ() {
  state.visibleJ.clear();
  syncButtons(); render();
}

function syncButtons() {
  DATA.datasets.forEach((ds, di) => {
    const b = document.getElementById("btn-ds-" + di);
    b.className = state.visibleDatasets.has(di) ? "active" : "";
  });
  DATA.category_order.forEach(c => {
    const b = document.getElementById("btn-cat-" + c);
    const on = state.visibleCats.has(c);
    if (on) {
      b.style.background = DATA.category_colors[String(c)];
      b.style.color = "#fff";
      b.style.borderColor = DATA.category_colors[String(c)];
    } else {
      b.style.background = "#fff";
      b.style.color = DATA.category_colors[String(c)];
      b.style.borderColor = DATA.category_colors[String(c)];
    }
  });
  document.getElementById("btn-lines").className = state.showLines ? "active" : "";
  for (let j = 0; j < DATA.m; j++) {
    const b = document.getElementById("btn-j-" + j);
    b.className = state.visibleJ.has(j) ? "active" : "";
  }
}

function convCategory(conv) {
  if (state.judge === "intersection") return conv.intersection;
  const j = conv.judges[state.judge];
  return j ? j.cat : null;
}

function buildHoverTexts(ds, conv, n_pts) {
  // One hover text per generated-token marker
  const sys = esc(conv.system);
  const usr = esc(conv.user);
  const tokens = conv.tokens;
  // Pre-render the full answer with each token's escaped form
  const tokEsc = tokens.map(esc);

  // Judges block (rendered once, reused)
  const judgeLines = [];
  for (const jn of DATA.judges) {
    const jr = conv.judges[jn];
    if (!jr) continue;
    const color = DATA.category_colors[String(jr.cat)];
    judgeLines.push(
      `<b>${esc(jn)}</b>: <b style="color:${color}">${jr.score}</b> &middot; ${esc(jr.reasoning)}`
    );
  }
  const interLine = conv.intersection === null
    ? "<i>intersection: disagree</i>"
    : `<i>intersection: <b style="color:${DATA.category_colors[String(conv.intersection)]}">${conv.intersection}</b></i>`;
  const judgeBlock = judgeLines.concat([interLine]).join("<br>");

  const out = [];
  for (let j = 0; j < n_pts; j++) {
    const parts = [];
    for (let jj = 0; jj < tokens.length; jj++) {
      if (jj === j) parts.push(`<b style="color:#e91e8a">${tokEsc[jj]}</b>`);
      else parts.push(tokEsc[jj]);
    }
    const answer = parts.join("");
    out.push(
      `<b>${esc(ds.name)} &middot; k=${conv.k}</b> &middot; j=${j}/${conv.n_gen - 1}<br>` +
      `<b>S:</b> ${sys}<br><b>U:</b> ${usr}<br>` +
      `<b>A:</b> ${answer}<br>` +
      `<span style="color:#888">─────────</span><br>${judgeBlock}`
    );
  }
  return out;
}

function render() {
  const layerKey = "layer" + DATA.layers[state.layerIdx] + "_" + state.method;
  const traces = [];
  const counts = { total: 0 };
  DATA.category_order.forEach(c => counts[c] = 0);

  for (let di = 0; di < DATA.datasets.length; di++) {
    if (!state.visibleDatasets.has(di)) continue;
    const ds = DATA.datasets[di];
    const proj = ds.projections[layerKey];
    if (!proj) continue;

    for (let ci = 0; ci < ds.convs.length; ci++) {
      const conv = ds.convs[ci];
      const cat = convCategory(conv);
      if (cat === null || cat === undefined) continue;
      if (!state.visibleCats.has(cat)) continue;
      counts[cat] += 1; counts.total += 1;

      const color = DATA.category_colors[String(cat)];
      const n_pts = Math.min(conv.n_gen, DATA.m);
      const hovers = buildHoverTexts(ds, conv, n_pts);

      // Collect j's that are both visible and have a valid coord, then
      // split into runs of consecutive j's so lines don't jump over gaps.
      const enabled = [];
      for (let j = 0; j < n_pts; j++) {
        if (!state.visibleJ.has(j)) continue;
        const pt = proj[conv.k][j];
        if (pt == null || pt[0] === null || isNaN(pt[0])) continue;
        enabled.push(j);
      }
      if (enabled.length === 0) continue;

      const runs = [];
      for (let idx = 0; idx < enabled.length; idx++) {
        if (idx === 0 || enabled[idx] !== enabled[idx - 1] + 1) runs.push([]);
        runs[runs.length - 1].push(enabled[idx]);
      }
      for (const run of runs) {
        const xs = [], ys = [], texts = [];
        for (const j of run) {
          const pt = proj[conv.k][j];
          xs.push(pt[0]); ys.push(pt[1]); texts.push(hovers[j]);
        }
        const mode = (state.showLines && xs.length > 1) ? "lines+markers" : "markers";
        traces.push({
          x: xs, y: ys, mode: mode,
          marker: { symbol: ds.symbol, size: 6, color: color,
                    line: { color: "#000", width: 0.3 } },
          line: { color: color, width: 1, dash: "dot" },
          hovertext: texts, hoverinfo: "text",
          showlegend: false,
        });
      }
    }
  }

  const countsEl = document.getElementById("counts");
  countsEl.innerHTML = "<span style='font-size:0.78rem;color:#555'>" +
    "convs shown: <b>" + counts.total + "</b> &middot; " +
    DATA.category_order.map(c => {
      const col = DATA.category_colors[String(c)];
      return `<span style="color:${col}">${(c>0?"+":"")+c}</span>: ${counts[c]}`;
    }).join(" &middot; ") + "</span>";

  const layout = {
    hovermode: "closest",
    hoverlabel: { bgcolor: "#fff", bordercolor: "#888",
                  font: { size: 11, family: "system-ui, sans-serif" },
                  align: "left", namelength: -1 },
    xaxis: { title: state.method + " axis 1", zeroline: false },
    yaxis: { title: state.method + " axis 2", zeroline: false },
    margin: { l: 50, r: 20, t: 10, b: 40 },
  };
  Plotly.react("plot", traces, layout, { responsive: true });
}
</script>
</body>
</html>"""


def generate_html(payload, output_path, title):
    data_json = json.dumps(payload, ensure_ascii=False)
    page = HTML_TEMPLATE.replace("__DATA_PLACEHOLDER__", data_json)
    page = page.replace("{title}", title)
    with open(output_path, "w") as f:
        f.write(page)


# ── Main ─────────────────────────────────────────────────────────────────────

def process_model(model_id, dataset_names, exclude_variants):
    entries = []
    for name in dataset_names:
        e = load_dataset(name, model_id, exclude_variants)
        if e is not None:
            entries.append(e)

    if not entries:
        print("  [warn] No datasets loaded — skipping model")
        return

    if not any(e["scored"] for e in entries):
        print("  [warn] No scored-*.json files found in any dataset — skipping model")
        return

    print(f"  Computing joint PCA over {len(entries)} dataset(s) ...")
    payload = build_payload(entries)

    fname = model_id_to_filename(model_id)
    suffix = "+".join(e["name"] for e in entries)
    out = f"scored_{suffix}_{fname}.html"
    print(f"  Writing {out} ...")
    title = " + ".join(e["name"] for e in entries) + " — " + model_id
    generate_html(payload, out, title)
    print("  Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize scored natural-generation activations as interactive HTML.")
    parser.add_argument(
        "--datasets", nargs="+", required=True,
        help="Autoconv-style dataset names (e.g. autoconv10 more-roleplay). "
             "Each maps to data-<name>/.")
    parser.add_argument(
        "--model", type=str, default=None,
        help="HuggingFace model ID. If omitted, all models from models.md are processed.")
    parser.add_argument(
        "--exclude-variants", nargs="*", default=["ac1", "ac3", "ac5"],
        help="Variant tags to ignore in scored-<variant>-*.json filenames.")
    args = parser.parse_args()

    models = [args.model] if args.model else load_models_list()
    exclude_variants = set(args.exclude_variants)

    for model_id in models:
        print(f"\n{'=' * 60}\n  {model_id}\n{'=' * 60}")
        process_model(model_id, args.datasets, exclude_variants)

    print("\nAll done.")


if __name__ == "__main__":
    main()
