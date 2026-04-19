#!/usr/bin/env python3
"""Visualize union of multiple autoconv datasets as an interactive HTML page.

Reads JSON + PT files from multiple data-autoconvN/ directories, projects all
activations into a shared 2-D space via PCA / UMAP / Isomap, and writes a
self-contained interactive HTML file per model.

Usage:
    python visualize_union.py \\
        --datasets autoconv4 autoconv5 autoconv6 \\
        [--model meta-llama/Meta-Llama-3.1-8B-Instruct] \\
        [--methods PCA UMAP Isomap]
"""

import argparse
import html as html_lib
import json
import os
import warnings
import numpy as np
import torch
from scipy.sparse import SparseEfficiencyWarning

warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)

from sklearn.decomposition import PCA
from sklearn.manifold import Isomap

ISOMAP_N_NEIGHBORS = 10
UMAP_N_NEIGHBORS   = 15
UMAP_MIN_DIST      = 0.1
UMAP_METRIC        = "euclidean"

MODELS_FILE = "models.md"

SYMBOLS = [
    "circle", "square", "diamond", "triangle-up",
    "triangle-down", "cross", "x", "star",
]
DATASET_COLORS = [
    "#4a90d9", "#e67e22", "#27ae60", "#9b59b6",
    "#e74c3c", "#1abc9c", "#f39c12", "#8e44ad",
]


def load_models_list():
    with open(MODELS_FILE) as f:
        return [line.strip() for line in f if line.strip()]


def model_id_to_filename(model_id):
    return model_id.replace("/", "--")


# ── Dimensionality reduction ────────────────────────────────────────────────

def _try_import_umap():
    try:
        import umap
        return umap
    except ImportError:
        return None


def project_all(acts_flat, valid_mask, methods):
    """Project acts_flat (n_points, d) to 2-D with each method.

    Only valid rows (where valid_mask is True) are used for fitting.
    Invalid rows receive NaN coordinates.
    Returns {method_name: (n_points, 2) ndarray}.
    """
    results = {}
    valid_acts = acts_flat[valid_mask]

    for name in methods:
        coords = np.full((len(acts_flat), 2), np.nan)

        if name == "PCA":
            model = PCA(n_components=2)
            model.fit(valid_acts)
            coords[valid_mask] = model.transform(valid_acts)

        elif name == "UMAP":
            umap_mod = _try_import_umap()
            if umap_mod is None:
                print("  [warn] umap-learn not installed, skipping UMAP")
                continue
            model = umap_mod.UMAP(
                n_components=2,
                n_neighbors=UMAP_N_NEIGHBORS,
                min_dist=UMAP_MIN_DIST,
                metric=UMAP_METRIC,
            )
            coords[valid_mask] = model.fit_transform(valid_acts)

        elif name == "Isomap":
            n_neighbors = min(ISOMAP_N_NEIGHBORS, len(valid_acts) - 1)
            model = Isomap(n_components=2, n_neighbors=n_neighbors)
            coords[valid_mask] = model.fit_transform(valid_acts)

        results[name] = coords

    return results


# ── Label helpers (identical to visualize_labeled_activations.py) ────────────

def _esc(s):
    return html_lib.escape(s)


def _bold_color(k):
    return "#e91e8a" if k % 2 == 0 else "lightgreen"


def make_prefix_label(ref_tokens, i, c_k, k):
    color = _bold_color(k)
    parts = []
    for idx in range(c_k):
        tok = _esc(ref_tokens[idx])
        if idx == i:
            parts.append(f'<b style="color:{color}">{tok}</b>')
        else:
            parts.append(tok)
    return "".join(parts)


def make_full_convo_label(ref_tokens):
    return "".join(_esc(t) for t in ref_tokens)


def make_last_act_label(ref_tokens, c_k):
    parts = []
    for idx, tok in enumerate(ref_tokens):
        tok_esc = _esc(tok)
        if idx == c_k - 1:
            parts.append(f"<u>{tok_esc}</u>")
        else:
            parts.append(tok_esc)
    return "".join(parts)


def make_autocomplete_label(ref_tokens, ac_tokens, i, j, k):
    color = _bold_color(k)
    parts = []
    if i > 0:
        prefix_text = "".join(_esc(t) for t in ref_tokens[:i])
        parts.append(f"<u>{prefix_text}</u>")
    for jj, tok in enumerate(ac_tokens):
        if tok == "":
            continue
        tok_esc = _esc(tok)
        if jj == j:
            parts.append(f'<b style="color:{color}">{tok_esc}</b>')
        else:
            parts.append(tok_esc)
    return "".join(parts)


def build_labels_for_dataset(json_data):
    """Return the per-conversation label list for one dataset."""
    c = json_data["c"]
    m = json_data["m"]
    label_data = []
    for conv in json_data["conversations"]:
        k     = conv["k"]
        n_k   = conv["n"]
        c_k   = min(c, n_k)
        ref   = conv["reference_tokens"]

        prefix_labels = [make_prefix_label(ref, i, c_k, k) for i in range(c_k)]
        fc_label      = make_full_convo_label(ref)
        last_act_label = make_last_act_label(ref, c_k) if c_k > 0 else ""

        ac_labels = []
        for ac in conv["autocompletions"]:
            i   = ac["i"]
            row = []
            for j in range(m):
                if ac["tokens"][j] == "":
                    row.append("")
                else:
                    row.append(make_autocomplete_label(ref, ac["tokens"], i, j, k))
            ac_labels.append(row)

        label_data.append({
            "k":              k,
            "c_k":            c_k,
            "n_k":            n_k,
            "prefix_labels":  prefix_labels,
            "full_convo_label": fc_label,
            "last_act_label": last_act_label,
            "ac_labels":      ac_labels,
            "eos_positions":  [ac.get("eos_position") for ac in conv["autocompletions"]],
        })
    return label_data


# ── Joint projection and data-payload builder ────────────────────────────────

def build_union_vis_data(dataset_entries, methods):
    """Build the JSON-serialisable DATA object for the union HTML.

    dataset_entries: list of dicts with keys
        name, json_data, ac_tensor (N,c,m,L,d), fc_tensor (N,L,d)
    methods: list of method names to try
    """
    layer_indices = dataset_entries[0]["json_data"]["layers"]

    # Build output skeletons (labels are layer-independent)
    datasets_out = []
    for di, entry in enumerate(dataset_entries):
        jd  = entry["json_data"]
        N   = len(jd["conversations"])
        datasets_out.append({
            "name":        entry["name"],
            "symbol":      SYMBOLS[di % len(SYMBOLS)],
            "color":       DATASET_COLORS[di % len(DATASET_COLORS)],
            "model":       jd.get("model", ""),
            "c":           jd["c"],
            "m":           jd["m"],
            "N":           N,
            "labels":      build_labels_for_dataset(jd),
            "projections": {},
        })

    # Compute joint projection per layer
    for l_idx, layer in enumerate(layer_indices):
        ac_parts, fc_parts, ac_sizes, fc_sizes = [], [], [], []

        for entry in dataset_entries:
            act = entry["ac_tensor"]
            fct = entry["fc_tensor"]
            N_i, c_i, m_i = act.shape[0], act.shape[1], act.shape[2]
            d = act.shape[-1]
            ac_flat = act[:, :, :, l_idx, :].reshape(-1, d).float().numpy()
            fc_flat = fct[:, l_idx, :].float().numpy()
            ac_parts.append(ac_flat)
            fc_parts.append(fc_flat)
            ac_sizes.append(N_i * c_i * m_i)
            fc_sizes.append(N_i)

        all_acts = np.concatenate(ac_parts + fc_parts, axis=0)
        valid    = np.any(all_acts != 0, axis=1)

        if valid.sum() < 3:
            print(f"  [warn] Layer {layer}: fewer than 3 valid activations, skipping")
            continue

        proj_results = project_all(all_acts, valid, methods)

        for method_name, coords in proj_results.items():
            n_ac_total = sum(ac_sizes)
            ac_coords_all = coords[:n_ac_total]
            fc_coords_all = coords[n_ac_total:]

            ac_off = fc_off = 0
            for di, entry in enumerate(dataset_entries):
                jd   = entry["json_data"]
                N_i  = len(jd["conversations"])
                c_i, m_i = jd["c"], jd["m"]
                key  = f"layer{layer}_{method_name}"

                datasets_out[di]["projections"][key] = (
                    ac_coords_all[ac_off: ac_off + ac_sizes[di]]
                    .reshape(N_i, c_i, m_i, 2).tolist()
                )
                datasets_out[di]["projections"][key + "_fc"] = (
                    fc_coords_all[fc_off: fc_off + fc_sizes[di]].tolist()
                )
                ac_off += ac_sizes[di]
                fc_off += fc_sizes[di]

    valid_methods = [
        m for m in methods
        if any(
            f"layer{layer}_{m}" in ds["projections"]
            for ds in datasets_out
            for layer in layer_indices
        )
    ]

    return {
        "datasets": datasets_out,
        "layers":   layer_indices,
        "methods":  valid_methods,
    }


# ── HTML template ────────────────────────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Union Activation Visualisation – {title}</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #fafafa; }
  h1 { font-size: 1.1rem; padding: 10px 16px; }
  .controls { padding: 6px 16px; display: flex; flex-wrap: wrap; gap: 6px;
              align-items: center; border-bottom: 1px solid #ddd; }
  .controls .group { display: flex; flex-wrap: wrap; gap: 4px;
                     align-items: center; margin-right: 14px; }
  .controls label { font-size: 0.78rem; font-weight: 600; margin-right: 4px; }
  .ds-group { display: flex; flex-wrap: wrap; gap: 4px; align-items: center;
              margin-right: 12px; border-left: 3px solid #ccc; padding-left: 8px; }
  .ds-label { font-size: 0.75rem; font-weight: 700; margin-right: 4px; white-space: nowrap; }
  button { font-size: 0.75rem; padding: 3px 8px; border: 1px solid #aaa;
           border-radius: 4px; background: #fff; cursor: pointer; }
  button.active { background: #4a90d9; color: #fff; border-color: #4a90d9; }
  button.disabled { opacity: 0.4; pointer-events: none; }
  select { font-size: 0.78rem; padding: 2px 4px; }
  #plot { width: 100%; height: calc(100vh - 270px); }
  .hidden { display: none !important; }
</style>
</head>
<body>

<h1>Union Activation Visualisation &ndash; <span id="model-name"></span></h1>

<!-- Row 1: layer & method -->
<div class="controls">
  <div class="group">
    <label>Layer:</label><select id="sel-layer"></select>
  </div>
  <div class="group">
    <label>Method:</label><select id="sel-method"></select>
  </div>
</div>

<!-- Row 2: dataset toggles -->
<div class="controls" id="dataset-bar">
  <div class="group"><label>Datasets:</label></div>
  <div class="group" id="dataset-buttons"></div>
</div>

<!-- Row 3: per-dataset conversation toggles -->
<div class="controls" id="conv-bar">
  <div class="group"><label>Conversations:</label></div>
  <div id="conv-groups" style="display:flex;flex-wrap:wrap;gap:6px;"></div>
</div>

<!-- Row 4: views (global) -->
<div class="controls">
  <div class="group">
    <label>View:</label>
    <button id="btn-prefix" onclick="toggleAllPrefix()">prefix</button>
    <button id="btn-fc"     onclick="toggleView('fullConvo')">full convo</button>
    <button id="btn-lastact" onclick="toggleView('lastAct')">last act</button>
    <button id="btn-ac"     onclick="toggleAllAC()">autocomplete</button>
  </div>
</div>

<!-- Row 5: per-dataset prefix-i sub-buttons -->
<div class="controls hidden" id="prefix-bar">
  <div class="group"><label>Prefix i:</label></div>
  <div id="prefix-groups" style="display:flex;flex-wrap:wrap;gap:6px;"></div>
</div>

<!-- Row 6: per-dataset autocomplete-i sub-buttons -->
<div class="controls hidden" id="ac-bar">
  <div class="group"><label>AC i:</label></div>
  <div id="ac-groups" style="display:flex;flex-wrap:wrap;gap:6px;"></div>
</div>

<div id="plot"></div>

<script>
// ── Embedded data ──────────────────────────────────────────────────────────
const DATA = __DATA_PLACEHOLDER__;

// ── Colour helpers ─────────────────────────────────────────────────────────
function prefixColor(k) { return k % 2 === 0 ? "#2563eb" : "#dc2626"; }
function acColor(k)     { return k % 2 === 0 ? "#16a34a" : "#ea580c"; }

// ── State ──────────────────────────────────────────────────────────────────
const state = {
  layerIdx: 0,
  method: DATA.methods[0],
  visibleDatasets: new Set(DATA.datasets.map((_, i) => i)),
  // Arrays indexed by dataset index
  visibleK: DATA.datasets.map(ds => new Set(Array.from({length: ds.N}, (_, i) => i))),
  prefixI:  DATA.datasets.map(ds => new Set(Array.from({length: ds.c}, (_, i) => i))),
  autoI:    DATA.datasets.map(() => new Set()),
  showFullConvo: false,
  showLastAct:   false,
};

// ── Init controls ──────────────────────────────────────────────────────────
(function init() {
  document.getElementById("model-name").textContent =
    DATA.datasets.map(d => d.name).join(" + ") + "  \u2013  " + (DATA.datasets[0].model || "");

  // Layer selector
  const selL = document.getElementById("sel-layer");
  DATA.layers.forEach((l, idx) => {
    const o = document.createElement("option");
    o.value = idx; o.textContent = "Layer " + l;
    selL.appendChild(o);
  });
  selL.onchange = () => { state.layerIdx = +selL.value; render(); };

  // Method selector
  const selM = document.getElementById("sel-method");
  DATA.methods.forEach(m => {
    const o = document.createElement("option");
    o.value = m; o.textContent = m;
    selM.appendChild(o);
  });
  selM.onchange = () => { state.method = selM.value; render(); };

  // Row 2: dataset toggle buttons
  const db = document.getElementById("dataset-buttons");
  DATA.datasets.forEach((ds, di) => {
    const b = document.createElement("button");
    b.textContent = ds.name;
    b.id = "btn-ds-" + di;
    b.onclick = () => toggleDataset(di);
    db.appendChild(b);
  });

  // Row 3: per-dataset conversation buttons
  const cg = document.getElementById("conv-groups");
  DATA.datasets.forEach((ds, di) => {
    const wrap = document.createElement("div");
    wrap.className = "ds-group";
    wrap.id = "conv-group-" + di;
    wrap.style.borderLeftColor = ds.color;

    const lbl = document.createElement("span");
    lbl.className = "ds-label";
    lbl.textContent = ds.name + ":";
    wrap.appendChild(lbl);

    const hideBtn = document.createElement("button");
    hideBtn.textContent = "hide all";
    hideBtn.onclick = () => hideAllK(di);
    wrap.appendChild(hideBtn);

    const showBtn = document.createElement("button");
    showBtn.textContent = "show all";
    showBtn.onclick = () => showAllK(di);
    wrap.appendChild(showBtn);

    for (let k = 0; k < ds.N; k++) {
      const b = document.createElement("button");
      b.textContent = "k=" + k;
      b.id = "btn-k-" + di + "-" + k;
      b.onclick = () => toggleK(di, k);
      wrap.appendChild(b);
    }
    cg.appendChild(wrap);
  });

  // Row 5: per-dataset prefix-i buttons
  const pg = document.getElementById("prefix-groups");
  DATA.datasets.forEach((ds, di) => {
    const wrap = document.createElement("div");
    wrap.className = "ds-group";
    wrap.id = "prefix-group-" + di;
    wrap.style.borderLeftColor = ds.color;

    const lbl = document.createElement("span");
    lbl.className = "ds-label";
    lbl.textContent = ds.name + ":";
    wrap.appendChild(lbl);

    const showBtn = document.createElement("button");
    showBtn.textContent = "show all";
    showBtn.onclick = () => showAllPrefixI(di);
    wrap.appendChild(showBtn);

    const hideBtn = document.createElement("button");
    hideBtn.textContent = "hide all";
    hideBtn.onclick = () => hideAllPrefixI(di);
    wrap.appendChild(hideBtn);

    for (let i = 0; i < ds.c; i++) {
      const b = document.createElement("button");
      b.textContent = "i=" + i;
      b.id = "btn-pi-" + di + "-" + i;
      b.onclick = () => togglePrefixI(di, i);
      wrap.appendChild(b);
    }
    pg.appendChild(wrap);
  });

  // Row 6: per-dataset AC-i buttons
  const ag = document.getElementById("ac-groups");
  DATA.datasets.forEach((ds, di) => {
    const wrap = document.createElement("div");
    wrap.className = "ds-group";
    wrap.id = "ac-group-" + di;
    wrap.style.borderLeftColor = ds.color;

    const lbl = document.createElement("span");
    lbl.className = "ds-label";
    lbl.textContent = ds.name + ":";
    wrap.appendChild(lbl);

    const showBtn = document.createElement("button");
    showBtn.textContent = "show all";
    showBtn.onclick = () => showAllACI(di);
    wrap.appendChild(showBtn);

    const hideBtn = document.createElement("button");
    hideBtn.textContent = "hide all";
    hideBtn.onclick = () => hideAllACI(di);
    wrap.appendChild(hideBtn);

    for (let i = 0; i < ds.c; i++) {
      const b = document.createElement("button");
      b.textContent = "i=" + i;
      b.id = "btn-i-" + di + "-" + i;
      b.onclick = () => toggleI(di, i);
      wrap.appendChild(b);
    }
    ag.appendChild(wrap);
  });

  syncButtons();
  render();
})();

// ── Button handlers ────────────────────────────────────────────────────────
function toggleDataset(di) {
  if (state.visibleDatasets.has(di)) state.visibleDatasets.delete(di);
  else state.visibleDatasets.add(di);
  syncButtons(); render();
}
function hideAllK(di) {
  state.visibleK[di].clear(); syncButtons(); render();
}
function showAllK(di) {
  for (let k = 0; k < DATA.datasets[di].N; k++) state.visibleK[di].add(k);
  syncButtons(); render();
}
function toggleK(di, k) {
  if (state.visibleK[di].has(k)) state.visibleK[di].delete(k);
  else state.visibleK[di].add(k);
  syncButtons(); render();
}
function toggleView(v) {
  if (v === "fullConvo") state.showFullConvo = !state.showFullConvo;
  if (v === "lastAct")   state.showLastAct   = !state.showLastAct;
  syncButtons(); render();
}
function toggleAllPrefix() {
  const allEmpty = DATA.datasets.every((_, di) => state.prefixI[di].size === 0);
  DATA.datasets.forEach((ds, di) => {
    state.prefixI[di].clear();
    if (allEmpty) for (let i = 0; i < ds.c; i++) state.prefixI[di].add(i);
  });
  syncButtons(); render();
}
function togglePrefixI(di, i) {
  if (state.prefixI[di].has(i)) state.prefixI[di].delete(i);
  else state.prefixI[di].add(i);
  syncButtons(); render();
}
function showAllPrefixI(di) {
  for (let i = 0; i < DATA.datasets[di].c; i++) state.prefixI[di].add(i);
  syncButtons(); render();
}
function hideAllPrefixI(di) {
  state.prefixI[di].clear(); syncButtons(); render();
}
function toggleAllAC() {
  const allEmpty = DATA.datasets.every((_, di) => state.autoI[di].size === 0);
  DATA.datasets.forEach((ds, di) => {
    state.autoI[di].clear();
    if (allEmpty) for (let i = 0; i < ds.c; i++) state.autoI[di].add(i);
  });
  syncButtons(); render();
}
function toggleI(di, i) {
  if (state.autoI[di].has(i)) state.autoI[di].delete(i);
  else state.autoI[di].add(i);
  syncButtons(); render();
}
function showAllACI(di) {
  for (let i = 0; i < DATA.datasets[di].c; i++) state.autoI[di].add(i);
  syncButtons(); render();
}
function hideAllACI(di) {
  state.autoI[di].clear(); syncButtons(); render();
}

// ── Sync button styles ─────────────────────────────────────────────────────
function syncButtons() {
  DATA.datasets.forEach((ds, di) => {
    const dsVisible = state.visibleDatasets.has(di);

    // Dataset toggle button
    const dsBtn = document.getElementById("btn-ds-" + di);
    if (dsVisible) {
      dsBtn.style.background   = ds.color;
      dsBtn.style.color        = "#fff";
      dsBtn.style.borderColor  = ds.color;
    } else {
      dsBtn.style.background  = "#fff";
      dsBtn.style.color       = "#333";
      dsBtn.style.borderColor = ds.color;
    }

    // Conversation buttons
    for (let k = 0; k < ds.N; k++) {
      const b = document.getElementById("btn-k-" + di + "-" + k);
      b.className = !dsVisible ? "disabled" : (state.visibleK[di].has(k) ? "active" : "");
    }

    // Prefix-i buttons
    for (let i = 0; i < ds.c; i++) {
      const b = document.getElementById("btn-pi-" + di + "-" + i);
      b.className = !dsVisible ? "disabled" : (state.prefixI[di].has(i) ? "active" : "");
    }

    // AC-i buttons
    for (let i = 0; i < ds.c; i++) {
      const b = document.getElementById("btn-i-" + di + "-" + i);
      b.className = !dsVisible ? "disabled" : (state.autoI[di].has(i) ? "active" : "");
    }
  });

  // Global view buttons
  const anyPrefix = DATA.datasets.some((_, di) => state.prefixI[di].size > 0);
  const anyAC     = DATA.datasets.some((_, di) => state.autoI[di].size > 0);
  document.getElementById("btn-prefix").className  = anyPrefix          ? "active" : "";
  document.getElementById("btn-fc").className      = state.showFullConvo ? "active" : "";
  document.getElementById("btn-lastact").className = state.showLastAct   ? "active" : "";
  document.getElementById("btn-ac").className      = anyAC              ? "active" : "";

  // Show/hide sub-button bars
  document.getElementById("prefix-bar").classList.toggle("hidden", !anyPrefix);
  document.getElementById("ac-bar").classList.toggle("hidden",     !anyAC);
}

// ── Render ─────────────────────────────────────────────────────────────────
function render() {
  const layerNum = DATA.layers[state.layerIdx];
  const key      = "layer" + layerNum + "_" + state.method;
  const traces   = [];

  for (let di = 0; di < DATA.datasets.length; di++) {
    if (!state.visibleDatasets.has(di)) continue;

    const ds      = DATA.datasets[di];
    const proj    = ds.projections[key];
    const proj_fc = ds.projections[key + "_fc"];
    if (!proj) continue;

    const sym = ds.symbol;

    for (const k of state.visibleK[di]) {
      const lab = ds.labels[k];
      const c_k = lab.c_k;

      // ── Prefix traces ────────────────────────────────────────────────
      if (state.prefixI[di].size > 0 && c_k > 0) {
        const enabled = [];
        for (let i = 0; i < c_k; i++) {
          if (!state.prefixI[di].has(i)) continue;
          const pt = proj[k][i][0];
          if (pt[0] === null || isNaN(pt[0])) continue;
          enabled.push(i);
        }
        // Group into runs of consecutive indices so lines don't jump gaps
        const runs = [];
        for (let idx = 0; idx < enabled.length; idx++) {
          if (idx === 0 || enabled[idx] !== enabled[idx - 1] + 1) runs.push([]);
          runs[runs.length - 1].push(enabled[idx]);
        }
        for (const run of runs) {
          const xs = [], ys = [], texts = [], sizes = [];
          for (const i of run) {
            const pt = proj[k][i][0];
            xs.push(pt[0]); ys.push(pt[1]);
            texts.push(lab.prefix_labels[i]);
            sizes.push(i === 0 ? 12 : 8);
          }
          traces.push({
            x: xs, y: ys,
            mode: run.length > 1 ? "lines+markers" : "markers",
            marker: { symbol: sym, size: sizes, color: prefixColor(k) },
            line:   { color: prefixColor(k), width: 1.5 },
            hovertext: texts, hoverinfo: "text",
            showlegend: false,
          });
        }
      }

      // ── Full-convo trace ─────────────────────────────────────────────
      if (state.showFullConvo && proj_fc) {
        const pt = proj_fc[k];
        if (pt && pt[0] !== null && !isNaN(pt[0])) {
          traces.push({
            x: [pt[0]], y: [pt[1]], mode: "markers",
            marker: { symbol: sym, size: 14, color: prefixColor(k),
                      line: { color: "#000", width: 2 } },
            hovertext: [lab.full_convo_label], hoverinfo: "text",
            showlegend: false,
          });
        }
      }

      // ── Last-act trace ───────────────────────────────────────────────
      if (state.showLastAct) {
        const a = lab.c_k - 1;
        if (a >= 0) {
          const pt = proj[k][a][0];
          if (pt && pt[0] !== null && !isNaN(pt[0])) {
            traces.push({
              x: [pt[0]], y: [pt[1]], mode: "markers",
              marker: { symbol: sym, size: 11, color: prefixColor(k),
                        line: { color: "#000", width: 1.5 } },
              hovertext: [lab.last_act_label], hoverinfo: "text",
              showlegend: false,
            });
          }
        }
      }

      // ── Autocomplete traces ──────────────────────────────────────────
      for (const i of state.autoI[di]) {
        if (i >= c_k) continue;
        const eos    = lab.eos_positions[i];
        const ac_len = eos !== null ? eos + 1 : ds.m;
        const xs = [], ys = [], texts = [];
        for (let j = 0; j < ac_len; j++) {
          const pt = proj[k][i][j];
          if (pt[0] === null || isNaN(pt[0])) continue;
          xs.push(pt[0]); ys.push(pt[1]);
          texts.push(lab.ac_labels[i][j]);
        }
        if (xs.length > 0) {
          traces.push({
            x: xs, y: ys, mode: "lines+markers",
            marker: { symbol: sym, size: 6, color: acColor(k) },
            line:   { color: acColor(k), width: 1, dash: "dot" },
            hovertext: texts, hoverinfo: "text",
            showlegend: false,
          });
        }
      }
    }
  }

  const layout = {
    hovermode: "closest",
    xaxis: { title: "Component 1", zeroline: false },
    yaxis: { title: "Component 2", zeroline: false },
    margin: { l: 50, r: 20, t: 10, b: 40 },
  };
  Plotly.react("plot", traces, layout, { responsive: true });
}
</script>
</body>
</html>"""


def generate_union_html(vis_data, output_path):
    title = " + ".join(ds["name"] for ds in vis_data["datasets"])
    data_json = json.dumps(vis_data, ensure_ascii=False)
    page = HTML_TEMPLATE.replace("__DATA_PLACEHOLDER__", data_json)
    page = page.replace("{title}", title)
    with open(output_path, "w") as f:
        f.write(page)


# ── Loading ──────────────────────────────────────────────────────────────────

def load_dataset(dataset_name, model_id):
    """Load JSON + PT for one dataset+model. Returns None if files are missing."""
    data_dir  = "data-" + dataset_name
    fname     = model_id_to_filename(model_id)
    json_path = os.path.join(data_dir, f"{fname}.json")
    pt_path   = os.path.join(data_dir, f"{fname}.pt")

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

    return {
        "name":      dataset_name,
        "json_data": json_data,
        "ac_tensor": pt_data["autocompletion_activations"],
        "fc_tensor": pt_data["full_convo_activations"],
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def process_model_union(model_id, dataset_names, methods):
    dataset_entries = []
    for name in dataset_names:
        entry = load_dataset(name, model_id)
        if entry is not None:
            dataset_entries.append(entry)

    if not dataset_entries:
        print("  [warn] No datasets loaded — skipping model")
        return

    print(f"  Computing joint projections for {len(dataset_entries)} dataset(s) ...")
    vis_data = build_union_vis_data(dataset_entries, methods)

    if not vis_data["methods"]:
        print("  [warn] No projection succeeded — skipping HTML output")
        return

    fname  = model_id_to_filename(model_id)
    suffix = "+".join(d["name"].replace("autoconv", "") for d in dataset_entries)
    out    = f"union_autoconv{suffix}_{fname}.html"
    print(f"  Writing {out} ...")
    generate_union_html(vis_data, out)
    print("  Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Visualise the union of multiple autoconv datasets as interactive HTML.")
    parser.add_argument(
        "--datasets", nargs="+", required=True,
        help="Autoconv dataset names (e.g. autoconv4 autoconv5 autoconv6). "
             "Each name N maps to data-N/ directory.")
    parser.add_argument(
        "--model", type=str, default=None,
        help="HuggingFace model ID. If omitted, all models from models.md are processed.")
    parser.add_argument(
        "--methods", nargs="+", default=["PCA", "UMAP", "Isomap"],
        choices=["PCA", "UMAP", "Isomap"],
        help="Projection methods to compute (default: PCA UMAP Isomap).")
    args = parser.parse_args()

    models = [args.model] if args.model else load_models_list()

    for model_id in models:
        print(f"\n{'=' * 60}")
        print(f"  {model_id}")
        print(f"{'=' * 60}")
        process_model_union(model_id, args.datasets, args.methods)

    print("\nAll done.")


if __name__ == "__main__":
    main()
