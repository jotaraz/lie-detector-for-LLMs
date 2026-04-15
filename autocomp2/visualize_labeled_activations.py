#!/usr/bin/env python3
"""Visualize autocompletion activations as an interactive HTML page.

Reads the .pt activation tensor and the .json text data produced by
generate_text_and_activations.py, projects activations to 2D via PCA / UMAP /
Isomap (one layer at a time), and writes a self-contained HTML file with
Plotly.js for interactive exploration.
"""

import argparse
import html as html_lib
import json
import os
import numpy as np
import torch

from sklearn.decomposition import PCA
from sklearn.manifold import Isomap

# ── Dimensionality-reduction hyperparameters (adjust as needed) ──────────────
ISOMAP_N_NEIGHBORS = 10   # Isomap: k for the neighbourhood graph
# UMAP (only used if umap-learn is installed):
UMAP_N_NEIGHBORS = 15     # UMAP: size of local neighbourhood
UMAP_MIN_DIST = 0.1       # UMAP: minimum distance between embedded points
UMAP_METRIC = "euclidean"

MODELS_FILE = "models.md"
DATA_DIR = "data"


def load_models_list():
    with open(MODELS_FILE) as f:
        return [line.strip() for line in f if line.strip()]


def model_id_to_filename(model_id):
    return model_id.replace("/", "--")


# ── Projection helpers ──────────────────────────────────────────────────────

def _try_import_umap():
    try:
        import umap
        return umap
    except ImportError:
        return None


def project_all(acts_flat, valid_mask, methods):
    """Project *acts_flat* (n_points, d) to 2-D with each method in *methods*.

    Only rows where *valid_mask* is True are used for fitting.  Invalid rows
    get NaN coordinates (they are never plotted).

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


# ── Label construction ──────────────────────────────────────────────────────

def _esc(s):
    return html_lib.escape(s)


def _bold_color(k):
    """Pink for even-k (blue/green traces), lightgreen for odd-k (red/orange)."""
    return "#e91e8a" if k % 2 == 0 else "lightgreen"


def make_prefix_label(ref_tokens, i, c_k, k):
    """Hover label for prefix-activation_{k,i}: show first c_k reference
    tokens with token i in bold."""
    color = _bold_color(k)
    parts = []
    for idx in range(c_k):
        tok = _esc(ref_tokens[idx])
        if idx == i:
            parts.append(f"<b style=\"color:{color}\">{tok}</b>")
        else:
            parts.append(tok)
    return "".join(parts)


def make_full_convo_label(ref_tokens):
    """Hover label for the 'full convo' point: show all reference tokens."""
    return "".join(_esc(t) for t in ref_tokens)


def make_autocomplete_label(ref_tokens, ac_tokens, i, j, k):
    """Hover label for autocompletion-activation_{k,i,j}:
    R^{(k)}_i underlined, full AC shown, token ac_{i,j} bold."""
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
            parts.append(f"<b style=\"color:{color}\">{tok_esc}</b>")
        else:
            parts.append(tok_esc)
    return "".join(parts)


# ── Build the data payload for the HTML ─────────────────────────────────────

def build_vis_data(json_data, ac_tensor, fc_tensor, layer_indices, methods):
    """Return a JSON-serialisable dict consumed by the HTML/JS visualisation.

    ac_tensor: autocompletion activations, shape (N, c, m, L, d)
    fc_tensor: full-convo activations, shape (N, L, d)
    """

    N = len(json_data["conversations"])
    c = json_data["c"]
    m = json_data["m"]
    L = len(layer_indices)

    # Pre-build labels (independent of layer/method) ─────────────────────
    label_data = []  # per conversation
    for conv in json_data["conversations"]:
        k = conv["k"]
        n_k = conv["n"]
        c_k = min(c, n_k)
        ref_tokens = conv["reference_tokens"]

        prefix_labels = []
        for i in range(c_k):
            prefix_labels.append(make_prefix_label(ref_tokens, i, c_k, k))

        fc_label = make_full_convo_label(ref_tokens)

        ac_labels = []  # [i][j]
        for ac in conv["autocompletions"]:
            i = ac["i"]
            row = []
            for j in range(m):
                if ac["tokens"][j] == "":
                    row.append("")
                else:
                    row.append(make_autocomplete_label(
                        ref_tokens, ac["tokens"], i, j, k))
            ac_labels.append(row)

        label_data.append({
            "k": k,
            "c_k": c_k,
            "n_k": n_k,
            "prefix_labels": prefix_labels,
            "full_convo_label": fc_label,
            "ac_labels": ac_labels,
            "eos_positions": [ac.get("eos_position") for ac in conv["autocompletions"]],
        })

    # Projections per (layer, method) ────────────────────────────────────
    projections = {}
    d = ac_tensor.shape[-1]
    for l_idx, layer in enumerate(layer_indices):
        # Autocompletion activations: (N*c*m, d)
        ac_flat = ac_tensor[:, :, :, l_idx, :].reshape(-1, d)
        ac_flat_np = ac_flat.float().numpy()
        n_ac = len(ac_flat_np)

        # Full-convo activations: (N, d)
        fc_flat = fc_tensor[:, l_idx, :]
        fc_flat_np = fc_flat.float().numpy()

        # Combine for joint projection
        all_acts_np = np.concatenate([ac_flat_np, fc_flat_np], axis=0)
        valid = np.any(all_acts_np != 0, axis=1)

        if valid.sum() < 3:
            print(f"  [warn] Layer {layer}: fewer than 3 valid activations, skipping")
            continue

        proj_results = project_all(all_acts_np, valid, methods)

        for method_name, coords in proj_results.items():
            key = f"layer{layer}_{method_name}"
            # Split back: autocompletion coords → (N, c, m, 2)
            ac_coords = coords[:n_ac].reshape(N, c, m, 2)
            # Full-convo coords → (N, 2)
            fc_coords = coords[n_ac:]
            projections[key] = ac_coords.tolist()
            projections[key + "_fc"] = fc_coords.tolist()

    return {
        "model": json_data["model"],
        "layers": layer_indices,
        "methods": [meth for meth in methods if any(
            f"layer{l}_{meth}" in projections for l in layer_indices)],
        "c": c,
        "m": m,
        "N": N,
        "labels": label_data,
        "projections": projections,
    }


# ── HTML generation ─────────────────────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Activation Visualisation – {model_name}</title>
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
  button { font-size: 0.75rem; padding: 3px 8px; border: 1px solid #aaa;
           border-radius: 4px; background: #fff; cursor: pointer; }
  button.active { background: #4a90d9; color: #fff; border-color: #4a90d9; }
  button.active-red { background: #d94a4a; color: #fff; border-color: #d94a4a; }
  select { font-size: 0.78rem; padding: 2px 4px; }
  #plot { width: 100%; height: calc(100vh - 210px); }
</style>
</head>
<body>

<h1>Activation Visualisation &ndash; <span id="model-name"></span></h1>

<!-- Row 1: layer & method -->
<div class="controls">
  <div class="group">
    <label>Layer:</label><select id="sel-layer"></select>
  </div>
  <div class="group">
    <label>Method:</label><select id="sel-method"></select>
  </div>
</div>

<!-- Row 2: conversations -->
<div class="controls" id="conv-bar">
  <div class="group">
    <label>Conversations:</label>
    <button onclick="hideAll()">hide all</button>
    <button onclick="showAll()">show all</button>
  </div>
  <div class="group" id="conv-buttons"></div>
</div>

<!-- Row 3: views -->
<div class="controls">
  <div class="group">
    <label>View:</label>
    <button id="btn-prefix" onclick="togglePrefix()">prefix</button>
    <button id="btn-fc" onclick="toggleView('fullConvo')">full convo</button>
    <button id="btn-ac" onclick="toggleAC()">autocomplete</button>
  </div>
</div>

<!-- Row 4: prefix sub-buttons -->
<div class="controls" id="prefix-bar">
  <div class="group">
    <label>Prefix i:</label>
    <button onclick="showAllPrefixI()">show all</button>
    <button onclick="hideAllPrefixI()">hide all</button>
  </div>
  <div class="group" id="prefix-buttons"></div>
</div>

<!-- Row 5: autocomplete sub-buttons -->
<div class="controls" id="ac-bar">
  <div class="group">
    <label>AC i:</label>
    <button onclick="showAllACI()">show all</button>
    <button onclick="hideAllACI()">hide all</button>
  </div>
  <div class="group" id="ac-buttons"></div>
</div>

<div id="plot"></div>

<script>
// ── Embedded data ──────────────────────────────────────────────────────────
const DATA = __DATA_PLACEHOLDER__;

// ── State ──────────────────────────────────────────────────────────────────
const state = {
  layerIdx: 0,
  method: DATA.methods[0],
  visibleK: new Set(Array.from({length: DATA.N}, (_, i) => i)),
  prefixI: new Set(Array.from({length: DATA.c}, (_, i) => i)),
  showFullConvo: false,
  autoI: new Set(),
};

// ── Colours ────────────────────────────────────────────────────────────────
function prefixColor(k) { return k % 2 === 0 ? "#2563eb" : "#dc2626"; }
function acColor(k)     { return k % 2 === 0 ? "#16a34a" : "#ea580c"; }

// ── Init controls ──────────────────────────────────────────────────────────
(function init() {
  document.getElementById("model-name").textContent = DATA.model;

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
  selM.onchange = () => { state.method = selM.value; render(); };

  const cb = document.getElementById("conv-buttons");
  for (let k = 0; k < DATA.N; k++) {
    const b = document.createElement("button");
    b.textContent = "k=" + k;
    b.id = "btn-k-" + k;
    b.className = "active";
    b.onclick = () => toggleK(k);
    cb.appendChild(b);
  }

  const pb = document.getElementById("prefix-buttons");
  for (let i = 0; i < DATA.c; i++) {
    const b = document.createElement("button");
    b.textContent = "i=" + i;
    b.id = "btn-pi-" + i;
    b.onclick = () => togglePrefixI(i);
    pb.appendChild(b);
  }

  const ab = document.getElementById("ac-buttons");
  for (let i = 0; i < DATA.c; i++) {
    const b = document.createElement("button");
    b.textContent = "i=" + i;
    b.id = "btn-i-" + i;
    b.onclick = () => toggleI(i);
    ab.appendChild(b);
  }

  syncButtons();
  render();
})();

// ── Button logic ───────────────────────────────────────────────────────────
function hideAll() {
  state.visibleK.clear(); syncButtons(); render();
}
function showAll() {
  for (let k = 0; k < DATA.N; k++) state.visibleK.add(k);
  syncButtons(); render();
}
function toggleK(k) {
  if (state.visibleK.has(k)) state.visibleK.delete(k);
  else state.visibleK.add(k);
  syncButtons(); render();
}
function toggleView(v) {
  if (v === "fullConvo") state.showFullConvo = !state.showFullConvo;
  syncButtons(); render();
}
function togglePrefix() {
  if (state.prefixI.size > 0) state.prefixI.clear();
  else for (let i = 0; i < DATA.c; i++) state.prefixI.add(i);
  syncButtons(); render();
}
function togglePrefixI(i) {
  if (state.prefixI.has(i)) state.prefixI.delete(i);
  else state.prefixI.add(i);
  syncButtons(); render();
}
function showAllPrefixI() {
  for (let i = 0; i < DATA.c; i++) state.prefixI.add(i);
  syncButtons(); render();
}
function hideAllPrefixI() {
  state.prefixI.clear();
  syncButtons(); render();
}
function toggleAC() {
  if (state.autoI.size > 0) state.autoI.clear();
  else for (let i = 0; i < DATA.c; i++) state.autoI.add(i);
  syncButtons(); render();
}
function toggleI(i) {
  if (state.autoI.has(i)) state.autoI.delete(i);
  else state.autoI.add(i);
  syncButtons(); render();
}
function showAllACI() {
  for (let i = 0; i < DATA.c; i++) state.autoI.add(i);
  syncButtons(); render();
}
function hideAllACI() {
  state.autoI.clear();
  syncButtons(); render();
}
function syncButtons() {
  for (let k = 0; k < DATA.N; k++) {
    const b = document.getElementById("btn-k-" + k);
    b.className = state.visibleK.has(k) ? "active" : "";
  }
  document.getElementById("btn-prefix").className =
      state.prefixI.size > 0 ? "active" : "";
  document.getElementById("btn-fc").className =
      state.showFullConvo ? "active" : "";
  document.getElementById("btn-ac").className =
      state.autoI.size > 0 ? "active" : "";
  for (let i = 0; i < DATA.c; i++) {
    const b = document.getElementById("btn-pi-" + i);
    b.className = state.prefixI.has(i) ? "active" : "";
  }
  for (let i = 0; i < DATA.c; i++) {
    const b = document.getElementById("btn-i-" + i);
    b.className = state.autoI.has(i) ? "active" : "";
  }
}

// ── Render ──────────────────────────────────────────────────────────────────
function render() {
  const key = "layer" + DATA.layers[state.layerIdx] + "_" + state.method;
  const proj = DATA.projections[key];
  const proj_fc = DATA.projections[key + "_fc"];
  if (!proj) { Plotly.purge("plot"); return; }

  const traces = [];

  for (const k of state.visibleK) {
    const lab = DATA.labels[k];
    const c_k = lab.c_k;

    // ── Prefix traces ──────────────────────────────────────────────────
    // Build consecutive runs of enabled prefix indices, then emit one
    // trace per run (so gaps between non-adjacent enabled i's have no line).
    if (state.prefixI.size > 0 && c_k > 0) {
      const enabled = [];
      for (let i = 0; i < c_k; i++) {
        if (!state.prefixI.has(i)) continue;
        const pt = proj[k][i][0];  // j=0
        if (pt[0] === null || isNaN(pt[0])) continue;
        enabled.push(i);
      }
      // Split into runs of consecutive indices
      const runs = [];
      for (let idx = 0; idx < enabled.length; idx++) {
        if (idx === 0 || enabled[idx] !== enabled[idx - 1] + 1) {
          runs.push([]);
        }
        runs[runs.length - 1].push(enabled[idx]);
      }
      for (const run of runs) {
        const xs = [], ys = [], texts = [], symbols = [], sizes = [];
        for (const i of run) {
          const pt = proj[k][i][0];
          xs.push(pt[0]); ys.push(pt[1]);
          texts.push(lab.prefix_labels[i]);
          symbols.push(i === 0 ? "star" : "circle");
          sizes.push(i === 0 ? 12 : 8);
        }
        const mode = run.length > 1 ? "lines+markers" : "markers";
        traces.push({
          x: xs, y: ys, mode: mode,
          marker: { symbol: symbols, size: sizes, color: prefixColor(k) },
          line: { color: prefixColor(k), width: 1.5 },
          hovertext: texts, hoverinfo: "text",
          showlegend: false,
        });
      }
    }

    // ── Full-convo trace ───────────────────────────────────────────────
    if (state.showFullConvo && proj_fc) {
      const pt = proj_fc[k];
      if (pt && pt[0] !== null && !isNaN(pt[0])) {
        traces.push({
          x: [pt[0]], y: [pt[1]], mode: "markers",
          marker: { symbol: "square", size: 11, color: prefixColor(k),
                    line: { color: "#000", width: 1 } },
          hovertext: [lab.full_convo_label], hoverinfo: "text",
          showlegend: false,
        });
      }
    }

    // ── Autocomplete traces ────────────────────────────────────────────
    for (const i of state.autoI) {
      if (i >= c_k) continue;
      const eos = lab.eos_positions[i];
      const ac_len = eos !== null ? eos + 1 : DATA.m;
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
          marker: { size: 6, color: acColor(k) },
          line: { color: acColor(k), width: 1, dash: "dot" },
          hovertext: texts, hoverinfo: "text",
          showlegend: false,
        });
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


def generate_html(vis_data, output_path):
    data_json = json.dumps(vis_data, ensure_ascii=False)
    page = HTML_TEMPLATE.replace("__DATA_PLACEHOLDER__", data_json)
    page = page.replace("{model_name}", vis_data["model"])
    with open(output_path, "w") as f:
        f.write(page)


# ── Main ────────────────────────────────────────────────────────────────────

def process_model(model_id):
    fname = model_id_to_filename(model_id)
    json_path = os.path.join(DATA_DIR, f"{fname}.json")
    pt_path = os.path.join(DATA_DIR, f"{fname}.pt")
    html_path = os.path.join(DATA_DIR, f"{fname}.html")

    print(f"Loading data for {model_id} ...")
    with open(json_path) as f:
        json_data = json.load(f)
    pt_data = torch.load(pt_path, map_location="cpu", weights_only=True)
    ac_tensor = pt_data["autocompletion_activations"]
    fc_tensor = pt_data["full_convo_activations"]

    layer_indices = json_data["layers"]
    methods = ["PCA", "UMAP", "Isomap"]

    print("  Computing projections ...")
    vis_data = build_vis_data(json_data, ac_tensor, fc_tensor, layer_indices, methods)

    print(f"  Writing {html_path} ...")
    generate_html(vis_data, html_path)
    print("  Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Visualise autocompletion activations as interactive HTML.")
    parser.add_argument("--model", type=str, default=None,
                        help="HuggingFace model ID. If omitted, process all "
                             "models from models.md.")
    args = parser.parse_args()

    models = [args.model] if args.model else load_models_list()

    for model_id in models:
        print(f"\n{'=' * 60}")
        print(f"  {model_id}")
        print(f"{'=' * 60}")
        process_model(model_id)

    print("\nAll done.")


if __name__ == "__main__":
    main()
