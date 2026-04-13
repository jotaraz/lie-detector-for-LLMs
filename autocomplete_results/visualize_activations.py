#!/usr/bin/env python3
"""
Visualize hidden-state activations from extract_activations.py output.

For each conversation a three-level tree is drawn in PCA space:
  • Solid diamond  — prefilled activation (model read full conversation)
  • Solid trajectory (blue = honest conv, red = deceptive conv)
      connects prefix points i = 0 → 9 (light → dark gradient)
  • Dotted generation trajectories branch from each shown prefix point
      (green = honest, orange = deceptive; shade encodes prefix i light→dark;
       marker size encodes generation step small→large)

All activations from the selected layer are pooled to compute the global
mean and the first visdim principal components.

Interaction (via HTML buttons above the plot):
  • Type buttons   — independently show/hide prefilled / prefix / generation
  • Conv buttons   — independently show/hide individual conversations
  These two controls compose: hiding conv 3 then clicking "Show prefilled"
  will NOT restore conv 3's prefilled dot.

autoconv3 convention:
    even indices (0, 2, …) → honest   (blue / green)
    odd  indices (1, 3, …) → deceptive (red / orange)

Usage:
    python visualize_activations.py <pt_file> [--layer {0,1,2}] [--dim {1,2,3}]
    python visualize_activations.py <pt_file> -o my_plot.html

Layer mapping:  0 → L//4   1 → L//2 (default)   2 → 3·L//4
"""

import argparse
import html as _html_mod
import json
import sys
from pathlib import Path

import numpy as np
import torch

try:
    import plotly.graph_objects as go
except ImportError:
    sys.exit("plotly is required.  Install it with:  pip install plotly")


# ── Colour helpers ────────────────────────────────────────────────────────────

def _rgb(arr_01: np.ndarray) -> str:
    r, g, b = (int(255 * float(np.clip(v, 0, 1))) for v in arr_01)
    return f"rgb({r},{g},{b})"


def _gradient(light: np.ndarray, dark: np.ndarray, n: int) -> list[str]:
    if n == 1:
        return [_rgb(dark)]
    return [_rgb(light + k / (n - 1) * (dark - light)) for k in range(n)]


_P_EVEN_L = np.array([174, 199, 232]) / 255
_P_EVEN_D = np.array([ 31, 119, 180]) / 255
_P_ODD_L  = np.array([255, 176, 153]) / 255
_P_ODD_D  = np.array([214,  39,  40]) / 255

_G_EVEN_L = np.array([161, 217, 155]) / 255
_G_EVEN_D = np.array([  0, 109,  44]) / 255
_G_ODD_L  = np.array([253, 141,  60]) / 255   # saturated light orange (was near-yellow)
_G_ODD_D  = np.array([204,  76,   2]) / 255


def prefix_marker_colors(parity: int, n: int) -> list[str]:
    return _gradient(_P_EVEN_L if parity == 0 else _P_ODD_L,
                     _P_EVEN_D if parity == 0 else _P_ODD_D, n)

def prefix_line_color(parity: int) -> str:
    return _rgb((_P_EVEN_D if parity == 0 else _P_ODD_D) * 0.85)

def label_color(parity: int) -> str:
    return _rgb(_P_EVEN_D if parity == 0 else _P_ODD_D)

def gen_branch_color(parity: int, i: int, n_pre: int) -> str:
    light = _G_EVEN_L if parity == 0 else _G_ODD_L
    dark  = _G_EVEN_D if parity == 0 else _G_ODD_D
    t = i / max(n_pre - 1, 1)
    return _rgb(light + t * (dark - light))

def gen_step_sizes(x: int) -> list[float]:
    if x == 1:
        return [6.0]
    return [3.0 + s / (x - 1) * 5.0 for s in range(x)]


# ── PCA ──────────────────────────────────────────────────────────────────────

def compute_pca(all_acts: torch.Tensor, visdim: int):
    mean = all_acts.mean(dim=0)
    centred = (all_acts - mean).to(torch.float32)
    _, S, Vh = torch.linalg.svd(centred, full_matrices=False)
    var_ratio = (S[:visdim] ** 2 / (S ** 2).sum()).numpy()
    print("Explained variance: " +
          ", ".join(f"PC{k+1}={100*v:.1f}%" for k, v in enumerate(var_ratio)))
    return mean, Vh[:visdim]


def project(acts: torch.Tensor, mean: torch.Tensor, comps: torch.Tensor) -> np.ndarray:
    return ((acts.float() - mean) @ comps.T).numpy()


# ── Trace factory ─────────────────────────────────────────────────────────────

def make_trace(visdim: int, proj: np.ndarray, y1d: float = 0.0, **kw):
    # Convert to plain Python lists so fig.to_json() emits standard JSON numbers
    # rather than the plotly-6 binary bdata format, which Plotly.newPlot can't
    # decode when the JSON is embedded as a raw JS object literal.
    if visdim == 3:
        return go.Scatter3d(
            x=proj[:, 0].tolist(), y=proj[:, 1].tolist(), z=proj[:, 2].tolist(), **kw
        )
    elif visdim == 2:
        return go.Scatter(x=proj[:, 0].tolist(), y=proj[:, 1].tolist(), **kw)
    else:
        return go.Scatter(
            x=proj[:, 0].tolist(), y=np.full(len(proj), y1d).tolist(), **kw
        )


# ── HTML template ─────────────────────────────────────────────────────────────

_HTML = """\
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: Arial, sans-serif; background: #fff; }}
    #ctrl {{
      padding: 7px 14px 6px;
      background: #f5f5f5;
      border-bottom: 1px solid #ddd;
      display: flex; gap: 20px; flex-wrap: wrap; align-items: flex-start;
    }}
    .cg {{ display: flex; flex-direction: column; gap: 3px; }}
    .cg > .lbl {{
      font-size: 10px; font-weight: bold; letter-spacing: .07em;
      text-transform: uppercase; color: #888;
    }}
    .brow {{ display: flex; flex-wrap: wrap; gap: 3px; }}
    button {{
      padding: 2px 9px; font-size: 12px; border-radius: 3px;
      cursor: pointer; border: 1.5px solid currentColor;
      background: transparent; transition: background .1s, color .1s;
    }}
    button.on  {{ background: currentColor; }}
    button.on > span  {{ color: #fff; }}
    button > span {{ color: inherit; }}
    .bn {{ color: #555; }}
    .bh {{ color: #1a5fa8; }}
    .bd {{ color: #bb3020; }}
    #plot {{ width: 100%; height: calc(100vh - {ctrl_h}px); }}
    #colorkey {{
      padding: 4px 14px;
      font-size: 11px; color: #555;
      border-top: 1px solid #eee;
      background: #fafafa;
    }}
  </style>
</head>
<body>
<div id="ctrl">
  <div class="cg">
    <span class="lbl">Layer</span>
    <div class="brow">
      <button id="btn-l0" class="bn"    onclick="switchLayer(0)"><span>L//4</span></button>
      <button id="btn-l1" class="bn on" onclick="switchLayer(1)"><span>L//2</span></button>
      <button id="btn-l2" class="bn"    onclick="switchLayer(2)"><span>3L//4</span></button>
    </div>
  </div>
  <div class="cg">
    <span class="lbl">Type</span>
    <div class="brow">
      <button id="btn-pf"  class="bn on" onclick="toggleType('pf')"><span>Prefilled</span></button>
      <button id="btn-pr"  class="bn on" onclick="toggleType('pr')"><span>Prefix</span></button>
      <button id="btn-gen" class="bn on" onclick="toggleType('gen')"><span>Generation</span></button>
    </div>
  </div>
  <div class="cg">
    <span class="lbl">Conversations</span>
    <div class="brow">
      <button class="bn on" onclick="setAll(true)"><span>Show all</span></button>
      <button class="bn"    onclick="setAll(false)"><span>Hide all</span></button>
    </div>
    <div class="brow">{conv_btns}</div>
  </div>
  <div class="cg">
    <span class="lbl">Gen prefix i</span>
    <div class="brow">
      <button class="bn on" onclick="setAllGen(true)"><span>Show all</span></button>
      <button class="bn"    onclick="setAllGen(false)"><span>Hide all</span></button>
    </div>
    <div class="brow">{gen_btns}</div>
  </div>
</div>
<div id="plot"></div>
<div id="colorkey">
  ◆ diamond&nbsp;=&nbsp;prefilled &nbsp;|&nbsp;
  ● solid line&nbsp;=&nbsp;prefix (light→dark: i&nbsp;0→9) &nbsp;|&nbsp;
  ● dotted line&nbsp;=&nbsp;generation branch (shade: prefix&nbsp;i, size: gen&nbsp;step)
  &nbsp;&nbsp;—&nbsp;&nbsp;
  <span style="color:#1a5fa8">■ blue = honest</span> &nbsp;
  <span style="color:#bb3020">■ red = deceptive</span>
</div>

<script>
// One figure per layer (0=L//4, 1=L//2, 2=3L//4); all share the same trace structure.
const LAYERS = [{layer0_json}, {layer1_json}, {layer2_json}];
let currentLayer = 1;
Plotly.newPlot('plot', LAYERS[1].data, LAYERS[1].layout, {{responsive: true}});

function switchLayer(lid) {{
  if (lid === currentLayer) return;
  currentLayer = lid;
  [0,1,2].forEach(i => document.getElementById('btn-l'+i).classList.toggle('on', i===lid));
  const fd = LAYERS[lid];
  Plotly.react('plot', fd.data, fd.layout).then(() => applyVis());
}}

// TM[c] = {{pf: idx, pr: idx, gen: {{"0": idx, "1": idx, ...}}}}
const TM  = {tm_json};
const N   = {N};
const NPR = {NPR};   // number of prefix positions (gen branches per conv)
const NT  = {NT};

// Three independent toggle dimensions that compose multiplicatively
const CS = Object.fromEntries([...Array(N).keys()].map(i => [i, true]));
const TS = {{pf: true, pr: true, gen: true}};
const GS = Object.fromEntries([...Array(NPR).keys()].map(i => [i, true]));

function computeVis() {{
  const v = new Array(NT).fill(false);
  for (let c = 0; c < N; c++) {{
    if (!CS[c]) continue;
    const t = TM[c];
    if (TS.pf) v[t.pf] = true;
    if (TS.pr) v[t.pr] = true;
    if (TS.gen) {{
      for (const [iStr, idx] of Object.entries(t.gen)) {{
        if (GS[iStr]) v[idx] = true;
      }}
    }}
  }}
  return v;
}}

function applyVis() {{
  Plotly.restyle('plot', {{visible: computeVis()}});
}}

function toggleType(k) {{
  TS[k] = !TS[k];
  document.getElementById('btn-' + k).classList.toggle('on', TS[k]);
  applyVis();
}}

function toggleConv(c) {{
  CS[c] = !CS[c];
  document.getElementById('btn-c' + c).classList.toggle('on', CS[c]);
  applyVis();
}}

function setAll(show) {{
  for (let c = 0; c < N; c++) {{
    CS[c] = show;
    document.getElementById('btn-c' + c).classList.toggle('on', show);
  }}
  applyVis();
}}

function toggleGen(i) {{
  GS[i] = !GS[i];
  document.getElementById('btn-gi' + i).classList.toggle('on', GS[i]);
  applyVis();
}}

function setAllGen(show) {{
  for (let i = 0; i < NPR; i++) {{
    GS[i] = show;
    document.getElementById('btn-gi' + i).classList.toggle('on', show);
  }}
  applyVis();
}}
</script>
</body>
</html>
"""


# ── JSONL completion loader ───────────────────────────────────────────────────

def _esc(s: str) -> str:
    """Escape a string for safe embedding in HTML hover text."""
    return _html_mod.escape(s, quote=False)


def load_jsonl_completions(
    pt_path: Path, n_conversations: int, n_prefix: int
) -> tuple[list[list[str]], list[list[str]]] | tuple[None, None]:
    """Find the matching JSONL file and return completion text and next tokens per (conv, prefix).

    The .pt file is named  <tag>_activations_<model_tag>.pt;
    the JSONL is named     <tag>_<model_tag>.jsonl  (same directory).

    Returns a tuple (next_texts, next_tokens) where:
      next_texts[c][i]  = the full completion string the model generated
                          when given the first i ground-truth assistant tokens
      next_tokens[c][i] = the single token that was about to be generated at
                          prefix position i (derived from consecutive assistant
                          prefixes: prefix[i+1][len(prefix[i]):])
    Returns (None, None) if the file cannot be found.
    """
    stem = pt_path.stem  # e.g. "autoconv3_activations_meta-llama_Llama-3.1-8B-Instruct"
    jsonl_stem = stem.replace("_activations_", "_", 1)
    jsonl_path = pt_path.parent / f"{jsonl_stem}.jsonl"
    if not jsonl_path.exists():
        print(
            f"  No matching JSONL at {jsonl_path.name} — hover text will lack token info.",
            flush=True,
        )
        return None, None

    print(f"  Loading completion data from {jsonl_path.name} …", flush=True)
    entries = []
    with jsonl_path.open() as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    if len(entries) < n_conversations:
        print(
            f"  Warning: JSONL has {len(entries)} entries, expected {n_conversations}.",
            flush=True,
        )
        return None, None

    next_texts:  list[list[str]] = []
    next_tokens: list[list[str]] = []
    for entry in entries[:n_conversations]:
        acs = entry.get("autocompletes", [])
        # Index autocompletes by tokens_provided for O(1) lookup
        ac_by_i: dict[int, dict] = {}
        for ac in acs:
            i = ac.get("tokens_provided", -1)
            if 0 <= i < n_prefix:
                ac_by_i[i] = ac

        conv_texts  = [""] * n_prefix
        conv_tokens = [""] * n_prefix
        for i in range(n_prefix):
            ac = ac_by_i.get(i)
            if ac is None:
                continue
            conv_texts[i] = ac.get("completion", "")
            # Next token = text added between prefix i and prefix i+1
            prefix_i   = ac.get("assistant_prefix", "")
            ac_next    = ac_by_i.get(i + 1)
            if ac_next is not None:
                prefix_i1      = ac_next.get("assistant_prefix", "")
                conv_tokens[i] = prefix_i1[len(prefix_i):]
            else:
                # Last prefix: no i+1 exists, so we cannot derive the next token.
                conv_tokens[i] = ""

        next_texts.append(conv_texts)
        next_tokens.append(conv_tokens)

    return next_texts, next_tokens


# ── Hover-text formatters ─────────────────────────────────────────────────────

def _fmt_prefix_hover(next_tok: str, rest_text: str, snip: int) -> str:
    """Single merged line: bold(next_tok) + rest_text (truncated).

    'rest_text' should be next_texts[c][i+1] — the model's completion
    when given i+1 ground-truth tokens, so it is never contaminated by
    the string-stripping divergence bug.
    """
    rest = rest_text.rstrip()
    if not next_tok:
        # Last prefix position — next token unknown; show rest unbolded.
        snippet = rest[:snip]
        return "<br>" + _esc(snippet) + ("…" if len(rest) > snip else "")
    snippet = rest[:snip]
    return (
        "<br>"
        + f"<b>{_esc(next_tok)}</b>"
        + _esc(snippet)
        + ("…" if len(rest) > snip else "")
    )


def _fmt_gen_hover(
    anchor_tok: str,
    next_tok: str | None,
    rest_text: str,
    snip: int,
    have_data: bool,
) -> str:
    """Single merged line: (anchor_tok) bold(next_tok) rest_text."""
    if not have_data:
        return ""
    rest = rest_text.rstrip()
    anchor_part = f"({_esc(anchor_tok)}) " if anchor_tok else ""
    if next_tok is None:
        return "<br>" + anchor_part + "[beyond tracked prefix]"
    if not next_tok:
        # next token unknown (last prefix position)
        snippet = rest[:snip]
        return (
            "<br>"
            + anchor_part
            + _esc(snippet)
            + ("…" if len(rest) > snip else "")
        )
    snippet = rest[:snip]
    return (
        "<br>"
        + anchor_part
        + f"<b>{_esc(next_tok)}</b>"
        + _esc(snippet)
        + ("…" if len(rest) > snip else "")
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("pt_file", help="Path to activations .pt file")
    parser.add_argument(
        "--dim", type=int, default=2, choices=[1, 2, 3],
        help="PCA visualisation dimension (default: 2)",
    )
    parser.add_argument(
        "--output", "-o", default=None, metavar="FILE.html",
        help="Output HTML file (default: plot_<pt_file_stem>.html)",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = f"plot_{Path(args.pt_file).stem}.html"

    print(f"Loading {args.pt_file} …", flush=True)
    data = torch.load(args.pt_file, weights_only=True)

    visdim = args.dim

    # Shape constants (layer-independent)
    N     = data["prefilled"].shape[0]
    D     = data["prefilled"].shape[2]
    n_pre = data["prefix"].shape[1]
    x     = data["generation"].shape[2]
    model_short = data["model_id"].split("/")[-1]
    pc_str      = " × ".join(f"PC{k+1}" for k in range(visdim))

    print(f"N={N}  D={D}  n_prefix={n_pre}  n_gen_steps={x}  "
          f"model={model_short}", flush=True)

    # ── Load per-(conv, prefix) completion strings and next tokens from JSONL ──
    next_texts, next_tokens = load_jsonl_completions(Path(args.pt_file), N, n_pre)
    _SNIP = 70  # max chars shown in hover

    step_sizes = gen_step_sizes(x)
    y1d_lanes  = np.linspace(-0.45, 0.45, N)

    # ── Build traces + figure for each layer ──────────────────────────────────
    # trace_map is identical across layers (same structure), so we capture it
    # from the first layer and reuse it.
    layer_figs: list = []
    trace_map:  dict = {}

    for layer_id in range(3):
        pf  = data["prefilled"][:, layer_id, :].float()          # [N, D]
        pr  = data["prefix"][:, :, layer_id, :].float()          # [N, n_pre, D]
        gen = data["generation"][:, :, :, layer_id, :].float()   # [N, n_pre, x, D]

        layer_abs = data["layers"][layer_id]
        print(f"Layer {layer_id} → hidden_states[{layer_abs}]", end="  ", flush=True)

        all_acts = torch.cat([pf, pr.reshape(-1, D), gen.reshape(-1, D)], dim=0)
        mean, comps = compute_pca(all_acts, visdim)

        pf_proj  = project(pf, mean, comps)
        pr_proj  = project(pr.reshape(-1, D), mean, comps).reshape(N, n_pre, visdim)
        gen_proj = project(gen.reshape(-1, D), mean, comps).reshape(N, n_pre, x, visdim)

        traces: list = []
        tm: dict = {}

        for c in range(N):
            parity = c % 2
            label  = "Honest" if parity == 0 else "Deceptive"
            y1d    = y1d_lanes[c]
            lc     = label_color(parity)

            p_mcols = prefix_marker_colors(parity, n_pre)
            p_lcol  = prefix_line_color(parity)

            tm[c] = {"pf": None, "pr": None, "gen": {}}

            # Prefilled (diamond)
            tm[c]["pf"] = len(traces)
            traces.append(make_trace(
                visdim, pf_proj[c:c+1], y1d,
                mode="markers", showlegend=False,
                marker=dict(color=lc, size=11, symbol="diamond",
                            line=dict(width=1.2, color="rgba(0,0,0,0.5)")),
                hovertext=[f"<b>Conv {c}</b> — Prefilled ({label})"],
                hoverinfo="text",
            ))

            # Prefix trajectory
            tm[c]["pr"] = len(traces)
            traces.append(make_trace(
                visdim, pr_proj[c], y1d,
                mode="lines+markers", showlegend=False,
                marker=dict(color=p_mcols, size=[11] + [7] * (n_pre - 1),
                            symbol=["star"] + ["circle"] * (n_pre - 1),
                            line=dict(width=0.5, color="rgba(255,255,255,0.6)")),
                line=dict(color=p_lcol, width=3.5),
                hovertext=[
                    f"<b>Conv {c}</b> — Prefix i={i} ({label})"
                    + (
                        _fmt_prefix_hover(
                            next_tokens[c][i],
                            next_texts[c][i + 1] if i + 1 < n_pre else "",
                            _SNIP,
                        )
                        if next_texts else ""
                    )
                    for i in range(n_pre)
                ],
                hoverinfo="text",
            ))

            # Generation branches (prepend prefix anchor for visual link)
            for i in range(n_pre):
                col    = gen_branch_color(parity, i, n_pre)
                anchor = pr_proj[c, i:i+1]
                branch = np.vstack([anchor, gen_proj[c, i]])
                tm[c]["gen"][i] = len(traces)
                traces.append(make_trace(
                    visdim, branch, y1d,
                    mode="lines+markers", showlegend=False,
                    marker=dict(color=col, size=[0.0] + step_sizes, symbol="circle",
                                line=dict(width=0, color="rgba(0,0,0,0)")),
                    line=dict(color=col, width=2.0, dash="dot"),
                    hovertext=[""] + [
                        f"<b>Conv {c}</b> — Gen prefix i={i} step={s} ({label})"
                        + _fmt_gen_hover(
                            anchor_tok=next_tokens[c][i],
                            next_tok=next_tokens[c][i + s] if i + s < n_pre else None,
                            rest_text=next_texts[c][i + s + 1] if i + s + 1 < n_pre else "",
                            snip=_SNIP,
                            have_data=bool(next_texts),
                        )
                        for s in range(x)
                    ],
                    hoverinfo="text",
                ))

        if layer_id == 0:
            trace_map = tm   # identical structure for all layers; capture once

        title = (f"Activation PCA — {model_short} | "
                 f"Layer {layer_id} (hs[{layer_abs}]) | {pc_str}")

        if visdim == 3:
            layout = go.Layout(title=title, showlegend=False,
                               scene=dict(xaxis_title="PC1", yaxis_title="PC2",
                                          zaxis_title="PC3"))
        elif visdim == 2:
            layout = go.Layout(title=title, showlegend=False,
                               xaxis_title="PC1", yaxis_title="PC2")
        else:
            layout = go.Layout(
                title=title, showlegend=False, xaxis_title="PC1",
                yaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
                           range=[-0.6, 0.6], title="conversations (lanes)"))

        layer_figs.append(go.Figure(data=traces, layout=layout))

    # ── Conversation buttons ──────────────────────────────────────────────────
    conv_btns = " ".join(
        f'<button id="btn-c{c}" class="{"bh" if c % 2 == 0 else "bd"} on" '
        f'onclick="toggleConv({c})"><span>C{c}</span></button>'
        for c in range(N)
    )

    # ── Gen-prefix-i buttons ──────────────────────────────────────────────────
    gen_btns = " ".join(
        f'<button id="btn-gi{i}" class="bn on" '
        f'onclick="toggleGen({i})"><span>i={i}</span></button>'
        for i in range(n_pre)
    )

    ctrl_h = 95

    html = _HTML.format(
        title=f"Activation PCA — {model_short} | {pc_str}",
        ctrl_h=ctrl_h,
        conv_btns=conv_btns,
        gen_btns=gen_btns,
        layer0_json=layer_figs[0].to_json(),
        layer1_json=layer_figs[1].to_json(),
        layer2_json=layer_figs[2].to_json(),
        tm_json=json.dumps(trace_map),
        N=N,
        NPR=n_pre,
        NT=len(layer_figs[0].data),
    )

    Path(args.output).write_text(html, encoding="utf-8")
    print(f"Saved → {args.output}  (open in a browser)", flush=True)


if __name__ == "__main__":
    main()
