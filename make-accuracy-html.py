#!/usr/bin/env python3
"""Render the interactive accuracy.html viewer from accuracy.csv.

Reads the control-calibrated balanced-accuracy table written by
`compute-accuracy.py` and emits a self-contained HTML page: one panel per
model, each a sortable/filterable table of (layer, probe, control, eval) rows.
Pure stdlib; reads only the CSV, so it runs anywhere the results live.

The page layout (CSS + toolbar + JS) is embedded verbatim below; only the
per-model panels are generated from the CSV. Rows are grouped by (layer, probe)
into the `data-grp` units the JS sorts as a block; the first row of each group
carries `data-ord` (original order). The thick block-divider line is drawn at
runtime on each block's first *visible* row (`grp-top`), so it survives both
filtering and re-sorting.

Usage:
    python make-accuracy-html.py [--csv results/probe-comparison/accuracy.csv]
                                 [--out autocomp2/accuracy.html]
"""

import argparse
import csv
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = SCRIPT_DIR / "results" / "probe-comparison" / "accuracy.csv"
DEFAULT_OUT = SCRIPT_DIR / "autocomp2" / "accuracy.html"

# --- short-name mappings (must match the toolbar/CSS class names below) ------

# accuracy.csv `probe` -> probe-column display label (combined probes keep "+").
# The data-combo attribute is this label with "+" -> "_" (matches the CSS class
# names like hide-combo-unprompted_prompted).
PROBE_DISPLAY = {
    "unprompted_roleplay": "unprompted",
    "prompted_roleplay": "prompted",
    "company_assistant": "company",
    "unprompted+prompted": "unprompted+prompted",
    "unprompted+company": "unprompted+company",
    "prompted+company": "prompted+company",
    "all3": "all3",
}

# accuracy.csv `probe` -> short probe-column display label
PROBE_SHORT = {
    "unprompted_roleplay": "u",
    "prompted_roleplay": "p",
    "company_assistant": "c",
    "unprompted+prompted": "u+p",
    "unprompted+company": "u+c",
    "prompted+company": "p+c",
    "all3": "all3",
}

# datasets that count as a probe's own ("self") evaluation -> data-isself
PROBE_TRAINSETS = {
    "unprompted_roleplay": ["unprompted_roleplay"],
    "prompted_roleplay": ["prompted_roleplay"],
    "company_assistant": ["company_assistant"],
    "unprompted+prompted": ["unprompted_roleplay", "prompted_roleplay"],
    "unprompted+company": ["unprompted_roleplay", "company_assistant"],
    "prompted+company": ["prompted_roleplay", "company_assistant"],
    "all3": ["unprompted_roleplay", "prompted_roleplay", "company_assistant"],
}

# accuracy.csv `eval_dataset` -> data-eval attribute (drives hide-eval-* classes)
EVAL_ATTR = {
    "unprompted_roleplay": "unprompted",
    "prompted_roleplay": "prompted",
    "company_assistant": "company",
}

# accuracy.csv `eval_dataset` -> short eval-column display label
EVAL_DISP = {
    "unprompted_roleplay": "u",
    "prompted_roleplay": "p",
    "company_assistant": "c",
}

# prefix used in the self-test control's raw name ("<prefix>_self", data-rawctrl)
SELF_PREFIX = {
    "unprompted_roleplay": "unpr",
    "prompted_roleplay": "prom",
    "company_assistant": "comp",
}

# short prefix shown in the control column for self-tests ("<prefix>_self")
SELF_DISP = {
    "unprompted_roleplay": "u",
    "prompted_roleplay": "p",
    "company_assistant": "c",
}

# fixed controls: control_dataset -> (data-control, data-rawctrl, display).
# true/fake come at two FPR levels: the original 1% (true_stories/fake_stories)
# and the new 5% (*_5pct); each gets its own data-control so it filters apart.
FIXED_CONTROL = {
    "true_stories": ("true1", "true1", "true1%"),
    "true_stories_5pct": ("true5", "true5", "true5%"),
    "fake_stories": ("fake1", "fake1", "fake1%"),
    "fake_stories_5pct": ("fake5", "fake5", "fake5%"),
    "evasive_company_assistant": ("evasive", "evasive", "eva"),
}


def control_short(control_dataset: str):
    """Return (data-control, data-rawctrl, display) for one control_dataset."""
    if control_dataset in FIXED_CONTROL:
        return FIXED_CONTROL[control_dataset]
    if control_dataset.endswith("_test_honest"):
        ds = control_dataset[: -len("_test_honest")]
        return ("self", f"{SELF_PREFIX.get(ds, ds)}_self",
                f"{SELF_DISP.get(ds, ds)}_self")
    # Unknown control: fall back to the raw name in all slots.
    return control_dataset, control_dataset, control_dataset


def model_key(model: str) -> str:
    """CSS/JS-safe slug for a model id, e.g. 'qwen-qwen2-5-32b-instruct'."""
    return re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")


def panel_label(model: str) -> str:
    """e.g. 'Qwen/Qwen2.5-32B-Instruct' -> 'Qwen32', 'google/gemma-2-9b-it' ->
    'Gemma9' (family + parameter count, matching the original viewer)."""
    name = model.split("/")[-1].lower()
    if "qwen" in name:
        fam = "Qwen"
    elif "llama" in name:
        fam = "Llama"
    elif "gemma" in name:
        fam = "Gemma"
    else:
        fam = model.split("/")[-1].split("-")[0].capitalize()
    m = re.search(r"(\d+)b", name)   # parameter count, e.g. 32b / 8b / 9b / 27b
    return fam + (m.group(1) if m else "")


# panel display order: family (Qwen, then Llama, then Gemma, then others),
# and within a family by ascending parameter count.
FAMILY_ORDER = ["qwen", "llama", "gemma"]


def model_sort_key(model: str):
    name = model.split("/")[-1].lower()
    fam = next((i for i, f in enumerate(FAMILY_ORDER) if f in name),
               len(FAMILY_ORDER))
    m = re.search(r"(\d+)b", name)
    return (fam, int(m.group(1)) if m else 0, name)


THEAD = ("<thead><tr><th>layer</th><th>probe</th><th>control</th><th>fpr</th>"
         "<th>thresh</th><th>eval</th><th>#conv</th><th>tpr</th>"
         "<th>tnr</th><th>bal_acc</th></tr></thead>")


def row_html(r: dict, grp: int, first: bool) -> str:
    probe = r["probe"]
    combo = PROBE_DISPLAY.get(probe, probe).replace("+", "_")
    disp = PROBE_SHORT.get(probe, probe)
    ctrl, rawctrl, ctrl_disp = control_short(r["control_dataset"])
    ev = EVAL_ATTR.get(r["eval_dataset"], r["eval_dataset"])
    ev_disp = EVAL_DISP.get(r["eval_dataset"], r["eval_dataset"])
    bacc = float(r["balanced_accuracy"])
    isself = 1 if r["eval_dataset"] in PROBE_TRAINSETS.get(probe, []) else 0
    nh, nd = int(r["n_honest"]), int(r["n_deceptive"])
    # No static separator class: the block-divider line is drawn entirely by the
    # JS (grp-top on each block's first *visible* row), so it survives filtering
    # and re-sorting. See markBoundaries().
    attrs = (f' data-grp="{grp}" data-combo="{combo}" data-control="{ctrl}"'
             f' data-eval="{ev}" data-bacc="{bacc:.6f}" data-rawctrl="{rawctrl}"'
             f' data-isself="{isself}"')
    if first:
        attrs += f' data-ord="{grp}"'
    tds = (f'<td>{int(r["layer"])}</td><td>{disp}</td><td>{ctrl_disp}</td>'
           f'<td>{float(r["control_fpr"]):.2f}</td>'
           f'<td>{float(r["threshold"]):.2f}</td><td>{ev_disp}</td>'
           f'<td>{nh + nd} ({nh} / {nd})</td>'
           f'<td>{float(r["tpr"]):.2f}</td><td>{float(r["tnr"]):.2f}</td>'
           f'<td>{bacc:.2f}</td>')
    return f"<tr{attrs}>{tds}</tr>"


def panel_html(model: str, rows: list) -> str:
    body, grp = [], -1
    for i, r in enumerate(rows):
        key = (r["layer"], r["probe"])
        first = i == 0 or key != (rows[i - 1]["layer"], rows[i - 1]["probe"])
        if first:
            grp += 1
        body.append(row_html(r, grp, first))
    return (f'<section class="panel" data-model="{model_key(model)}">'
            f'<h2>{panel_label(model)}'
            f'<span class="count">{len(rows)} rows</span></h2>'
            f'<div class="scroll"><table>{THEAD}<tbody>'
            f'{"".join(body)}</tbody></table></div></section>')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default=str(DEFAULT_CSV),
                   help="accuracy.csv from compute-accuracy.py "
                        "(default: results/probe-comparison/accuracy.csv)")
    p.add_argument("--out", default=str(DEFAULT_OUT),
                   help="output HTML path (default: autocomp2/accuracy.html)")
    args = p.parse_args()

    by_model = {}            # model -> list of rows, in CSV order
    with open(args.csv, newline="") as fh:
        for r in csv.DictReader(fh):
            by_model.setdefault(r["model"], []).append(r)
    if not by_model:
        raise SystemExit(f"no rows in {args.csv}")

    # Per-model show/hide: a CSS rule + a toolbar toggle button per model. The
    # generic toggleClass() in the JS already drives the hide-model-<key> body
    # class, so no script change is needed.
    models = sorted(by_model, key=model_sort_key)
    model_css = "".join(
        '  body.hide-model-' + model_key(m) + ' section[data-model="'
        + model_key(m) + '"] { display: none; }\n' for m in models)
    model_btns = "".join(
        '<button class="combobtn active" onclick="toggleClass(\'model-'
        + model_key(m) + '\', this)">' + panel_label(m) + '</button>'
        for m in models)
    model_group = ('  <span class="sep"></span>\n'
                   '  <span class="lbl">models:</span>\n'
                   '  ' + model_btns + '\n')
    head = HEAD.replace("</style>", model_css + "</style>")
    head = head.replace('</div>\n<div class="grid">',
                        model_group + '</div>\n<div class="grid">')

    panels = "".join(panel_html(m, by_model[m]) for m in models)
    html = head + panels + TAIL
    Path(args.out).write_text(html)
    print(f"wrote {args.out} ({len(by_model)} models, "
          f"{sum(len(v) for v in by_model.values())} rows)")


# --- static template, embedded verbatim from the original accuracy.html ------

HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>accuracy comparison</title>
<style>
  body { font: 12px/1.4 -apple-system, system-ui, sans-serif; margin: 0; background: #f4f5f7; }
  .toolbar { display: flex; gap: 6px; align-items: center; flex-wrap: wrap;
              padding: 6px 8px; background: #1a202c; color: #fff; }
  .toolbar .lbl { font-size: 12px; opacity: .7; margin-right: 4px; }
  .toolbar button { font: 12px system-ui, sans-serif; padding: 4px 9px; cursor: pointer;
                     border: 1px solid #4a5568; background: #2d3748; color: #fff; border-radius: 4px; }
  .toolbar button:hover { background: #3d4758; }
  .toolbar button.active { background: #3182ce; border-color: #3182ce; }
  .toolbar .sep { width: 1px; height: 18px; background: #4a5568; margin: 0 2px; }
  .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px;
           padding: 8px; height: calc(100vh - 46px); box-sizing: border-box; }
  body.one-row .grid { grid-template-columns: none; grid-template-rows: 1fr;
           grid-auto-flow: column; grid-auto-columns: minmax(0, 1fr); }
  .panel { display: flex; flex-direction: column; min-width: 0; background: #fff;
            border: 1px solid #d0d3d9; border-radius: 6px; overflow: hidden; }
  h2 { margin: 0; padding: 8px 10px; font-size: 13px; background: #2d3748; color: #fff;
        display: flex; justify-content: space-between; align-items: baseline; }
  .count { font-weight: normal; opacity: .7; font-size: 11px; }
  .scroll { overflow: auto; flex: 1; }
  table { border-collapse: collapse; white-space: nowrap; }
  th, td { padding: 3px 7px; border-bottom: 1px solid #edeff2; text-align: right; }
  th { position: sticky; top: 0; background: #eef1f5; text-align: right;
        border-bottom: 2px solid #cbd2da; }
  tbody tr:hover { background: #fff7e0; }
  /* Block divider: the first *visible* row of each (layer, probe) block gets a
     thick top line, drawn dynamically by markBoundaries() so it always sits
     between blocks no matter how rows are filtered or how blocks are sorted.
     Painted as an inset box-shadow (not a border) so it is independent of the
     border-collapse grid, which Chromium drops at fractional zoom-out levels. */
  tbody tr.grp-top td { box-shadow: inset 0 3px 0 0 #1a202c; }
  .missing { padding: 12px; color: #c0392b; }
  body.hide-combo-unprompted tr[data-combo="unprompted"] { display: none; }
  body.hide-combo-prompted tr[data-combo="prompted"] { display: none; }
  body.hide-combo-company tr[data-combo="company"] { display: none; }
  body.hide-combo-unprompted_prompted tr[data-combo="unprompted_prompted"] { display: none; }
  body.hide-combo-prompted_company tr[data-combo="prompted_company"] { display: none; }
  body.hide-combo-unprompted_company tr[data-combo="unprompted_company"] { display: none; }
  body.hide-combo-all3 tr[data-combo="all3"] { display: none; }
  body.hide-ctrl-evasive tr[data-control="evasive"] { display: none; }
  body.hide-ctrl-self tr[data-control="self"] { display: none; }
  body.hide-ctrl-true1 tr[data-control="true1"] { display: none; }
  body.hide-ctrl-true5 tr[data-control="true5"] { display: none; }
  body.hide-ctrl-fake1 tr[data-control="fake1"] { display: none; }
  body.hide-ctrl-fake5 tr[data-control="fake5"] { display: none; }
  body.hide-eval-unprompted tr[data-eval="unprompted"] { display: none; }
  body.hide-eval-prompted tr[data-eval="prompted"] { display: none; }
  body.hide-eval-company tr[data-eval="company"] { display: none; }
</style>
</head>
<body>
<div class="toolbar">
  <span class="lbl">sort by:</span>
  <button class="sortbtn" data-attr="self-bacc" onclick="sortAll('self-bacc','desc')">self bal_acc</button><button class="sortbtn" data-attr="worst-bacc" onclick="sortAll('worst-bacc','desc')">worst bal_acc</button><button class="sortbtn" data-attr="best-u-bacc" onclick="sortAll('best-u-bacc','desc')">best u bal_acc</button><button class="sortbtn" data-attr="best-p-bacc" onclick="sortAll('best-p-bacc','desc')">best p bal_acc</button><button class="sortbtn" data-attr="best-c-bacc" onclick="sortAll('best-c-bacc','desc')">best c bal_acc</button><button class="sortbtn" data-attr="ord" onclick="sortAll('ord','asc')">original</button>
  <span class="sep"></span>
  <span class="lbl">probes:</span>
  <button class="combobtn active" onclick="toggleClass('combo-unprompted', this)">u</button><button class="combobtn active" onclick="toggleClass('combo-prompted', this)">p</button><button class="combobtn active" onclick="toggleClass('combo-company', this)">c</button><button class="combobtn active" onclick="toggleClass('combo-unprompted_prompted', this)">u+p</button><button class="combobtn active" onclick="toggleClass('combo-prompted_company', this)">p+c</button><button class="combobtn active" onclick="toggleClass('combo-unprompted_company', this)">u+c</button><button class="combobtn active" onclick="toggleClass('combo-all3', this)">all3</button>
  <span class="sep"></span>
  <span class="lbl">control:</span>
  <button class="combobtn active" onclick="toggleClass('ctrl-evasive', this)">eva</button><button class="combobtn active" onclick="toggleClass('ctrl-self', this)">self</button><button class="combobtn active" onclick="toggleClass('ctrl-true1', this)">true1%</button><button class="combobtn active" onclick="toggleClass('ctrl-true5', this)">true5%</button><button class="combobtn active" onclick="toggleClass('ctrl-fake1', this)">fake1%</button><button class="combobtn active" onclick="toggleClass('ctrl-fake5', this)">fake5%</button>
  <span class="sep"></span>
  <span class="lbl">eval:</span>
  <button class="combobtn active" onclick="toggleClass('eval-unprompted', this)">u</button><button class="combobtn active" onclick="toggleClass('eval-prompted', this)">p</button><button class="combobtn active" onclick="toggleClass('eval-company', this)">c</button>
  <span class="sep"></span>
  <span class="lbl">layout:</span>
  <button class="layoutbtn" onclick="toggleLayout(this)">next to each other</button>
</div>
<div class="grid">
"""

TAIL = """
</div>
<script>
// current sort; keys (self/worst bal_acc) are recomputed over VISIBLE rows only,
// so toggling a control/probe filter re-ranks the probes.
let curAttr = 'ord', curDir = 'asc';

function rowHidden(r) {
  const b = document.body.classList;
  return b.contains('hide-combo-' + r.getAttribute('data-combo'))
      || b.contains('hide-ctrl-' + r.getAttribute('data-control'))
      || b.contains('hide-eval-' + r.getAttribute('data-eval'));
}
function recomputeKeys(tb) {
  const groups = new Map();
  for (const r of tb.rows) {
    const g = r.getAttribute('data-grp');
    if (!groups.has(g)) groups.set(g, []);
    groups.get(g).push(r);
  }
  for (const rows of groups.values()) {
    let worst = null;            // min bal_acc over visible rows
    const byCtrl = {};          // per-control bal_acc on the probe's own (self) datasets
    const bestEval = { unprompted: null, prompted: null, company: null };  // best bal_acc per eval dataset
    for (const r of rows) {
      if (rowHidden(r)) continue;
      const v = parseFloat(r.getAttribute('data-bacc'));
      if (isNaN(v)) continue;
      worst = (worst === null) ? v : Math.min(worst, v);
      const ev = r.getAttribute('data-eval');
      if (ev in bestEval) bestEval[ev] = (bestEval[ev] === null) ? v : Math.max(bestEval[ev], v);
      if (r.getAttribute('data-isself') === '1') {
        const c = r.getAttribute('data-rawctrl');
        (byCtrl[c] = byCtrl[c] || []).push(v);
      }
    }
    let self = null;             // best control's mean over the trained dataset(s)
    for (const c in byCtrl) {
      const a = byCtrl[c], m = a.reduce((x, y) => x + y, 0) / a.length;
      self = (self === null) ? m : Math.max(self, m);
    }
    const first = rows[0];
    first.setAttribute('data-self-bacc', self === null ? '' : self);
    first.setAttribute('data-worst-bacc', worst === null ? '' : worst);
    first.setAttribute('data-best-u-bacc', bestEval.unprompted === null ? '' : bestEval.unprompted);
    first.setAttribute('data-best-p-bacc', bestEval.prompted === null ? '' : bestEval.prompted);
    first.setAttribute('data-best-c-bacc', bestEval.company === null ? '' : bestEval.company);
  }
}
function groupKey(firstRow, attr) {
  const v = firstRow.getAttribute('data-' + attr);
  if (v === null || v === '') return null;
  const n = parseFloat(v);
  return isNaN(n) ? null : n;
}
function sortPanel(tb, attr, dir) {
  const groups = new Map();
  for (const r of tb.rows) {
    const g = r.getAttribute('data-grp');
    if (!groups.has(g)) groups.set(g, []);
    groups.get(g).push(r);
  }
  const order = [...groups.keys()].sort((a, b) => {
    const va = groupKey(groups.get(a)[0], attr), vb = groupKey(groups.get(b)[0], attr);
    if (va === null && vb === null) return 0;
    if (va === null) return 1;        // undefined keys sink to the bottom
    if (vb === null) return -1;
    return dir === 'asc' ? va - vb : vb - va;
  });
  const frag = document.createDocumentFragment();
  for (const g of order) for (const r of groups.get(g)) frag.appendChild(r);
  tb.appendChild(frag);
}
function markBoundaries(tb) {
  // draw the group separator on each group's first *visible* row, so it survives filtering
  const groups = new Map();
  for (const r of tb.rows) {
    r.classList.remove('grp-top');
    const g = r.getAttribute('data-grp');
    if (!groups.has(g)) groups.set(g, []);
    groups.get(g).push(r);
  }
  for (const rows of groups.values()) {
    const first = rows.find(r => !rowHidden(r));
    if (first) first.classList.add('grp-top');
  }
}
function applySort() {
  document.querySelectorAll('tbody').forEach(tb => {
    recomputeKeys(tb);
    sortPanel(tb, curAttr, curDir);
    markBoundaries(tb);
  });
  document.querySelectorAll('.toolbar .sortbtn').forEach(
    b => b.classList.toggle('active', b.dataset.attr === curAttr));
}
function sortAll(attr, dir) {
  curAttr = attr; curDir = dir;
  applySort();
}
function toggleClass(name, btn) {
  btn.classList.toggle('active');                       // active == shown
  document.body.classList.toggle('hide-' + name, !btn.classList.contains('active'));
  applySort();                                          // filter affects the ranking
}
function toggleLayout(btn) {
  btn.classList.toggle('active');                       // active == single row
  document.body.classList.toggle('one-row', btn.classList.contains('active'));
}
document.querySelectorAll('tbody').forEach(markBoundaries);  // initial separators
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
