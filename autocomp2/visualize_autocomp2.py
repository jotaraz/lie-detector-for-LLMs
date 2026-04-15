#!/usr/bin/env python3
"""
visualize_autocomp2.py

Generates a self-contained HTML visualization of autocomp2 results.
Each conversation is listed in a left panel; hovering updates a right panel
showing how completions AC^(k)_i change as the prefix R^(k)_i grows token by
token, framed above and below by the full reference answer R^(k).

Usage:
    python visualize_autocomp2.py [data_dir] [output_file]

Defaults:
    data_dir     = data/
    output_file  = autocomp2_viz.html
"""

import argparse
import json
import sys
from pathlib import Path

DEFAULT_JSONL = "autoconv5.jsonl"
DATA_DIR = Path("data-" + Path(DEFAULT_JSONL).stem)
OUTPUT_FILE = Path(Path(DEFAULT_JSONL).stem + "_viz.html")


def load_models(data_dir: Path) -> dict:
    files = sorted(data_dir.glob("*.json"))
    if not files:
        print(f"No *.json files found in {data_dir}/", file=sys.stderr)
        sys.exit(1)

    models = {}
    for f in files:
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        model_name = data.get("model", f.stem.replace("--", "/"))
        models[model_name] = data
        n = len(data["conversations"])
        print(f"  {model_name}: {n} conversations, m={data['m']}, c={data['c']}")
    return models


def build_data(models: dict) -> tuple[list, list]:
    model_names = list(models.keys())
    first = models[model_names[0]]

    conversations = []
    for conv in first["conversations"]:
        k = conv["k"]

        model_data = {}
        for name in model_names:
            mconv_list = models[name]["conversations"]
            if k >= len(mconv_list):
                continue
            mc = mconv_list[k]
            model_data[name] = {
                "reference_answer": mc["reference_answer"],
                "reference_tokens": mc["reference_tokens"],
                "n": mc["n"],
                "autocompletions": mc["autocompletions"],
            }

        conversations.append({
            "k": k,
            "system_prompt": conv["system_prompt"],
            "user_prompt": conv["user_prompt"],
            "reference_answer": conv["reference_answer"],
            "reference_tokens": conv["reference_tokens"],
            "n": conv["n"],
            "models": model_data,
        })

    return conversations, model_names


def generate_html(conversations: list, model_names: list) -> str:
    data_json = json.dumps(conversations, ensure_ascii=False, separators=(",", ":"))
    models_json = json.dumps(model_names, ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Autocomp2 Visualization</title>
<style>
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

body {{
  font-family: 'Courier New', Courier, monospace;
  font-size: 13px;
  background: #0d1117;
  color: #c9d1d9;
  display: flex;
  height: 100vh;
  overflow: hidden;
}}

/* ── Left panel ──────────────────────────────────────────────── */
#list-panel {{
  width: 340px;
  min-width: 240px;
  min-height: 0;
  border-right: 1px solid #30363d;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  resize: horizontal;
}}

#list-header {{
  padding: 10px 14px;
  background: #161b22;
  border-bottom: 1px solid #30363d;
  color: #58a6ff;
  font-weight: bold;
  font-size: 14px;
  flex-shrink: 0;
  user-select: none;
}}

#list-header span {{
  color: #6e7681;
  font-size: 11px;
  font-weight: normal;
}}

#conv-list {{
  overflow-y: auto;
  flex: 1;
  padding: 6px;
}}

.conv-card {{
  padding: 8px 10px;
  margin-bottom: 4px;
  border: 1px solid #21262d;
  border-radius: 5px;
  cursor: pointer;
  background: #161b22;
  transition: border-color 0.12s, background 0.12s;
}}
.conv-card:hover   {{ border-color: #388bfd; background: #1c2432; }}
.conv-card.active  {{ border-color: #58a6ff; background: #1c2432; }}

.card-idx      {{ color: #6e7681; font-size: 10px; margin-bottom: 2px; }}
.card-scenario {{
  color: #8b949e; font-size: 11px;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  margin-bottom: 3px;
}}
.card-question {{
  color: #79c0ff;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  margin-bottom: 3px;
}}
.card-answer {{
  color: #c9d1d9; font-size: 11px;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}}

/* ── Right panel ─────────────────────────────────────────────── */
#detail-panel {{
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  padding: 16px 20px;
}}

#detail-inner {{
  display: flex;
  flex-direction: column;
  gap: 10px;
  min-height: 100%;
}}

#placeholder {{
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: #484f58;
  font-size: 15px;
}}

.info-box {{
  background: #161b22;
  border: 1px solid #30363d;
  border-radius: 6px;
  padding: 9px 13px;
}}
.info-label {{
  color: #58a6ff;
  font-size: 10px;
  font-weight: bold;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin-bottom: 5px;
}}
.info-text {{
  line-height: 1.6;
  white-space: pre-wrap;
  word-break: break-word;
  color: #8b949e;
}}
.info-text.bright {{ color: #c9d1d9; }}

.collapsible {{ cursor: pointer; }}
.collapsible .toggle {{
  color: #484f58; font-size: 10px; margin-left: 6px;
}}
.collapsed .info-text {{
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}}

/* Model tabs */
.model-tabs {{ display: flex; gap: 6px; flex-wrap: wrap; }}
.tab {{
  padding: 4px 12px;
  border: 1px solid #30363d;
  border-radius: 20px;
  cursor: pointer;
  background: #161b22;
  color: #8b949e;
  font-size: 12px;
  transition: all 0.12s;
  user-select: none;
}}
.tab:hover  {{ border-color: #388bfd; color: #c9d1d9; }}
.tab.active {{ background: #1f6feb; border-color: #1f6feb; color: #fff; }}

/* Answer frames */
.answer-frame {{
  background: #0d2137;
  border: 1px solid #1f6feb;
  border-radius: 6px;
  padding: 10px 14px;
}}
.frame-label {{
  color: #58a6ff;
  font-size: 10px;
  font-weight: bold;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin-bottom: 6px;
}}
.frame-text {{
  line-height: 1.6;
  white-space: pre-wrap;
  word-break: break-word;
  color: #e6edf3;
}}

/* Autocomplete table */
.ac-wrap {{
  border: 1px solid #30363d;
  border-radius: 6px;
  overflow: hidden;
}}
.ac-table {{
  width: 100%;
  border-collapse: collapse;
  background: #161b22;
}}
.ac-table thead tr {{ background: #21262d; }}
.ac-table th {{
  padding: 6px 10px;
  text-align: left;
  color: #58a6ff;
  font-size: 10px;
  letter-spacing: 0.08em;
  border-bottom: 1px solid #30363d;
}}
.ac-table td {{
  padding: 6px 10px;
  vertical-align: top;
  border-bottom: 1px solid #21262d;
  line-height: 1.55;
  white-space: pre-wrap;
  word-break: break-word;
}}
.ac-table tr:last-child td {{ border-bottom: none; }}
.ac-table tbody tr:hover td {{ background: #1c2432; }}

.col-i      {{ width: 28px; color: #6e7681; font-size: 11px; }}
.col-prefix {{ width: 36%; color: #8b949e; }}
.col-comp   {{ color: #c9d1d9; }}

.new-token    {{ color: #e3b341; font-weight: bold; }}
.empty-label  {{ color: #484f58; font-style: italic; }}

sup, sub {{ font-size: 0.75em; }}

::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-track {{ background: #0d1117; }}
::-webkit-scrollbar-thumb {{ background: #30363d; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: #484f58; }}
</style>
</head>
<body>

<div id="list-panel">
  <div id="list-header">
    Autocomp2 Viz &nbsp;<span id="conv-count"></span>
  </div>
  <div id="conv-list"></div>
</div>

<div id="detail-panel">
  <div id="detail-inner">
    <div id="placeholder">&larr; Hover over a conversation</div>
  </div>
</div>

<script>
const CONVERSATIONS = {data_json};
const MODEL_NAMES   = {models_json};

let activeIdx   = null;
let activeModel = MODEL_NAMES[0];

const listEl   = document.getElementById('conv-list');
const countEl  = document.getElementById('conv-count');
const detailEl = document.getElementById('detail-inner');

countEl.textContent = CONVERSATIONS.length + ' conversations';

CONVERSATIONS.forEach((conv, idx) => {{
  const card = document.createElement('div');
  card.className   = 'conv-card';
  card.dataset.idx = idx;

  const scenario = conv.system_prompt.replace(/^You are /, '').replace(/\\.\\s*Answer directly\\.?$/i, '').trim();

  card.innerHTML = `
    <div class="card-idx">k=${{conv.k}}</div>
    <div class="card-scenario" title="${{esc(scenario)}}">S<sup>(${{conv.k}})</sup>: ${{esc(trunc(scenario, 65))}}</div>
    <div class="card-question" title="${{esc(conv.user_prompt)}}">U<sup>(${{conv.k}})</sup>: ${{esc(trunc(conv.user_prompt, 55))}}</div>
    <div class="card-answer"   title="${{esc(conv.reference_answer)}}">R<sup>(${{conv.k}})</sup>: ${{esc(trunc(conv.reference_answer, 55))}}</div>
  `;

  card.addEventListener('mouseenter', () => showDetail(idx));
  card.addEventListener('click',      () => showDetail(idx));
  listEl.appendChild(card);
}});

detailEl.addEventListener('click', e => {{
  const tab = e.target.closest('.tab');
  if (!tab) return;
  activeModel = tab.dataset.model;
  renderDetail();
}});

detailEl.addEventListener('click', e => {{
  const box = e.target.closest('.collapsible');
  if (!box) return;
  box.classList.toggle('collapsed');
  box.querySelector('.toggle').textContent = box.classList.contains('collapsed') ? '[+]' : '[\\u2013]';
}});

function showDetail(idx) {{
  document.querySelectorAll('.conv-card').forEach(c => c.classList.remove('active'));
  const card = document.querySelector(`.conv-card[data-idx="${{idx}}"]`);
  if (card) card.classList.add('active');
  if (activeIdx === idx) return;
  activeIdx = idx;
  renderDetail();
}}

function renderDetail() {{
  const conv = CONVERSATIONS[activeIdx];
  if (!conv) return;

  const md = conv.models[activeModel];
  if (!md) return;

  const k = conv.k;
  const refTokens = md.reference_tokens;
  const autocompletions = md.autocompletions;

  const tabsHtml = MODEL_NAMES.map(m =>
    `<span class="tab${{m === activeModel ? ' active' : ''}}" data-model="${{esc(m)}}">${{esc(m)}}</span>`
  ).join('');

  let rowsHtml = '';
  autocompletions.forEach((ac) => {{
    const i = ac.i;
    const prevI = i > 0 ? i - 1 : -1;

    /* Build prefix: join reference_tokens[0..i) */
    let prefixCell;
    if (i === 0) {{
      prefixCell = '<span class="empty-label">(empty)</span>';
    }} else {{
      const oldTokens = refTokens.slice(0, i - 1).join('');
      const newToken  = refTokens[i - 1];
      prefixCell = esc(oldTokens) + '<span class="new-token">' + esc(newToken) + '</span>';
    }}

    const compText = ac.text;
    const compCell = compText
      ? esc(compText)
      : '<span class="empty-label">(empty)</span>';

    rowsHtml += `
      <tr>
        <td class="col-i">${{i}}</td>
        <td class="col-prefix">${{prefixCell}}</td>
        <td class="col-comp">${{compCell}}</td>
      </tr>`;
  }});

  detailEl.innerHTML = `
    <!-- S^(k) -->
    <div class="info-box collapsible collapsed">
      <div class="info-label">S<sup>(${{k}})</sup> &mdash; System prompt <span class="toggle">[+]</span></div>
      <div class="info-text">${{esc(conv.system_prompt)}}</div>
    </div>

    <!-- U^(k) -->
    <div class="info-box">
      <div class="info-label">U<sup>(${{k}})</sup> &mdash; User prompt</div>
      <div class="info-text bright">${{esc(conv.user_prompt)}}</div>
    </div>

    <!-- Model tabs -->
    <div class="model-tabs">${{tabsHtml}}</div>

    <!-- Top frame: R^(k) -->
    <div class="answer-frame">
      <div class="frame-label">&#9650; R<sup>(${{k}})</sup> &mdash; Reference answer &nbsp;<span style="color:#6e7681;font-size:10px;font-weight:normal">(n<sup>(${{k}})</sup>=${{md.n}} tokens)</span></div>
      <div class="frame-text">${{esc(md.reference_answer)}}</div>
    </div>

    <!-- Autocomplete table -->
    <div class="ac-wrap">
      <table class="ac-table">
        <thead>
          <tr>
            <th class="col-i"><i>i</i></th>
            <th>R<sup>(${{k}})</sup><sub><i>i</i></sub> &nbsp;<span style="color:#6e7681;font-size:10px">(prefix &mdash; <span style="color:#e3b341">gold</span> = new token)</span></th>
            <th>AC<sup>(${{k}})</sup><sub><i>i</i></sub> &nbsp;<span style="color:#6e7681;font-size:10px">(model completion after prefix)</span></th>
          </tr>
        </thead>
        <tbody>${{rowsHtml}}</tbody>
      </table>
    </div>

    <!-- Bottom frame: R^(k) -->
    <div class="answer-frame">
      <div class="frame-label">&#9660; R<sup>(${{k}})</sup> &mdash; Reference answer</div>
      <div class="frame-text">${{esc(md.reference_answer)}}</div>
    </div>
  `;
}}

function esc(str) {{
  return String(str ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}}

function trunc(str, n) {{
  return str.length > n ? str.slice(0, n) + '\\u2026' : str;
}}
</script>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate self-contained HTML visualization of autocomp results.")
    parser.add_argument("--jsonl", type=str, default=None,
                        help="JSONL conversations file used for generation "
                             "(default: autoconv3.jsonl). data_dir and output "
                             "filename are derived from its stem, e.g. "
                             "autoconv4.jsonl -> data-autoconv4/ + autoconv4_viz.html.")
    parser.add_argument("data_dir", nargs="?", default=None,
                        help="Override data directory (positional, optional).")
    parser.add_argument("output_file", nargs="?", default=None,
                        help="Override output HTML path (positional, optional).")
    args = parser.parse_args()

    if args.jsonl:
        stem = Path(args.jsonl).stem
        data_dir = Path(f"data-{stem}")
        output_file = Path(f"{stem}_viz.html")
    else:
        data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
        output_file = Path(args.output_file) if args.output_file else OUTPUT_FILE

    print(f"Loading from {data_dir}/")
    models = load_models(data_dir)

    print(f"\nBuilding conversation data...")
    conversations, model_names = build_data(models)
    print(f"  {len(conversations)} conversations, {len(model_names)} models")

    print(f"\nGenerating HTML...")
    html = generate_html(conversations, model_names)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Written -> {output_file}  ({len(html):,} bytes)")
    print(f"\nOpen with:  open {output_file}")


if __name__ == "__main__":
    main()
