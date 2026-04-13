#!/usr/bin/env python3
"""
visualize_autocomplete.py

Generates a self-contained HTML visualization of autocomplete results.
Each conversation is listed in a left panel; hovering updates a right panel
showing how completions change as the prefix grows token by token, framed
above and below by the full target answer.

Usage:
    python visualize_autocomplete.py [results_dir] [output_file]

Defaults:
    results_dir  = autocomplete_results/
    output_file  = autocomplete_viz.html
"""

import json
import sys
from pathlib import Path

RESULTS_DIR = Path("autocomplete_results")
OUTPUT_FILE = Path("autocomplete_viz.html")


def load_models(results_dir: Path) -> dict:
    files = sorted(results_dir.glob("autoconv3_*.jsonl"))
    if not files:
        print(f"No autoconv3_*.jsonl files found in {results_dir}/", file=sys.stderr)
        sys.exit(1)

    models = {}
    for f in files:
        model_name = f.stem.replace("autoconv3_", "")
        entries = []
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        models[model_name] = entries
        print(f"  {model_name}: {len(entries)} conversations")
    return models


def build_data(models: dict) -> tuple[list, list]:
    model_names = list(models.keys())
    first = models[model_names[0]]

    conversations = []
    for i, entry in enumerate(first):
        conv = entry["conversation"]
        scenario = next((m["content"] for m in conv if m["role"] == "system"), "")
        question = next((m["content"] for m in conv if m["role"] == "user"), "")
        answer   = next((m["content"] for m in conv if m["role"] == "assistant"), "")

        model_data = {}
        for name in model_names:
            if i >= len(models[name]):
                continue
            me = models[name][i]
            mc = me["conversation"]
            model_data[name] = {
                "answer": next((m["content"] for m in mc if m["role"] == "assistant"), ""),
                "autocompletes": me["autocompletes"],
            }

        conversations.append({
            "idx": i,
            "scenario": scenario,
            "question": question,
            "answer": answer,
            "models": model_data,
        })

    return conversations, model_names


def generate_html(conversations: list, model_names: list) -> str:
    data_json   = json.dumps(conversations,  ensure_ascii=False, separators=(",", ":"))
    models_json = json.dumps(model_names,    ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Autocomplete Visualization</title>
<style>
/* ── Reset & base ──────────────────────────────────────────── */
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

/* ── Left panel: conversation list ────────────────────────── */
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

/* ── Right panel: detail ───────────────────────────────────── */
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

/* Info boxes */
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

/* Collapsible scenario */
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

/* Answer frames (top & bottom) */
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
  text-transform: uppercase;
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

.col-n      {{ width: 28px; color: #6e7681; font-size: 11px; }}
.col-prefix {{ width: 36%; color: #8b949e; }}
.col-comp   {{ color: #c9d1d9; }}

.new-token    {{ color: #e3b341; font-weight: bold; }}
.empty-label  {{ color: #484f58; font-style: italic; }}

/* Scrollbars */
::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-track {{ background: #0d1117; }}
::-webkit-scrollbar-thumb {{ background: #30363d; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: #484f58; }}
</style>
</head>
<body>

<div id="list-panel">
  <div id="list-header">
    Autocomplete Viz &nbsp;<span id="conv-count"></span>
  </div>
  <div id="conv-list"></div>
</div>

<div id="detail-panel">
  <div id="detail-inner">
    <div id="placeholder">← Hover over a conversation</div>
  </div>
</div>

<script>
/* ── Data ───────────────────────────────────────────────────── */
const CONVERSATIONS = {data_json};
const MODEL_NAMES   = {models_json};

/* ── State ──────────────────────────────────────────────────── */
let activeIdx   = null;
let activeModel = MODEL_NAMES[0];

/* ── Build list ─────────────────────────────────────────────── */
const listEl     = document.getElementById('conv-list');
const countEl    = document.getElementById('conv-count');
const detailEl   = document.getElementById('detail-inner');

countEl.textContent = CONVERSATIONS.length + ' conversations';

CONVERSATIONS.forEach((conv, i) => {{
  const card = document.createElement('div');
  card.className   = 'conv-card';
  card.dataset.idx = i;

  // Strip "You are " preamble and truncate
  const scenario = conv.scenario.replace(/^You are /, '').replace(/\\.\\s*Answer directly\\.?$/i, '').trim();
  const q = conv.question;
  const a = conv.answer;

  card.innerHTML = `
    <div class="card-idx">#${{i + 1}}</div>
    <div class="card-scenario" title="${{esc(scenario)}}">${{esc(trunc(scenario, 72))}}</div>
    <div class="card-question" title="${{esc(q)}}">${{esc(trunc(q, 60))}}</div>
    <div class="card-answer"   title="${{esc(a)}}">${{esc(trunc(a, 60))}}</div>
  `;

  card.addEventListener('mouseenter', () => showDetail(i));
  card.addEventListener('click',      () => showDetail(i));
  listEl.appendChild(card);
}});

/* ── Event delegation for tabs ──────────────────────────────── */
detailEl.addEventListener('click', e => {{
  const tab = e.target.closest('.tab');
  if (!tab) return;
  activeModel = tab.dataset.model;
  renderDetail();
}});

/* Collapsible scenario */
detailEl.addEventListener('click', e => {{
  const box = e.target.closest('.collapsible');
  if (!box) return;
  box.classList.toggle('collapsed');
  box.querySelector('.toggle').textContent = box.classList.contains('collapsed') ? '[+]' : '[–]';
}});

/* ── Show detail on hover ───────────────────────────────────── */
function showDetail(idx) {{
  document.querySelectorAll('.conv-card').forEach(c => c.classList.remove('active'));
  const card = document.querySelector(`.conv-card[data-idx="${{idx}}"]`);
  if (card) card.classList.add('active');

  if (activeIdx === idx) return;
  activeIdx = idx;
  renderDetail();
}}

/* ── Render the detail panel ────────────────────────────────── */
function renderDetail() {{
  const conv = CONVERSATIONS[activeIdx];
  if (!conv) return;

  const modelData = conv.models[activeModel];
  if (!modelData) return;

  const {{ answer, autocompletes }} = modelData;
  // Use the conversation's own answer as the target frame
  const targetAnswer = conv.answer;

  /* Model tabs */
  const tabsHtml = MODEL_NAMES.map(m =>
    `<span class="tab${{m === activeModel ? ' active' : ''}}" data-model="${{esc(m)}}">${{esc(m)}}</span>`
  ).join('');

  /* Autocomplete rows */
  let rowsHtml = '';
  autocompletes.forEach((ac, i) => {{
    const prefix     = ac.assistant_prefix;
    const completion = ac.completion;
    const prevPrefix = i > 0 ? autocompletes[i - 1].assistant_prefix : '';
    const oldPart    = esc(prefix.slice(0, prevPrefix.length));
    const newPart    = esc(prefix.slice(prevPrefix.length));

    const prefixCell = prefix === ''
      ? '<span class="empty-label">(empty)</span>'
      : (oldPart + (newPart ? `<span class="new-token">${{newPart}}</span>` : ''));

    const compCell = completion
      ? esc(completion)
      : '<span class="empty-label">(empty)</span>';

    rowsHtml += `
      <tr>
        <td class="col-n">${{i}}</td>
        <td class="col-prefix">${{prefixCell}}</td>
        <td class="col-comp">${{compCell}}</td>
      </tr>`;
  }});

  detailEl.innerHTML = `
    <!-- Scenario (collapsible) -->
    <div class="info-box collapsible collapsed">
      <div class="info-label">Scenario <span class="toggle">[+]</span></div>
      <div class="info-text">${{esc(conv.scenario)}}</div>
    </div>

    <!-- Question -->
    <div class="info-box">
      <div class="info-label">Question</div>
      <div class="info-text bright">${{esc(conv.question)}}</div>
    </div>

    <!-- Model tabs -->
    <div class="model-tabs">${{tabsHtml}}</div>

    <!-- ── Top frame: target answer ── -->
    <div class="answer-frame">
      <div class="frame-label">&#9650; Target answer</div>
      <div class="frame-text">${{esc(targetAnswer)}}</div>
    </div>

    <!-- ── Autocomplete table ── -->
    <div class="ac-wrap">
      <table class="ac-table">
        <thead>
          <tr>
            <th class="col-n">#</th>
            <th>Prefix &nbsp;<span style="color:#6e7681;font-size:10px">(tokens given — <span style="color:#e3b341">gold</span> = new token)</span></th>
            <th>Model completion &nbsp;<span style="color:#6e7681;font-size:10px">(after prefix)</span></th>
          </tr>
        </thead>
        <tbody>${{rowsHtml}}</tbody>
      </table>
    </div>

    <!-- ── Bottom frame: target answer ── -->
    <div class="answer-frame">
      <div class="frame-label">&#9660; Target answer</div>
      <div class="frame-text">${{esc(targetAnswer)}}</div>
    </div>
  `;
}}

/* ── Helpers ────────────────────────────────────────────────── */
function esc(str) {{
  return String(str ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}}

function trunc(str, n) {{
  return str.length > n ? str.slice(0, n) + '…' : str;
}}
</script>
</body>
</html>"""


def main() -> None:
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS_DIR
    output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else OUTPUT_FILE

    print(f"Loading from {results_dir}/")
    models = load_models(results_dir)

    print(f"\nBuilding conversation data…")
    conversations, model_names = build_data(models)
    print(f"  {len(conversations)} conversations, {len(model_names)} models")

    print(f"\nGenerating HTML…")
    html = generate_html(conversations, model_names)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Written → {output_file}  ({len(html):,} bytes)")
    print(f"\nOpen with:  open {output_file}")


if __name__ == "__main__":
    main()
