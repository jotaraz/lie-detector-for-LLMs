# Design: Multi-Dataset Union Visualisation

This document describes the design of `visualize_union.py`, a variant of
`visualize_labeled_activations.py` that combines several autoconv datasets
(e.g. autoconv4, autoconv5, autoconv6) into a single interactive HTML page.

---

## What the existing script does (brief recap)

`visualize_labeled_activations.py` processes one dataset at a time. For a
given model it loads:

- A **JSON file** (`data-autoconv5/meta-llama--Llama-3.1-8B-Instruct.json`)
  containing the conversation texts, tokens, and metadata. Key fields:
  - `N` — number of conversations
  - `c` — how many prefix positions were considered
  - `m` — how many autocompletion tokens were generated per prefix
  - `layers` — which transformer layers were recorded

- A **PT file** (`.pt`) containing two tensors:
  - `autocompletion_activations` — shape `(N, c, m, L, d)`:
    for each conversation, each prefix position, each generated token, each
    recorded layer, the hidden-state vector of dimension `d`.
  - `full_convo_activations` — shape `(N, L, d)`:
    the hidden state at the end of the full reference answer.

It then flattens all those vectors, fits a dimensionality-reduction method
(PCA, UMAP, Isomap) on them, and embeds everything into a single 2-D
coordinate space. The result is written as a self-contained HTML file with
Plotly.js, where the user can toggle which conversations, prefix positions,
and views are visible.

---

## Goal of the new script

Produce the **same kind of interactive HTML**, but showing activations from
**multiple datasets at once**, all projected into a **shared 2-D space**.

A "dataset" here means one autoconv run: autoconv4, autoconv5, autoconv6,
etc. Each has its own `data-autoconvN/` directory and its own JSON + PT
files for each model.

Key requirements:

1. **Shared projection space.** PCA / UMAP / Isomap is fitted on the union
   of all activation vectors from all datasets. Every point from every
   dataset lands in the same 2-D plane, so relative positions across
   datasets are meaningful.

2. **Datasets stay independent inside the HTML.** Because `c` and `m` differ
   between datasets, the controls for prefix positions and autocomplete
   positions must be per-dataset. The layer selector and method selector are
   shared (same layers are recorded for the same model across runs).

3. **Visual distinction by dataset.** Each dataset gets a distinct marker
   symbol (circle, square, diamond, triangle-up, …) so you can tell at a
   glance which dataset a cluster of points comes from, even when all
   datasets are shown simultaneously.

4. **Dataset-level on/off toggles.** A row of buttons lets you hide or show
   everything from a given dataset with one click, in addition to the
   fine-grained per-conversation buttons that already exist.

---

## Input and CLI

```
python visualize_union.py \
    --datasets autoconv4 autoconv5 autoconv6 \
    [--model meta-llama/Meta-Llama-3.1-8B-Instruct] \
    [--methods PCA UMAP Isomap]
```

- `--datasets` accepts one or more autoconv names. From each name the data
  directory is derived as `data-{name}/` (same convention as the existing
  script).
- `--model`, if omitted, processes all models listed in `models.md`.
- The output HTML is written next to the data directories, named e.g.
  `union_autoconv4+5+6_meta-llama--Llama-3.1-8B-Instruct.html`.

---

## Loading phase

For each dataset name and each model:

1. Open `data-{name}/{model_filename}.json` → read `N`, `c`, `m`, `layers`,
   and the list of conversations (tokens, labels, etc.).
2. Open `data-{name}/{model_filename}.pt` → load `autocompletion_activations`
   (shape `N × c × m × L × d`) and `full_convo_activations` (shape `N × L × d`).

After loading all datasets for a model we have a list of dataset objects,
each carrying its own `(N_i, c_i, m_i, L, d)` tensors. `L` and `d` are the
same across datasets for the same model (same layers recorded, same hidden
size); `N_i`, `c_i`, and `m_i` can all differ.

---

## Joint projection

For each layer and each method:

1. **Flatten** all activation vectors from all datasets into one big matrix:
   - Autocompletion vectors: reshape each dataset's `(N_i, c_i, m_i, d)` slice
     to `(N_i · c_i · m_i, d)` and concatenate across datasets.
   - Full-convo vectors: stack each dataset's `(N_i, d)` slice.
   - Concatenate autocompletion and full-convo vectors into a single matrix
     of shape `(total_points, d)`.

2. **Filter** rows that are all-zero (padding / missing positions) using a
   `valid_mask`, exactly as the existing script does.

3. **Fit** the dimensionality-reduction model (PCA / UMAP / Isomap) on the
   valid rows of the combined matrix. This is the step that makes all
   datasets share the same coordinate system.

4. **Transform** the combined matrix and split the resulting 2-D coordinates
   back into per-dataset arrays (by tracking how many rows each dataset
   contributed). Store them reshaped to `(N_i, c_i, m_i, 2)` for
   autocompletion and `(N_i, 2)` for full-convo.

The fitted projector is discarded after this — only the final 2-D
coordinates are kept.

---

## Data structure passed to the HTML

The existing script embeds a single `DATA` JSON object in the HTML. The new
script embeds a list of **dataset objects**, each self-contained:

```json
{
  "datasets": [
    {
      "name": "autoconv4",
      "symbol": "circle",
      "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
      "c": 5,
      "m": 30,
      "N": 20,
      "labels": [ ... ],       // same per-conversation label structure as before
      "projections": {          // keys like "layer8_PCA", "layer8_PCA_fc", ...
        "layer8_PCA": [ ... ],  // shape (N, c, m, 2)
        "layer8_PCA_fc": [ ... ] // shape (N, 2)
      }
    },
    {
      "name": "autoconv5",
      "symbol": "square",
      ...
    },
    {
      "name": "autoconv6",
      "symbol": "diamond",
      ...
    }
  ],
  "layers": [8, 12, 16, 24],
  "methods": ["PCA", "UMAP", "Isomap"]
}
```

`symbol` is assigned in order from a fixed list: circle, square, diamond,
triangle-up, triangle-down, cross, x, star. This is the marker symbol that
Plotly uses for every point belonging to that dataset.

The `labels` array and per-conversation structure are exactly the same as in
the existing script, just scoped to each dataset.

---

## HTML controls layout

The page has the following control rows, top to bottom:

### Row 1 — Layer and Method (shared, unchanged)
Dropdowns to select which layer and which projection method to display.

### Row 2 — Dataset toggles (new)
One button per dataset, labelled by name (e.g. "autoconv4", "autoconv5",
"autoconv6"). Clicking a button shows or hides **all** traces from that
dataset at once. Each button is coloured or styled to match the dataset's
symbol/colour scheme so it's visually connected to the points on the plot.

### Row 3 — Conversation toggles (per dataset, new)
Because each dataset has a different number of conversations, conversation
buttons are grouped by dataset. Each group is prefixed with the dataset name:

```
autoconv4:  [k=0] [k=1] [k=2] …   [hide all] [show all]
autoconv5:  [k=0] [k=1] [k=2] …   [hide all] [show all]
autoconv6:  [k=0] [k=1] …          [hide all] [show all]
```

When a dataset is globally hidden (Row 2), its conversation buttons are
greyed out and non-interactive.

### Row 4 — View toggles (shared)
The four view type buttons — **prefix**, **full convo**, **last act**,
**autocomplete** — remain global. Toggling "prefix" on/off applies to all
visible datasets simultaneously.

### Row 5 — Prefix-i sub-buttons (per dataset, new)
Because `c` differs between datasets, these are grouped per dataset:

```
autoconv4 prefix i:  [i=0] [i=1] … [i=4]    [show all] [hide all]
autoconv5 prefix i:  [i=0] [i=1] … [i=19]   [show all] [hide all]
autoconv6 prefix i:  [i=0] [i=1] … [i=9]    [show all] [hide all]
```

This row only appears when the "prefix" view is active.

### Row 6 — Autocomplete-i sub-buttons (per dataset, new)
Same structure as Row 5 but for the autocomplete view. Only appears when
"autocomplete" is active.

---

## Visual distinction between datasets

Each dataset is assigned a **marker symbol** from a fixed ordered list.
Within a dataset, the **colour** still encodes the conversation index `k`
exactly as before: even-k gets one colour, odd-k gets another (separate
colours for prefix traces vs. autocomplete traces).

This means:
- You can tell **which dataset** a point is from by its shape (circle,
  square, diamond, …).
- You can tell **which conversation** within a dataset by its colour.
- You can tell **which view type** (prefix vs. autocomplete) by the colour
  palette (blue/red for prefix, green/orange for autocomplete).

All three dimensions of information are visible simultaneously without any
ambiguity.

---

## What stays the same as the existing script

- The label construction functions (`make_prefix_label`, `make_autocomplete_label`,
  etc.) are reused unchanged.
- The projection helpers (`project_all`, `_try_import_umap`, etc.) are reused
  with minimal changes (they receive the combined matrix rather than one
  dataset's matrix).
- The Plotly rendering logic is structurally the same; the main change is that
  the outer loop iterates over datasets before iterating over conversations.
- The HTML template structure (head, style, script) is largely the same.

---

## Edge cases to handle

- **A dataset has no PT file yet** (e.g. text was generated but activations
  were not): skip that dataset with a warning, continue with the rest.
- **A dataset is missing a particular layer** (shouldn't happen per the
  problem statement, but just in case): that dataset contributes no points
  for that layer, and the projection for that layer is computed from the
  remaining datasets only.
- **Fewer than 3 valid points total** across all datasets for a given layer:
  skip that layer with a warning (same guard as the existing script).
- **Only one dataset provided**: the script should work correctly and produce
  output equivalent to the existing script (just with a dataset toggle row
  that has a single button).
