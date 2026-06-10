# Round 3 — spec closed, implementation design choices

## Final spec resolutions (from round 2/3)

- Balancing: subsampling only, sklearn `class_weight=None`; "balanced" also truncates
  honest vs deceptive within single-dataset probes; seed = 42 everywhere, with a fresh
  RNG instance per (model, dataset, class) pool so token selection is independent of
  execution order and of the layer.
- Metrics: recall at **5% and 10% FPR** (both, for all datasets), defined as recall at
  the largest empirical-ROC threshold with FPR ≤ the target (no interpolation), plus
  AUROC, ROC curve data, score lists.
- Everything else as recorded in `train-and-compare-probes-review-2.md` §Settled.

## Design choices — my recommendation first in each case

### 1. Training backend: torch on GPU, validated once against sklearn

~1,400+ logistic regressions on matrices up to ~300k × 8192 make sklearn-LBFGS-on-CPU
a multi-day job; on GPU each fit is seconds. Recommendation:

- Implement L2-regularized logistic regression in torch (full-batch LBFGS), with the
  objective matched to sklearn's (`0.5·wᵀw + C·Σ log-loss`, C = 0.1, no intercept,
  inputs standardized by a train-set StandardScaler computed in torch).
- **Validation step built in**: for one small config (Llama-8B, one layer, one dataset)
  also fit sklearn `LogisticRegression(C=0.1, fit_intercept=False)` and assert the
  probe directions' cosine similarity ≈ 1 and AUROCs match to ~1e-3. Cheap insurance
  that the torch implementation reproduces the repe convention.
- Fallback flag `--backend sklearn` for spot-checking.

### 2. Memory/loading: `torch.load(..., mmap=True)`, layer-major loop

The 70B/72B activation files are ~100–200 GB each; never materialize one fully.

- Outer loop over models; open all 6 dataset `.pt` files memory-mapped.
- Inner loop over layers (the per-model intersection): slice `[:, 0, :, layer_idx, :]`
  from each mmap, drop all-zero padding rows, cast fp32, move to GPU. Per layer this is
  ~10–15 GB fp32 across all 6 datasets — fits comfortably on an 80 GB GPU (flag to
  keep on CPU if not).
- Train all 7 probes + run all evaluations for that layer, write results, free, next
  layer. Token subsampling is layer-independent (seed 42 per pool), so the same
  token indices are reused at every layer — compute the index sets once per
  (model, dataset, class).

### 3. Output layout: one results tree + flat summary table

```
results/probe-comparison/
  summary.csv                      # one row per (model, layer, probe, eval_dataset):
                                   # auroc, recall@5%fpr, recall@10%fpr, n_honest, n_deceptive
  <model>/
    layer<L>/
      probe_<trainsets>/           # e.g. probe_unprompted_roleplay,
                                   #      probe_prompted_roleplay+company_assistant,
                                   #      probe_all3
        probe.pt                   # direction, scaler mean/scale, config (C, layer,
                                   #   train datasets, n per pool, seed, source files)
        eval_<dataset>.json        # per-conversation scores [(conv_id, score, label)],
                                   #   metrics, ROC curve points (fpr, tpr, thresholds)
  plots/                           # post-hoc step: best layer per (model, probe) by
                                   #   AUROC on its own test split (for combined probes:
                                   #   mean AUROC over their training datasets' test splits)
```

`summary.csv` is the file you'll actually analyze; everything needed to regenerate any
PNG later lives in the `eval_*.json` files. Note one decision embedded here: combined
probes have no single "own" test split, so "best layer" for them = best mean AUROC
across their constituent datasets' test splits. Shout if you'd rather use a fixed
reference dataset.

### 4. What scores to store: per-conversation only

Per-token scores for every (probe × eval) would be tens of GB across the grid. I'd
store per-conversation mean scores everywhere (that's what all metrics use), and add a
`--save-token-scores` flag that additionally dumps token-level scores, intended for
selected configs only. Default off.

### 5. Plotting as a separate sub-command, same script

`python train-and-compare-probes.py train` runs the grid; `... plot` reads
`summary.csv` + `eval_*.json` and renders the best-layer PNGs (ROC curve + honest/
deceptive histogram per full dataset, histogram per control dataset). Keeps the
expensive run free of matplotlib and lets you re-plot with different choices later.

### 6. Resumability and failure isolation

The full grid is hours of GPU time; the script should be killable. Before training a
probe, check whether its `probe.pt` + all `eval_*.json` exist and skip if so
(`--force` to redo). Each (model, layer) block is independent, so a crash loses at
most one block. Also `--models`, `--layers`, `--datasets` CLI filters for partial runs
and debugging.

### 7. Missing files: fail with a manifest, no auto-download

At startup, resolve every required file (activation `.pt` + scored json per the
selection rule) for the requested grid and print the full manifest; if anything is
missing, list it and exit before any compute. No silent HF downloading — you said the
node has everything, and an explicit manifest doubles as the log of which scored files
were used. (A `--download` convenience flag could be added later if needed.)

### 8. Numerics

Activations stored in fp16/bf16 → cast to fp32 for scaler statistics and training.
Scaler = per-dimension mean/std over the (subsampled, balanced) train tokens; std
floor of 1e-8. Score at eval = `(x - mean)/std · w` (raw logit, no sigmoid), mean over
the conversation's non-padding tokens; positive = deceptive.

## Sanity checks built into the run

- Layer lists: computed per model as the intersection across all 6 datasets; warn if
  smaller than expected (catches a stale more-roleplay file).
- Scored files: warn/fail if the selected file has fewer records than conversations.
- Label vocabularies: assert the expected labels are present in each selected file.
- The sklearn-vs-torch validation fit (design choice 1).

---

If these eight look right (or you only want to adjust a couple), the next step is
writing the actual script — it's a single file, and the structure above maps directly
onto ~6 functions plus a CLI.
