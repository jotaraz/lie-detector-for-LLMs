# train-and-compare-probes — consolidated specification

Final spec and design decisions for the probe-training/comparison script, as agreed in
the review rounds (`-review.md`, `-review-2.md`, `-design.md`, `-memory-options.md`).
This file supersedes those where they differ. The script runs on a remote node with
large GPUs and all activation files local.

## 1. Goal

Train L2-regularized linear (logistic-regression) deception probes on token-level
activations of four probed models, for every available layer and several training-set
configurations, and compare how they generalize: each probe is evaluated on the test
splits of all three labeled datasets and on three control datasets.

## 2. Probed models

| Model | d | Saved layers |
|---|---|---|
| meta-llama/Llama-3.1-8B-Instruct | 4096 | 0–31 |
| meta-llama/Llama-3.3-70B-Instruct | 8192 | 0–59 |
| Qwen/Qwen2.5-32B-Instruct | 5120 | 0–59 |
| Qwen/Qwen2.5-72B-Instruct | 8192 | 0–59 |

Per model, the usable layer set is the **intersection of the layers present across all
six datasets' activation files**; the script warns if that intersection is smaller
than expected (catches stale files — the local more-roleplay files only carry 4
layers for three models; the remote/HF versions carry all).

## 3. Datasets

All paths relative to `autocomp2/`. Activations: `<dir>/<model-with-->.pt`; metadata:
`<dir>/<model-with-->.json` (keys: `model`, `m`, `c`, `layers`, `conversations`).

### Full datasets (train + evaluate)

| # | Name | Directory / source | Honest label | Deceptive label |
|---|---|---|---|---|
| 1 | unprompted_roleplay | `data-more-roleplay` | `HONEST` | `DECEPTIVE` |
| 2 | prompted_roleplay | `data-ir-honest` (honest side), `data-ir-dishonest` (deceptive side) | `HONEST` (from ir-honest only) | `DECEPTIVE` (from ir-dishonest only) |
| 3 | company_assistant | `data-company-assistant-roleplay` | `honest_authorized` | `deceptive` |

All other labels (`MIXED`, `NULL`, `evasive`, `incomplete`, `withheld_authorized`,
`UNSURE`, …) are dropped.

### Control datasets (evaluate only; expectation: probe scores look "honest")

| # | Name | Directory | Conversations used |
|---|---|---|---|
| A | true_stories | `data-true-stories` | all (unscored) |
| B | fake_stories | `data-fake-stories` | all (unscored) |
| C | evasive_company_assistant | `data-company-assistant-roleplay` | those labeled `evasive` |

Note on B: fake stories are fiction **on request** — fiction is not deception, so a
good probe should score them honest, same as true stories. This is the point of the
control. Control C never overlaps full dataset 3's training data (disjoint labels).

### Scored-file selection rule (verified against all files on disk, round 2)

For each (dataset, model): among the dataset directory's judge files for that model,
keep those with `judge_model == "gpt-5.4-mini"` **and** a timestamp in the filename
(timestamp-less files are an older schema and always lose); pick the **latest
timestamp**. This yields a unique, full-size file for all 16 (dataset, model) pairs;
for company-assistant it is always the `96dc4d84` prompt variant. Guards: fail/warn if
the chosen file has fewer records than the dataset has conversations, and assert the
expected label vocabulary is present. Filename schemas:
`scored-[variant-]YYYYMMDD-HHMMSS-<model>.json` with `verdicts`/`index` (dataset 1) vs
`<model>-judge-<judge>[-hash]-YYYYMMDD-HHMMSS.json` with `scores`/`k` (datasets 2, 3, C)
— per-dataset adapter functions handle this.

## 4. Activations and token-level data

- `.pt` files hold `autocompletion_activations` of shape `(N, c, m, L, d)` in fp16;
  here always `c = 1` → use completion index 0. `m` = 100 (dataset 1) or 200 (others).
- Training/eval points are **per-token activations of the response**. Rows that are
  all-zero are padding (response shorter than `m`) and are dropped.
- Conversations are matched to judge labels via their `k` (= `index`) field.
- Cast fp16 → fp32 for all computation.

## 5. Train/test split

- Conversation-wise, **deterministic, no shuffle**: after label filtering, the first
  80% of each (dataset, class) conversation list goes to train, the rest to test
  (split fraction configurable, default 0.8).
- Side effect (intended): the split is identical across models and layers, and
  ir-honest/ir-dishonest stay scenario-aligned across the split.
- Token counts per conversation vary, so train/test *token* ratios deviate from 4:1;
  that's accepted.

## 6. Training-set configurations — 7 probes per (model, layer)

| Probe | Train data |
|---|---|
| 3 × single | one full dataset's train split |
| 3 × pair | datasets 1+2, 1+3, 2+3 |
| 1 × triple | datasets 1+2+3 |

### Balancing (token-level subsampling; `--balance` switch, default `balanced`)

- **balanced**: every (dataset × class) token pool entering a probe's training set is
  truncated to the same `n` = minimum token count over all pools involved (2 pools for
  single, 4 for pair, 6 for triple — also applies within single-dataset probes).
- **unbalanced**: per-class `n` = min over the datasets involved (classes may differ);
  single-dataset probes are not truncated.
- Truncation = **fixed permutation, then prefix**: each pool is shuffled with a fresh
  RNG seeded **42** (one RNG per (model, dataset, class) pool, so selection is
  independent of execution order and **identical across layers**), then the first `n`
  tokens are taken. Token-level — conversation grouping deliberately ignored in train.
- sklearn-style `class_weight` is always `None`; balancing happens only via
  subsampling. The test split is never subsampled and stays conversation-grouped.

## 7. Probe training

- Model: L2-regularized logistic regression, **no intercept**, on StandardScaler-
  normalized activations (scaler fit on the probe's own subsampled train set;
  per-dimension mean/std, std floor 1e-8).
- Regularization: repe convention, `reg_coeff = 10.0` → **sklearn `C = 1/reg_coeff =
  0.1`**, i.e., objective `0.5·wᵀw + C·Σ log-loss`.
- Positive class = deceptive (honest ↦ −1 / deceptive ↦ +1 in spirit; in the LR the
  deceptive class is "1").
- Backend: **torch on GPU, full-batch LBFGS**, objective matched to sklearn's. Built-in
  one-off validation: for one small config (Llama-8B, one layer, one dataset) also fit
  sklearn `LogisticRegression(C=0.1, fit_intercept=False)` and assert direction cosine
  ≈ 1 and AUROC agreement to ~1e-3. `--backend sklearn` flag for spot checks.

## 8. Evaluation

Every probe is evaluated on: the 3 full datasets' **test splits** + the 3 control
datasets (controls in full).

- **Per-token score** = `((x − μ_train)/σ_train) · w` — raw logit, no sigmoid.
- **Per-conversation score** = mean of token scores over the response's non-padding
  tokens. Higher = more deceptive.
- Metrics on full-dataset test splits (conversation level):
  - AUROC (deceptive = positive),
  - **recall @ 5% FPR and recall @ 10% FPR** — recall at the largest empirical-ROC
    threshold with FPR ≤ target, no interpolation. (Known coarse cell: Llama-8B
    prompted_roleplay has only ~24 honest test conversations → FPR resolution ~4%.)
  - full ROC curve points (fpr, tpr, thresholds),
  - per-conversation scores with conversation ids and labels.
- Controls: per-conversation scores with ids (histogram data; no AUROC — there is no
  positive class).
- The pre-trained repe probes on HF are **ignored for now** (possible later addition).

## 9. Outputs

```
results/probe-comparison/
  summary.csv                       # one row per (model, layer, probe, eval_dataset):
                                    #   auroc, recall@5, recall@10, n_honest, n_deceptive, paths
  <model>/
    layer<L>/
      token_scores.pt               # fp16; dict probe -> eval_dataset ->
                                    #   {scores: flat tensor, conv_ids: index tensor}
                                    #   (~4–8 MB per block, ~1 GB total — saved always)
      probe_<trainsets>/            # probe_unprompted_roleplay, probe_1+3, probe_all3, ...
        probe.pt                    # w, scaler mean/scale, config: C, layer, train
                                    #   datasets, balance mode, n per pool, seed,
                                    #   selected scored files
        eval_<dataset>.json         # metrics + ROC points + per-conversation scores
  plots/                            # from the `plot` subcommand
```

- PNGs (ROC curve + honest/deceptive score histograms per full dataset; score
  histogram per control) are rendered **only for the best layer per (model, probe)**,
  chosen by AUROC on the probe's own test split; for combined probes, by **mean AUROC
  over the constituent datasets' test splits**. All data to plot any other layer later
  is in `eval_*.json` / `token_scores.pt`.

## 10. Script structure and CLI

- One file, two subcommands: `train` and `plot`.
- **One invocation per model**, looping over all layers internally:
  `python train-and-compare-probes.py train --model <hf-id> [--layers 0-59]
  [--device cuda:0] [--balance balanced] [--reg-coeff 10.0] [--split 0.8]
  [--backend torch] [--out results/probe-comparison] [--force]`
- `--layers` enables splitting a model across GPUs (e.g. `0-29` / `30-59`).
- **Resumable**: a (model, layer) block whose outputs all exist is skipped; `--force`
  redoes. Layer-independent work (scored-file selection, label parsing, splits,
  subsample index sets) is computed once per model run.
- **Startup manifest**: resolve every required file for the requested grid, print the
  manifest (this doubles as the record of which scored files were used); if anything
  is missing, list it and exit before any compute. No automatic HF download.

## 11. Memory strategy (decided: option B of `-memory-options.md`)

- `torch.load(..., mmap=True)` on all six activation files of the model; per layer,
  materialize the slice `[:, 0, :, layer_idx, :]`, drop padding, cast fp32, move to
  GPU (~11–21 GB/layer; `--device cpu`-resident fallback for smaller GPUs).
- All access goes through one `load_layer(model, dataset, layer)` function, so the
  escape hatches (layer batching; one-time per-layer file conversion) are drop-in
  replacements if the node's storage proves slow. Startup check that `mmap=True`
  actually works on the files (zipfile serialization format required).
- Expected I/O: ~6 min/model on NVMe for all layers; expected runtime ~1–2 min/layer
  → ~1–2 h/model; the four models can run concurrently on separate GPUs.

## 12. Sanity checks (all built into `train`)

1. Per-model layer intersection across datasets; warn if smaller than expected.
2. Scored-file guard: chosen file must cover all conversations; expected label
   vocabulary must be present.
3. torch-vs-sklearn validation fit (§7).
4. mmap availability check on the actual files.
5. Manifest printed before compute; hard fail on missing files.

## 13. Reference numbers (for orientation, measured/derived in review rounds)

- Labeled conversation counts (honest/deceptive) per model — dataset 1: 198–405 /
  203–406; dataset 2: 122–325 / 153–315; dataset 3: 418–468 / 443–462. Controls:
  500 + 500 + ~33–43 conversations.
- Grid: 212 (model, layer) blocks × 7 probes = **1,484 probes**, each with 6
  evaluations.
- Activation files: ~170 GB (8B) to ~637 GB (70B/72B) per model across all 6 datasets.
