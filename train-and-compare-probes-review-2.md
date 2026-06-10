# Round 2 — answers to your answers

Short version: points 1, 2, 5, 7, 8, 9 and the layer question are fully settled.
Point 4 I verified on disk — **the rule works, picks are unique, labels match your
description, no manual file pinning needed.** Point 6 (recall@5%) is well-defined
everywhere, with one coarse spot. Point 3 I understand, but it has three small
residual sub-questions (the only real open items left).

## Point 4 — scored-file selection: verified, no ambiguity

I applied the rule "`judge_model == gpt-5.4-mini`, timestamped beats timestamp-less,
latest timestamp wins" to every candidate file in all four data dirs × four models:

| Dataset | Model | Winner (ts) | n | Relevant labels |
|---|---|---|---|---|
| more-roleplay | Llama-8B | 20260608-155041 | 1000 | HONEST 198 / DECEPTIVE 403 |
| more-roleplay | Llama-70B | 20260608-161205 | 1000 | HONEST 211 / DECEPTIVE 406 |
| more-roleplay | Qwen-32B | 20260608-162007 | 1000 | HONEST 231 / DECEPTIVE 203 |
| more-roleplay | Qwen-72B | 20260608-161919 | 1000 | HONEST 405 / DECEPTIVE 276 |
| ir-honest | Llama-8B | 20260609-031020 | 371 | HONEST 122 |
| ir-honest | Llama-70B | 20260609-034237 | 371 | HONEST 325 |
| ir-honest | Qwen-32B | 20260609-030542 | 371 | HONEST 160 |
| ir-honest | Qwen-72B | 20260609-040606 | 371 | HONEST 279 |
| ir-dishonest | Llama-8B | 20260609-031054 | 371 | DECEPTIVE 296 |
| ir-dishonest | Llama-70B | 20260609-034825 | 371 | DECEPTIVE 315 |
| ir-dishonest | Qwen-32B | 20260609-024103 | 371 | DECEPTIVE 153 |
| ir-dishonest | Qwen-72B | 20260609-023507 | 371 | DECEPTIVE 247 |
| company-assistant | Llama-8B | 20260609-121843 (96dc4d84) | 1000 | hon_auth 418 / decep 462 / evasive 42 |
| company-assistant | Llama-70B | 20260609-124737 (96dc4d84) | 1000 | hon_auth 445 / decep 457 / evasive 33 |
| company-assistant | Qwen-32B | 20260609-131150 (96dc4d84) | 1000 | hon_auth 468 / decep 453 / evasive 37 |
| company-assistant | Qwen-72B | 20260609-134502 (96dc4d84) | 1000 | hon_auth 463 / decep 443 / evasive 43 |

Observations:

- **Unique winner everywhere.** No timestamp ties. The timestamp-less files turn out to
  be an older schema (no `label` field at all), so "timestamp beats no timestamp" also
  dodges a format problem.
- **Labels match your description** in every winning file (HONEST/DECEPTIVE for
  datasets 1–2; honest_authorized/deceptive/evasive for dataset 3).
- **The latest timestamp is also always the full-size run.** The small partial runs
  (n=20/30/50/100) all have earlier timestamps. For company-assistant the latest file is
  the 96dc4d84 prompt-variant for all four models, consistent with your examples.
- One defensive measure I'll add anyway: warn (or fail) if the chosen file has fewer
  records than the dataset has conversations, so a future small re-run with a newer
  timestamp can't silently shrink a dataset.

Worth noting for training sizes: Llama-8B prompted_roleplay is the skinniest class
(122 honest conversations); Qwen-32B more-roleplay has only 203 deceptive. Nothing
blocking, just sets the floor for the balanced-n truncation.

## Point 3 — balancing: I understand, three residual sub-questions

My understanding of your scheme (please confirm):

- A switch `balance` with default `"balanced"`.
- **balanced**: all token pools that enter a probe's training set are truncated to the
  same n = min token count over all (dataset × class) pools involved. Single-dataset
  probe: 2 pools; pair probe: 4 pools; triple probe: 6 pools.
- **unbalanced**: per-class n = min over datasets only (original spec formula), so
  honest-n and deceptive-n may differ; single-dataset probes are not truncated at all.
- **Truncation = fixed permutation then prefix**: each (dataset, class) token pool is
  shuffled with a seeded RNG — token-level, conversation grouping deliberately ignored —
  and the first n tokens are taken. Yes, I know exactly what you mean, and token-level
  is the right call since tokens are the training points.

Residual sub-questions:

1. **Is the balancing purely by subsampling**, i.e., sklearn's `class_weight` stays
   `None`? Subsampling to equal n already does the job; additionally setting sklearn
   `class_weight='balanced'` on the truncated set would be a no-op for classes but
   could fight the cross-dataset truncation. My proposal: the switch controls
   subsampling only, sklearn `class_weight=None` always.
2. **Does "balanced" also truncate within single-dataset probes** (honest vs deceptive
   to equal n)? Your "all types/classes and all datasets that are considered" reads as
   yes — assuming yes.
3. **Seed scope**: I propose deriving the permutation seed deterministically from
   (model, dataset, class) — *not* from the layer — so the same tokens are selected at
   every layer. That keeps layers comparable and is also what makes layer-major
   processing cheap. Train split only; the test split stays conversation-grouped and
   untruncated. OK?

## Point 6 — recall@5%FPR: well-defined everywhere, one coarse spot

Definition I'll use: recall at the largest empirical-ROC threshold with FPR ≤ 0.05 (no
interpolation). Honest test-conversation counts at an 80/20 split: more-roleplay ~40–81,
company-assistant ~84–94 → 5% corresponds to 2–4 allowed false positives, fine. The
coarse spot is **Llama-8B prompted_roleplay: ~24 honest test conversations**, so FPR
resolution is ~4.2% and recall@5% means "recall at exactly 1 false positive." Defined,
just noisy — worth remembering when reading that one cell.

## Settled (recording the decisions)

1. Regularization 10.0 = repe convention → sklearn `C = 0.1`.
2. Combined probes: all 3 pairs **and** the triple → **7 probes** per (model, layer)
   (3 single + 3 pair + 1 triple).
3. Balancing: see above (modulo the three sub-questions).
4. Scored-file rule: judge gpt-5.4-mini, timestamped only, latest wins. Verified above.
5. Repe probes from HF: ignored for now.
6. recall@5%FPR instead of 1%.
7. All scores/metrics saved as data for every (model, layer, probe, eval); PNGs
   rendered only for the best layer per probe by AUROC (on its own test split).
8. Fake stories: expected honest by design (fiction on request ≠ deception) — will
   state this in a comment.
9. Trained probes saved (direction, intercept if any, scaler mean/scale, plus config:
   layer, datasets, n per pool, seed).
10. Layers: full <60-layer activations exist on the remote node/HF for all datasets;
    loader will still compute the per-model layer intersection across datasets
    defensively and warn if it's smaller than expected.

If you confirm the three balancing sub-questions (or just say "your proposals are
fine"), the spec side is closed and we can move to the implementation design choices
(memory/mmap strategy, GPU vs sklearn, output layout).
