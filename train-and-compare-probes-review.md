# Review of `train-and-compare-probes.py` (spec)

**Bottom line: the experimental design makes sense** — it's a clean cross-dataset
generalization study for linear deception probes, and `autocomp2/train-probe.ipynb`
already implements maybe 70% of the single-config pipeline (loading `.pt` activations,
scored-file matching, token windowing, padding removal, StandardScaler + LogisticRegression).
Nothing is nonsensical. But there is one factual problem in the data that the spec
doesn't account for, and a handful of ambiguities to resolve before implementation.

## The one real problem: layer availability is inconsistent across datasets

The spec says "for each model we have the activations for multiple layers (at most the
first 60 layers)" and asks for a probe per (model, layer, dataset). But the local
metadata says:

| Dataset            | Llama-8B           | Llama-70B            | Qwen-32B | Qwen-72B             |
|--------------------|--------------------|----------------------|----------|----------------------|
| more-roleplay      | **[8, 12, 16, 24]** | **[20, 30, 40, 60]** | 0–59     | **[20, 30, 40, 60]** |
| ir-honest/dishonest | 0–31              | 0–59                 | 0–59     | 0–59                 |
| company-assistant  | 0–31               | 0–59                 | 0–59     | 0–59                 |

Two consequences:

1. For three of the four models, dataset 1 ("unprompted_roleplay") probes — and all
   combined probes involving dataset 1 — can only exist on at most 4 layers.
2. Llama-70B's and Qwen-72B's layer **60** in more-roleplay doesn't exist in the other
   datasets (they stop at 59), so cross-dataset evaluation there is impossible — the
   usable intersection is {20, 30, 40}.

You said the remote node has all activations, so maybe more-roleplay was re-extracted
with full layers there — but if not, the script needs a defined policy:
**train/evaluate on the intersection of layers available across all datasets for that
model** (my suggestion), and the spec should say so. Worth checking what's actually on
HF before writing the loader.

## Ambiguities to pin down

1. **"Regularization coefficient 10.0"** — `train-probe.ipynb` documents the convention
   `sklearn C = 1 / reg_coeff` (from the repe yaml). So does 10.0 mean `C=0.1` (repe
   convention) or `C=10`? Factor-of-100 difference; needs one sentence in the spec.

2. **Combined datasets: pairs only?** The count "6 probes (3 single + 3 combined)"
   implies pairs only, no triple combination (1+2+3). If you want the triple too, it's 7.
   I'd actually argue the triple is the most interesting "general probe" — consider
   adding it.

3. **The `[:n]` balancing formula is ambiguous twice over.**
   - (a) `n = min(len(ds1[type]), len(ds2[type]))` — is `n` computed separately per type
     (honest-n ≠ deceptive-n) or once over both?
   - (b) The lists are *token*-level, so `[:n]` truncates mid-conversation and takes the
     *first* n tokens, biasing toward early conversations/scenarios. I'd balance at the
     conversation level (or at least subsample tokens randomly with a fixed seed) rather
     than take a prefix.
   - Also: nothing balances honest vs. deceptive *within* a dataset (counts I checked:
     more-roleplay 231/203, company-assistant 418/462 — fine; but ir-honest for
     Llama-70B has 325 HONEST against an unknown number of DECEPTIVE from ir-dishonest —
     worth checking, and deciding whether to use `class_weight='balanced'`).

4. **"Most recent scored file" needs a tie-break rule.** Company-assistant has multiple
   judge-prompt variants distinguished by hash (`0f281a73`, `35699555`, `96dc4d84`, …),
   and both spec examples use `96dc4d84` — should the hash be pinned rather than "latest
   timestamp wins"? There are also files *without* timestamps
   (`...-judge-gpt-5.4-mini.json`). And note the three datasets have three different
   schemas (`verdicts` with `index` vs. `scores` with `k`; `HONEST` vs.
   `honest_authorized` casing) — fine, just needs per-dataset adapters.

5. **The HF repe probes (spec lines 6–7) are mentioned but never used.** Are they
   context only, or should the pre-trained repe probe be a 7th probe in the comparison,
   evaluated on the same test splits? The script title says "compare a lot of different
   probes," so I suspect you want it included — the spec doesn't say.

6. **recall@1%FPR is degenerate at these test-set sizes.** With 80/20 conversation-level
   splits, you'll have ~40–90 honest test conversations per dataset; 1% FPR is below the
   resolution (1 false positive ≈ 1–2.5%), so it collapses to "recall at zero false
   positives." Either accept that, report recall@5% alongside, or compute it token-level.

7. **PNG explosion.** 4 models × up to 60 layers × 6 probes × (3 test sets × 2 plots +
   3 controls × 1 plot) ≈ **13,000 PNGs**. I'd save scores + metrics as JSON for
   everything (cheap, complete) and either generate plots only for a shortlist (e.g.,
   best layer per probe by AUROC) or make plotting a separate post-hoc script.

8. **Fake stories expected "honest"** (spec line 32) — I assume this is intentional
   (fiction-on-request is not deception, and you want the probe *not* to fire), but
   since line 31 says the same thing for true stories it reads like a copy-paste. One
   clarifying word ("even though the content is false") would remove the doubt.

9. **The spec never says to save the trained probes themselves** (direction + scaler
   mean/scale, like the HF probe format). You'll obviously want them — add it.

## Remaining implementation design choices

- **Memory/compute strategy is the biggest one.** A 70B/72B activation file at
  1000 convs × 200 tokens × 60 layers × 8192 dims is ~200 GB in fp16 — `torch.load` of
  the whole thing per the notebook's pattern won't fly times 6 datasets. Options:
  `torch.load(..., mmap=True)` and slice one layer at a time, or loop layer-major (load
  file once, train all 6 probes + evals for every layer before freeing). And ~1,400
  logistic regressions on up to ~150k × 8192 matrices argue for a torch/GPU
  logistic-regression implementation rather than sklearn LBFGS on CPU — or at least
  making that switchable.
- Minor defaults, my suggestions (would flag in code):
  - `fit_intercept=False` (existing convention — keep?)
  - drop all-zero padding rows (notebook already does)
  - score = raw logit projected after the train-set StandardScaler (consistent with
    spec line 57), positive class = deceptive
  - deterministic first-80% split, no shuffle — fine, and nicely consistent across
    models/layers; implicit benefit: ir-honest/ir-dishonest stay scenario-aligned
    across the split
  - fixed seed for anything random
  - decide whether the script should pull missing `.pt`/scored files from the HF repos
    or just error.

If points 1–5 are resolved (layers, C convention, pairs-vs-triple, balancing semantics,
repe-probe inclusion), the rest can be sensible defaults flagged in the code.
