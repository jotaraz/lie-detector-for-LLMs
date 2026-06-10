# Memory/loading strategies in detail (design choice 2)

## The sizes we're dealing with

Activation files (fp16, full layer sets as on the remote node):

| Model | more-roleplay | ir-honest | ir-dishonest | company | true | fake | **total** |
|---|---|---|---|---|---|---|---|
| Llama-8B (d=4096, L=32) | 26 | 19 | 19 | 52 | 26 | 26 | **170 GB** |
| Qwen-32B (d=5120, L=60) | 61 | 46 | 46 | 123 | 61 | 61 | **398 GB** |
| Llama-70B (d=8192, L=60) | 98 | 73 | 73 | 197 | 98 | 98 | **637 GB** |
| Qwen-72B (d=8192, L=60) | 98 | 73 | 73 | 197 | 98 | 98 | **637 GB** |

What we actually need at any moment (layer-major loop): **one layer slice across all 6
datasets** = 648k token slots × d × 4 bytes fp32 ≈ **11 GB (8B) to 21 GB (70B/72B)** —
less after dropping padding and unlabeled conversations.

So the problem is purely "how to get one layer out of a 100–200 GB file without
loading the whole file."

## Option A — naive `torch.load` of whole files (the notebook pattern)

Load each `.pt` fully into RAM, index layers from there.

- **Pro:** simplest possible code; zero new concepts; after the one load, every layer
  access is free (RAM-speed), one single sequential disk read per file.
- **Con:** needs all 6 files of a model in RAM at once for the layer-major loop —
  **637 GB for the 70B/72B models**. Only works if the node has ~700+ GB RAM and
  nothing else running. Even then, the load itself reads 637 GB from disk before any
  compute starts.
- **Variant A′:** load one dataset at a time and loop datasets-outer/layers-inner
  instead. Cuts peak RAM to the biggest single file (197 GB) but breaks the layer-major
  structure: each probe needs all datasets for its layer, so you'd have to cache
  per-layer slices of earlier datasets anyway — you end up rebuilding option D with
  extra steps.

**Verdict:** fine for Llama-8B on a fat node, not viable for the big models.

## Option B — `torch.load(..., mmap=True)` (recommended)

PyTorch memory-maps the tensor storage; the file *appears* as a normal tensor but
pages are read from disk only when touched. Slicing `acts[:, 0, :, layer_idx, :]`
then copying (`.float()` / `.clone()`) materializes just that layer.

- **Pro:** near-zero RAM overhead beyond the one materialized layer; the code looks
  identical to option A (it's still just tensor indexing); no preprocessing, no extra
  disk space; resume/partial runs need no special handling.
- **Con:** the access pattern is strided — for one layer you read every (conv, token)
  position's d-vector, i.e., ~1/L of the file scattered across it. Each read chunk is
  d × 2 bytes = 8–16 KB, which is large enough that NVMe handles it well, but on slow
  or network storage this could crawl. Re-reading happens once per layer (the OS page
  cache won't hold a 200 GB file), so the total I/O over a full run ≈ one full read of
  every file — same total bytes as option A, just spread out.
- **Requirements:** files must be in the zipfile serialization format (`torch.save`
  default since torch 1.6 — almost certainly true here, but the script should check
  and fall back loudly if `mmap=True` fails).

**Throughput arithmetic:** per layer, the slices across all 6 datasets of a 70B model
are ~11 GB of fp16 reads. At ~2 GB/s (ordinary NVMe) that's ~6 s/layer → **~6 min of
I/O per model for all 60 layers**. Completely dwarfed by training time. Even at
500 MB/s (bad disk) it's ~25 min/model — still acceptable.

**Verdict:** the right default. Simple, no preprocessing, and the I/O math says it's
fast enough unless the storage is unusually slow.

## Option C — one-time conversion to per-layer files

A preprocessing pass rewrites each `.pt` into layer-major storage: either one file per
(model, dataset, layer), or one HDF5/zarr per (model, dataset) chunked along the layer
axis. The training loop then reads exactly one small contiguous file per layer.

- **Pro:** fastest possible per-layer access (pure sequential reads); trivially
  parallelizable across layers/GPUs later; and the conversion pass can *also* drop
  padding rows and unlabeled conversations, shrinking storage a lot (likely 2–3×:
  padding looks substantial with m=200, and e.g. more-roleplay keeps only ~600/1000
  conversations after label filtering). Re-runs (different reg, different balance
  mode) get cheap forever.
- **Con:** the conversion itself must stream through every file once (it would itself
  use mmap — so option B's machinery is needed anyway); ~1.8 TB read + up to ~1.8 TB
  written; temporarily ~doubles disk usage unless originals are deleted; more code,
  more failure modes, and a second on-disk format to keep consistent with the source
  data (stale-cache bugs).

**Verdict:** the right move only if (a) option B turns out I/O-bound on the actual
node, or (b) we expect to re-run the grid many times. Not worth it for run #1. The
script structure (a `load_layer(model, dataset, layer)` function) keeps the door open:
option C would be a drop-in alternative backend for that one function.

## Option D — layer batching: mmap + materialize k layers at a time

Same as B, but each pass through a file extracts a block of k layers (e.g., k=10) into
RAM, trains them all, then moves on.

- **Pro:** cuts the number of strided passes over each file from L to L/k; useful if
  the storage hates the strided pattern. RAM cost = k × per-layer size (k=10 → ~110 GB
  for 70B fp16 slices — so realistically k=3–5 on a 256 GB node).
- **Con:** complicates the loop and the resume logic for a benefit that the
  arithmetic above suggests we don't need; peak-RAM pressure reintroduces exactly the
  problem we were avoiding.

**Verdict:** keep in the back pocket; it's a 20-line change to the layer loop if
option B proves slow. Not the default.

## Orthogonal sub-choice: where the materialized layer lives

Per layer we hold ~11–21 GB fp32 of token matrices (before padding drop; less after).

- **GPU-resident (recommended on an 80 GB card):** scaler stats, subsampling, LBFGS
  fits, and all eval projections happen on-device; each probe fit is seconds; the 7
  probes per layer reuse the same resident matrices.
- **CPU-resident with GPU fits:** if the GPU is smaller (e.g., 40 GB), keep the full
  slices on CPU and ship only each probe's (subsampled) train matrix + eval batches to
  GPU. Slightly slower, much more forgiving. Controlled by a `--device`/`--keep-on-gpu`
  flag rather than two code paths: tensors just live where the flag says.

## Recommendation, condensed

**Option B (mmap) + GPU-resident layer slices**, behind a single
`load_layer(model, dataset, layer)` function; startup check that `mmap=True` works on
the actual files; `--layer-batch k` (option D) and a future conversion pass (option C)
as escape hatches if the node's storage disappoints. Option A is what we silently get
for free on small files anyway (mmap of a 26 GB file the OS can cache ≈ A's behavior).

## Update to design choice 4 (token scores), for the record

Recomputed: token-level scores for the **entire grid** (212 (model, layer) blocks × 7
probes × ~650k eval tokens) are **~0.8 GB in fp16 / 1.6 GB in fp32** stored as compact
`.pt` tensors — ~4–8 MB per (model, layer) block. The earlier "tens of GB" figure was
wrong (it implicitly assumed JSON's ~10× text overhead). So: **token scores are saved
by default** as one `token_scores.pt` per (model, layer) block (dict: probe →
eval_dataset → flat fp16 tensor + conv-id index tensor), and the per-conversation
means stay in the human-readable `eval_*.json`. The `--save-token-scores` flag is
dropped.
