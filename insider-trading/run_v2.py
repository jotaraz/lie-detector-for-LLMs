"""Run the v2 insider-trading generation (generate + judge in one fresh pass).

Implements insider-trading/how-this-will-be.md via generate_v2.py. Generation
only — no extract / rejudge phases. Writes to <output-dir> (default data/v2/).

Example:
  python run_v2.py --models meta-llama/Llama-3.1-8B-Instruct --n 100
"""
from __future__ import annotations

from filelock import SoftFileLock
import filelock

filelock.FileLock = SoftFileLock

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
ENV_FILE = REPO_ROOT / ".env"
sys.path.insert(0, str(HERE))

if "HF_TOKEN" in os.environ and "HUGGING_FACE_HUB_TOKEN" not in os.environ:
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

from generate_v2 import DEFAULT_GRADER_MODEL, generate_for_model  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", required=True, help="Single HuggingFace model id.")
    p.add_argument("--n", type=int, default=100, help="Rollouts (epochs).")
    p.add_argument("--grader-model", default=DEFAULT_GRADER_MODEL)
    p.add_argument("--output-dir", default=str(HERE / "data" / "v2"))
    p.add_argument("--variant", default="default/default.yaml")
    p.add_argument("--tensor-parallel-size", type=int, default=None)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--quantization", default=None)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--max-connections", type=int, default=32)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=512)
    args = p.parse_args()

    if "OPENAI_API_KEY" not in os.environ:
        sys.exit(f"error: generate phase needs OPENAI_API_KEY (set it in {ENV_FILE} / environment).")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if len(models) != 1:
        sys.exit("error: pass exactly one model id via --models (one job per model).")
    model_id = models[0]

    output_dir = Path(args.output_dir)
    print(f"\n=== v2 generate: {model_id} (n={args.n}) -> {output_dir} ===")
    generate_for_model(
        model_id=model_id,
        n=args.n,
        output_dir=output_dir,
        variant=args.variant,
        grader_model=args.grader_model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        quantization=args.quantization,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        max_connections=args.max_connections,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
