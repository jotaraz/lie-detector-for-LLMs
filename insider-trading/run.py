"""Run the insider-trading deception eval on open-weight HF models.

Phases (control via --phase):
  generate : rollouts + GPT-4o grading via inspect_ai + vLLM
             writes <hf-id-with-/-as-->-generations.json
  extract  : residual-stream activations via transformers + forward hooks
             writes <hf-id-with-/-as-->-activations.pt
  all      : both, in order (default)

Examples:
  # Both phases, two models, 100 rollouts each
  python run.py \\
      --models meta-llama/Llama-3.1-8B-Instruct,Qwen/Qwen2.5-32B-Instruct \\
      --n 100 \\
      --openai-api-key sk-... \\
      --hf-token hf_...

  # Only re-extract activations (transcripts already exist)
  python run.py \\
      --models meta-llama/Llama-3.1-8B-Instruct \\
      --phase extract

  # Only generate transcripts (no activations)
  python run.py --models <id> --n 100 --phase generate --openai-api-key sk-...

Requirements:
  pip install vllm                       # generation backend (gen phase only)
  pip install flashinfer                 # required by vLLM for Gemma-2 softcap
  HF_TOKEN env var (or --hf-token) for gated models (Llama, Gemma).
  OPENAI_API_KEY env var (or --openai-api-key) for the GPT-4o grader.

Storage note: activation files can be very large (all layers × all assistant
tokens × all rollouts, bf16). For Llama-70B at n=6000 expect ~hundreds of GB
per model. Reduce --n or --max-layer if needed.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from generate import generate_for_model  # noqa: E402
from extract import extract_for_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--models",
        required=True,
        help="Comma-separated list of HuggingFace model IDs.",
    )
    parser.add_argument(
        "--n", type=int, default=100, help="Number of rollouts per model (generation phase)."
    )
    parser.add_argument(
        "--phase",
        choices=["all", "generate", "extract"],
        default="all",
        help="Which phase(s) to run.",
    )
    parser.add_argument(
        "--openai-api-key",
        default=None,
        help="OpenAI API key for the GPT-4o grader. Falls back to OPENAI_API_KEY env var.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace token for gated models. Falls back to HF_TOKEN env var.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(HERE / "data"),
        help="Directory for generations JSON and activations PT files.",
    )
    parser.add_argument(
        "--variant",
        default="default/default.yaml",
        help="Prompt YAML under data/insider_trading/prompts/ (e.g. 'pressure/no_pressure.yaml').",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="vLLM tensor parallelism. Defaults to number of visible CUDA devices.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="vLLM gpu_memory_utilization (default 0.9).",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Optional override for vLLM max_model_len.",
    )
    parser.add_argument(
        "--max-connections",
        type=int,
        default=32,
        help="inspect_ai max_connections (effective batch size for vLLM is also bounded by this).",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature."
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Per-turn max new tokens during generation.",
    )
    parser.add_argument(
        "--max-layer",
        type=int,
        default=60,
        help="Activation extraction captures layers [0, min(max_layer, depth)).",
    )
    parser.add_argument(
        "--extract-limit",
        type=int,
        default=None,
        help="If set, only extract activations from the first N rollouts of the JSON.",
    )
    args = parser.parse_args()

    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = args.openai_api_key
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = args.hf_token

    if args.phase in ("all", "generate") and "OPENAI_API_KEY" not in os.environ:
        sys.exit(
            "error: generation phase needs OPENAI_API_KEY (pass --openai-api-key or set env)."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        sys.exit("error: --models must list at least one HF model id.")

    for model_id in models:
        print(f"\n=== {model_id} ===")
        if args.phase in ("all", "generate"):
            generate_for_model(
                model_id=model_id,
                n=args.n,
                output_dir=output_dir,
                variant=args.variant,
                tensor_parallel_size=args.tensor_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len,
                max_connections=args.max_connections,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
        if args.phase in ("all", "extract"):
            extract_for_model(
                model_id=model_id,
                output_dir=output_dir,
                max_layer=args.max_layer,
                limit=args.extract_limit,
            )


if __name__ == "__main__":
    main()
