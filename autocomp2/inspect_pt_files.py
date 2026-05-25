import sys
from pathlib import Path
import torch


def describe(obj, indent=2):
    pad = " " * indent
    if isinstance(obj, torch.Tensor):
        return f"Tensor(shape={tuple(obj.shape)}, dtype={obj.dtype}, device={obj.device})"
    if isinstance(obj, dict):
        lines = ["{"]
        for k, v in obj.items():
            lines.append(f"{pad}{k!r}: {describe(v, indent + 2)},")
        lines.append(" " * (indent - 2) + "}")
        return "\n".join(lines)
    if isinstance(obj, (list, tuple)):
        type_name = "list" if isinstance(obj, list) else "tuple"
        lines = [f"{type_name}(len={len(obj)})["]
        for i, v in enumerate(obj):
            lines.append(f"{pad}[{i}]: {describe(v, indent + 2)},")
        lines.append(" " * (indent - 2) + "]")
        return "\n".join(lines)
    return f"{type(obj).__name__}: {obj!r}"


def main(directory: str) -> None:
    root = Path(directory)
    pt_files = sorted(root.glob("*.pt"))
    if not pt_files:
        print(f"No .pt files found in {root}")
        return
    for path in pt_files:
        print(f"=== {path.name} ===")
        try:
            obj = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"  ERROR loading: {e}")
            continue
        print(describe(obj))
        print()


if __name__ == "__main__":
    directory = sys.argv[1] if len(sys.argv) > 1 else "."
    main(directory)
