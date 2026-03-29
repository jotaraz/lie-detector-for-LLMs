import subprocess
import sys

models = ["",
          "_qwen-32b",
          "_gemma3-12b",
          "_gemma-9b",
          "_llama_8b"]

for model in models:
    subprocess.run(
        [sys.executable, "deception_detection/scripts/experiment.py", "run", f"repe{model}"],
        check=True,
    )