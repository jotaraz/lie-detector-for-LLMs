import os
import sys

# Everything is referenced relative to where this file sits, so the script
# works on any node regardless of the current working directory or where the
# repo is checked out. The interpreter used is whatever runs this file
# (e.g. `.venv/bin/python run.py`), so no venv path needs to be hardcoded.
HERE = os.path.dirname(os.path.abspath(__file__))
GEN_SCRIPT = os.path.join(HERE, "generate_text_and_activations.py")

models = [
	  #   'meta-llama/Llama-3.3-70B-Instruct',
      #   'meta-llama/Llama-3.1-8B-Instruct',
	  # 'Qwen/Qwen2.5-32B-Instruct',
	  'google/gemma-2-9b-it',
	  'google/gemma-3-12b-it',
	  'google/gemma-3-27b-it',
	  # 'Qwen/Qwen2.5-72B-Instruct',
	]

scenarios = ["more-roleplay-explicit", "ir-honest-explicit", "ir-dishonest-explicit"]

for model in models:
	for scenario in scenarios:
		jsonl = os.path.join(HERE, "scenarios", f"{scenario}.jsonl")
		os.system(f"{sys.executable} {GEN_SCRIPT} --model {model} --jsonl {jsonl}")
