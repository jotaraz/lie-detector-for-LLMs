models = [
	  'meta-llama/Llama-3.1-8B-Instruct',
	  'Qwen/Qwen2.5-32B-Instruct',
	  'google/gemma-2-9b-it',
	  'google/gemma-3-12b-it',
	  'meta-llama/Llama-3.3-70B-Instruct',
	  'Qwen/Qwen2.5-72B-Instruct',
	]

import os

for model in models:
	os.system(f"/home/ubuntu/lie-detector-for-LLMs/.venv/bin/python generate_text_and_activations.py --model {model}")
