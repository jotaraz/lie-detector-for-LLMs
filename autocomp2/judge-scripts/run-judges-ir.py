import os
import sys
from pathlib import Path

# Directory this script lives in, so the judge scripts can be found regardless
# of the current working directory.
SCRIPT_DIR = Path(__file__).resolve().parent

models = ["llama8", "llama70", "qwen32", "qwen72"]

for model in models:
    os.system(f'{sys.executable} "{SCRIPT_DIR / "judge_company_assistant.py"}" {model}')
    #os.system(f'{sys.executable} "{SCRIPT_DIR / "judge_company_assistant.py"}" {model}')
