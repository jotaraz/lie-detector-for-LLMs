import os
import sys
from pathlib import Path

# Directory this script lives in, so the judge scripts can be found regardless
# of the current working directory.
SCRIPT_DIR = Path(__file__).resolve().parent

modeltag = sys.argv[1]
model = f"google--gemma-{modeltag}it"

#os.system(f'{sys.executable} "{SCRIPT_DIR / "judge_more_roleplay.py"}" data/data-more-roleplay-explicit/{model}.json gpt-5.4-mini')
#os.system(f'{sys.executable} "{SCRIPT_DIR / "judge_company_assistant.py"}" {model}')
os.system(f'{sys.executable} "{SCRIPT_DIR / "judge-ir.py"}" honest-explicit {model}')
os.system(f'{sys.executable} "{SCRIPT_DIR / "judge-ir.py"}" dishonest-explicit {model}')
