import os 


models = ["llama8", "llama70", "qwen32", "qwen72"]

for model in models:
    os.system(f"python judge_company_assistant.py {model}")
    #os.system(f"python judge_company_assistant.py {model}")
