import os

models = ["meta-llama--Llama-3.1-8B-Instruct",
          #"google--gemma-3-27b-it",
          #"Qwen--Qwen2.5-72B-Instruct"
         ]

for model in models:
    os.system(f"python judge-ir.py honest {model} 20")
    os.system(f"python judge-ir.py dishonest {model} 20")