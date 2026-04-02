
probes = [
        #"repe_lr__repe_honesty__plain__llama-70b-3.3__20260330_091722",
        "repe_lr__repe_honesty__plain__llama-70b__20260401_103304",
        "repe_lr__repe_honesty__plain__llama-8b__20260330_135003",
        "repe_lr__repe_honesty__plain__gemma3-12b__20260330_104334",
        "repe_lr__repe_honesty__plain_with_sys__qwen-32b__20260330_110159",
        ]


import os 

for probe in probes:
    print(probe)
    os.system(f"/home/ubuntu/lie-detector-for-LLMs/.venv/bin/python tryprobe.py --probes results/{probe} --conversations example_conversations.json")
