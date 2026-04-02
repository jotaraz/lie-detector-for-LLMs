import shutil
from pathlib import Path
import re

CONFIG_DIR = Path("deception_detection/scripts/configs")

CONFIG_FILES = [
            "repe_llama-8b_layer8.yaml",
            "repe_llama-8b_layer16.yaml",
            "repe_gemma-9b_layer10.yaml",
            "repe_gemma-9b_layer21.yaml",
            "repe_gemma3-12b_layer12.yaml",
            "repe_gemma3-12b_layer24.yaml",
            "repe_qwen-32b_layer16.yaml",
            "repe_qwen-32b_layer32.yaml",
        ]



REG_COEFFS = [0.1, 1, 3, 30, 100]

new_files = []

for config_file in CONFIG_FILES:
    src = CONFIG_DIR / config_file
    content = src.read_text()

    stem = src.stem  # filename without extension

    for reg_coeff in REG_COEFFS:
        # Replace the reg_coeff value in the content
        new_content = re.sub(
            r"^(reg_coeff:\s*)[\d.]+",
            f"\\g<1>{reg_coeff}",
            content,
            flags=re.MULTILINE,
        )

        # Format reg_coeff for filename (avoid dots for cleanliness)
        reg_str = str(reg_coeff).replace(".", "p")
        new_name = f"{stem}_reg{reg_str}.yaml"
        dst = CONFIG_DIR / new_name

        new_files.append(new_name)

        dst.write_text(new_content)
        print(f"Created {dst}")


print('')
for file in new_files:
    print(f'    "{file}",')