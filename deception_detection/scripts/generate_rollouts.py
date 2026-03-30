from pathlib import Path

import fire

from deception_detection.models import ModelName, get_model_and_tokenizer
from deception_detection.repository import DatasetRepository
from deception_detection.rollouts import RolloutGenerator
from deception_detection.utils import REPO_ROOT


def generate_rollouts(
    dataset_partial_id: str,
    num: int = 1,
    use_api: bool = True,
    model_name: ModelName = ModelName.LLAMA_70B_3_3,
    grade_rollouts: bool = True,
    out_file: str | Path | None = None,
    add_to_repo: bool = True,
    max_prompts: int | None = None,
):
    """
    Generate rollouts from a model on a particular dataset and save it to the dataset repository.

    Args:
        dataset_partial_id: The partial id of the dataset to generate rollouts for.
            Includes both dataset base name and variant, eg roleplaying__plain
        num: The number of rollouts to generate per dialogue.
        use_api: Whether to use the together API to get model completions. If false generates
            locally on the gpu.
        model_name: Which model to use for generation
        out_file: The file to save the rollouts to. If None, the default location in data/rollouts
            is used.
        add_to_repo: Whether to add the rollouts to the dataset repository.
        max_prompts: The maximum number of dialogues to generate rollouts for. It's useful to set
            this to a small value for testing. None runs for all dialogues in the dataset.
        grade_rollouts: Whether to grade the rollouts. If false, the rollouts are graded as
            either "honest" or "deceptive" based on the 'expected deception value' taken from the
            original datset.
    """
    if isinstance(model_name, str):
        model_name = ModelName(model_name)
    print(f"[generate_rollouts] Starting rollout generation")
    print(f"[generate_rollouts]   dataset : {dataset_partial_id}")
    print(f"[generate_rollouts]   model   : {model_name.value}")
    print(f"[generate_rollouts]   mode    : {'API (Together)' if use_api else 'local GPU'}")
    print(f"[generate_rollouts]   rollouts per prompt: {num}")
    if max_prompts:
        print(f"[generate_rollouts]   max prompts: {max_prompts}")

    print(f"[generate_rollouts] Loading model and tokenizer...")
    model, tokenizer = get_model_and_tokenizer(model_name, omit_model=use_api)
    print(f"[generate_rollouts] Model loaded.")

    print(f"[generate_rollouts] Loading dataset '{dataset_partial_id}'...")
    data_repository = DatasetRepository()
    dataset = data_repository.get(dataset_partial_id, trim_reasoning=False)[:max_prompts]
    print(f"[generate_rollouts] Dataset loaded: {len(dataset)} prompts.")

    generator = RolloutGenerator.from_dialogue_dataset(dataset, model or model_name, tokenizer)

    if out_file is None:
        out_file = (
            REPO_ROOT / "data" / "rollouts" / f"{dataset_partial_id}__{model_name.value}.json"
        )
    out_file = Path(out_file)
    print(f"[generate_rollouts] Output file: {out_file}")

    print(f"[generate_rollouts] Generating {num} rollout(s) for each of {len(dataset)} prompts...")
    generator.generate(num)
    print(f"[generate_rollouts] Generation complete.")

    if grade_rollouts:
        print(f"[generate_rollouts] Grading rollouts...")
        generator.grade_all_dialogues()
        print(f"[generate_rollouts] Grading complete.")
    else:
        print(f"[generate_rollouts] Skipping grading, using expected deception labels.")
        for rollout in generator.rollout_dialogues:
            grade = rollout.expected_label.value
            rollout.graded_deception = [grade] * len(rollout.output_messages)
            rollout.grader_reasoning = ["Expected deception value"] * len(rollout.output_messages)

    print(f"[generate_rollouts] Saving rollouts to {out_file}...")
    generator.save(out_file)
    print(f"[generate_rollouts] Saved.")

    if add_to_repo:
        data_repository.add(
            dataset_partial_id, model_name.value, str(out_file.relative_to(REPO_ROOT))
        )
        print(f"[generate_rollouts] Registered in dataset_paths.json as '{dataset_partial_id}__{model_name.value}'.")

    generator.print_grading_statistics()
    print(f"[generate_rollouts] Done.")


if __name__ == "__main__":
    fire.Fire(generate_rollouts)
