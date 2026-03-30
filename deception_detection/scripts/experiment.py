from pathlib import Path
from typing import Any, Literal

import fire
import torch
import yaml

from deception_detection.experiment import Experiment, ExperimentConfig
from deception_detection.log import logger
from deception_detection.postprocess import ExperimentPostprocessor
from deception_detection.prompt_scorer import PromptScorerMethod
from deception_detection.results.detection_masks import save_detection_masks
from deception_detection.utils import REPO_ROOT


def run(
    config_file: str | Path = REPO_ROOT / "deception_detection/scripts/configs/repe.yaml",
    **kwargs: dict[str, Any],
):
    config_file = Path(config_file)
    if not config_file.exists():
        config_file = REPO_ROOT / "deception_detection/scripts/configs" / config_file
        if not config_file.exists():
            config_file = config_file.parent / (config_file.name + ".yaml")
        if not config_file.exists():
            raise FileNotFoundError("Config file not found")

    print(f"[experiment] Loading config: {config_file}")
    with open(config_file) as file:
        config_dict = yaml.safe_load(file)

    config_dict |= kwargs
    config = ExperimentConfig(**config_dict)  # type: ignore[reportGeneralTypeIssues]
    print(f"[experiment] Model   : {config.model_name.value}")
    print(f"[experiment] Method  : {config.method}")
    print(f"[experiment] Train   : {config.train_data}")
    print(f"[experiment] Eval    : {config.eval_data}")
    print(f"[experiment] Layers  : {config.detect_layers}")
    print(f"[experiment] Results folder: {config.folder}")

    try:
        print(f"[experiment] Initializing experiment (loading tokenizer, datasets)...")
        experiment = Experiment(config)
        print(f"[experiment] Saving detection masks...")
        save_detection_masks(experiment)
        print(f"[experiment] Training / loading detector...")
        experiment.get_detector()
        print(f"[experiment] Detector ready.")
    except Exception as err:
        print(f"[experiment] ERROR during setup: {err}")
        # cleanup experiment folder and any files inside it
        for f in config.folder.iterdir():
            f.unlink()
        config.folder.rmdir()
        raise err

    print(f"[experiment] Computing eval scores...")
    experiment.get_eval_scores()
    print(f"[experiment] Computing control scores...")
    experiment.get_control_scores()

    max_memory = torch.cuda.max_memory_allocated() / 1024**3  # Convert to GB
    print(f"[experiment] Max GPU memory usage: {max_memory:.2f} GB")
    logger.info(f"Max GPU memory usage: {max_memory:.2f} GB")

    print(f"[experiment] Computing final results...")
    results(str(experiment.cfg.folder), scoring_method="mean")
    print(f"[experiment] Experiment complete: {experiment.cfg.name}")


def rerun(folder: str, scoring_method: Literal["mean", "max", "relu"] = "mean"):
    """Runs experiment from existing folder, reusing saved results when possible."""
    cfg = ExperimentConfig.from_path(Path(folder))
    experiment = Experiment(cfg)
    experiment.get_detector()
    experiment.get_eval_scores()
    experiment.get_control_scores()
    max_memory = torch.cuda.max_memory_allocated() / 1024**3  # Convert to GB
    logger.info(f"Max GPU memory usage: {max_memory:.2f} GB")
    results(str(experiment.cfg.folder), scoring_method=scoring_method)


def recompute_dataset(folder: str, dataset_name: str, shard_size: int = 12):
    cfg = ExperimentConfig.from_path(Path(folder))
    experiment = Experiment(cfg)
    experiment.recompute_scores_on_dataset(dataset_name, shard_size=shard_size)
    results(str(experiment.cfg.folder), scoring_method="mean")


def results(
    folder: str, scoring_method: PromptScorerMethod = "mean", include_black_box: bool = False
):
    cfg = ExperimentConfig.from_path(Path(folder))
    experiment = Experiment(cfg)
    experiment.get_eval_scores(all_tokens=False)
    postprocessor = ExperimentPostprocessor(
        experiment, scoring_method=scoring_method, include_black_box=include_black_box
    )
    postprocessor.run()


def just_black_box_scores(folder: str):
    cfg = ExperimentConfig.from_path(Path(folder))
    experiment = Experiment(cfg)
    experiment.get_black_box_scores()


if __name__ == "__main__":
    fire.Fire(
        {
            "run": run,
            "results": results,
            "rerun": rerun,
            "recompute": recompute_dataset,
            "black_box": just_black_box_scores,
        }
    )
