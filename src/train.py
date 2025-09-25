# Disable PyTorch 2.6 weights_only restriction for trusted LOCAL checkpoints
import os.path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf, open_dict

_original_torch_load = torch.load


def _patched_torch_load(
    f, map_location=None, pickle_module=None, weights_only=None, mmap=None, **kwargs
):
    # Only allow loading from local files, not URLs
    if isinstance(f, str):
        if f.startswith(("http://", "https://", "ftp://", "ftps://")):
            raise ValueError(f"Remote checkpoint loading not allowed for security: {f}")
        if not os.path.isfile(f):
            raise FileNotFoundError(f"Checkpoint file not found: {f}")
    # Force weights_only=False for trusted local research checkpoints
    return _original_torch_load(
        f,
        map_location=map_location,
        pickle_module=pickle_module,
        weights_only=False,
        mmap=mmap,
        **kwargs,
    )


torch.load = _patched_torch_load

# Also patch Lightning's internal checkpoint loading
try:
    from lightning.fabric.utilities import cloud_io

    cloud_io._load = _patched_torch_load
except ImportError:
    pass

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)
from src.utils.architecture_utils import ArchitectureMetadataExtractor
from src.utils.preflight import check_label_diversity
from src.utils.vimh_config import configure_model_from_metadata

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    # Ensure data is prepared before model auto-configuration
    if "vimh" in cfg.data._target_.lower():
        try:
            datamodule.prepare_data()
        except Exception as e:
            log.warning(f"Failed to prepare data during auto-configuration: {e}")

    # For VIMH datasets, configure model with parameter names from metadata
    if (
        "vimh" in cfg.data._target_.lower()
        and hasattr(cfg.model, "auto_configure_from_dataset")
        and cfg.model.auto_configure_from_dataset
    ):
        try:
            configure_model_from_metadata(cfg, cfg.data.data_dir)
        except Exception as e:
            log.warning(f"Failed to auto-configure model from dataset metadata: {e}")

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    # Log important model configuration details
    if hasattr(model, "output_mode"):
        log.info(f"Model output mode: {model.output_mode}")
    if hasattr(model, "criteria") and model.criteria:
        criteria_info = {
            name: type(criterion).__name__ for name, criterion in model.criteria.items()
        }
        log.info(f"Model loss functions: {criteria_info}")
    if hasattr(model, "is_multihead"):
        log.info(f"Model is multihead: {model.is_multihead}")

    # Log loss type for extract_logs.py parsing
    loss_type = "unknown"
    if hasattr(cfg.model, "loss_type") and cfg.model.loss_type:
        loss_type = cfg.model.loss_type
    elif hasattr(model, "criteria") and model.criteria:
        # Try to infer from criterion types
        criterion_types = [type(c).__name__ for c in model.criteria.values()]
        if "OrdinalRegressionLoss" in criterion_types:
            loss_type = "ordinal_regression"
        elif "CrossEntropyLoss" in criterion_types:
            loss_type = "cross_entropy"
        elif "MSELoss" in criterion_types or "L1Loss" in criterion_types:
            loss_type = "normalized_regression"
    log.info(f"EXPERIMENT CONFIG: loss_type={loss_type}")

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    # Extract and store architecture metadata for checkpoint reconstruction
    if hasattr(model, "net"):
        metadata_extractor = ArchitectureMetadataExtractor()
        metadata_extractor.extract_and_store_metadata(model, datamodule)

    if cfg.get("train"):
        # Preflight: ensure label diversity across a few batches before fitting
        enabled = True
        batches = 3
        try:
            if hasattr(cfg, "preflight"):
                enabled = getattr(cfg.preflight, "enabled", True)
                batches = getattr(cfg.preflight, "label_diversity_batches", 3)
        except Exception:
            pass

        if enabled:
            try:
                check_label_diversity(datamodule, max_batches=int(batches))
                log.info("Label preflight passed (diverse targets across heads)")
            except Exception as e:
                log.error(f"Label preflight failed: {e}")
                raise
        else:
            log.info("Preflight checks disabled via config")

        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

        # Log actual epochs completed for extract_logs.py parsing
        actual_epochs = trainer.current_epoch + 1 if trainer.current_epoch >= 0 else 0
        log.info(f"TRAINING COMPLETED: actual_epochs={actual_epochs}")

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = cfg.get("ckpt_path") or trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # Print the key configs being used
    log.info("=" * 60)
    # Extract config names from hydra context
    try:
        from hydra.core.hydra_config import HydraConfig

        hydra_cfg = HydraConfig.get()
        model_config = hydra_cfg.runtime.choices.get("model", "unknown")
        data_config = hydra_cfg.runtime.choices.get("data", "unknown")
        trainer_config = hydra_cfg.runtime.choices.get("trainer", "unknown")
    except:
        # Fallback if hydra context not available
        model_config = "unknown"
        data_config = "unknown"
        trainer_config = "unknown"

    log.info(f"MODEL CONFIG:     {model_config} ({cfg.model._target_})")
    data_dir = getattr(cfg.data, "data_dir", "unknown")
    # Show relative path if it's under project root
    import os

    if data_dir != "unknown" and os.path.isabs(data_dir):
        try:
            data_dir = os.path.relpath(data_dir)
        except:
            pass  # Keep original if relpath fails
    log.info(f"DATA CONFIG:      {data_config} (data_dir={data_dir})")
    log.info(f"TRAINER CONFIG:   {trainer_config} ({cfg.trainer._target_})")
    if cfg.get("experiment"):
        log.info(f"EXPERIMENT:       {cfg.experiment}")
    else:
        log.info(f"EXPERIMENT:       none")
    log.info(f"TAGS:             {cfg.get('tags', 'none')}")
    if cfg.get("seed"):
        log.info(f"SEED:             {cfg.seed}")
    log.info("=" * 60)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
