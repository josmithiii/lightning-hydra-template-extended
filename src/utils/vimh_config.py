from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf, open_dict

from . import pylogger
from .vimh_utils import (
    get_heads_config_from_metadata,
    get_image_dimensions_from_metadata,
    get_parameter_names_from_metadata,
    get_parameter_ranges_from_metadata,
)

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def configure_model_from_metadata(cfg: DictConfig, data_dir: str) -> None:
    """Mutate Hydra model config based on VIMH metadata."""
    if not data_dir:
        raise ValueError("VIMH auto-configuration requires data_dir to be set")

    parameter_names = get_parameter_names_from_metadata(data_dir)
    if not parameter_names:
        log.warning("No VIMH parameter names found; skipping auto-configuration")
        return

    _configure_input_channels(cfg, data_dir)

    if getattr(cfg.model, "output_mode", "classification") == "regression":
        _configure_regression(cfg, data_dir, parameter_names)
    else:
        heads_config = get_heads_config_from_metadata(data_dir)
        cfg.model.net.heads_config = heads_config

    if not getattr(cfg.model, "loss_weights", None):
        cfg.model.loss_weights = {name: 1.0 for name in parameter_names}
        log.info(f"Auto-configured loss_weights: {cfg.model.loss_weights}")


def _configure_input_channels(cfg: DictConfig, data_dir: str) -> None:
    net = getattr(cfg.model, "net", None)
    if not net or not hasattr(net, "input_channels"):
        return
    height, width, channels = get_image_dimensions_from_metadata(data_dir)
    if channels and net.input_channels != channels:
        log.info(
            f"Auto-configuring network input channels: {net.input_channels} -> {channels}"
        )
        net.input_channels = channels


def _configure_regression(cfg: DictConfig, data_dir: str, parameter_names: Any) -> None:
    net = getattr(cfg.model, "net", None)
    if not net:
        return
    net.parameter_names = parameter_names
    net.output_mode = "regression"
    net.heads_config = None

    param_ranges = get_parameter_ranges_from_metadata(data_dir)
    criteria_cfg = {}
    for name in parameter_names:
        rng = param_ranges.get(name, (0.0, 1.0))
        criteria_cfg[name] = {
            "_target_": "src.models.losses.NormalizedRegressionLoss",
            "param_range": (float(rng[0]), float(rng[1])),
        }
    with open_dict(cfg.model):
        cfg.model.criteria = OmegaConf.create(criteria_cfg)
    log.info(f"Auto-configured regression loss functions for: {list(criteria_cfg.keys())}")
