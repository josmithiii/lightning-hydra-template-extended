from __future__ import annotations

from typing import Dict

import torch
from lightning import LightningDataModule

from .pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def check_label_diversity(datamodule: LightningDataModule, max_batches: int = 3) -> None:
    """Validate that categorical labels vary across a few batches per head."""
    # Lightning datamodules may not be set up yet; attempt to prepare.
    try:
        datamodule.prepare_data()
        datamodule.setup("fit")
    except Exception as exc:  # pragma: no cover - informational
        log.warning(f"DataModule setup failed during preflight: {exc}")

    # Skip regression-style modules where label diversity does not apply.
    label_mode = getattr(getattr(datamodule, "hparams", object()), "label_mode", "classification")
    if isinstance(label_mode, str) and label_mode.lower() == "regression":
        log.info("Preflight skipped: regression label mode (continuous targets)")
        return

    loader = datamodule.train_dataloader()
    it = iter(loader)
    uniques: Dict[str, set] = {}
    sampled = 0

    while sampled < max_batches:
        try:
            batch = next(it)
        except StopIteration:
            break
        sampled += 1

        if not batch:
            continue
        data, labels = batch[0], batch[1]
        _record_label_diversity(labels, uniques)

    _log_label_preview(sampled, uniques)

    problems = [head for head, values in uniques.items() if len(values) <= 1]
    if problems:
        details = ", ".join(f"{h}: {sorted(list(uniques[h]))}" for h in problems)
        raise ValueError(
            "Label preflight failed: non-diverse targets for heads "
            f"[{', '.join(problems)}]. Observed unique labels across {sampled} batch(es): {details}. "
            "This often indicates label decoding issues."
        )


def _record_label_diversity(labels, uniques: Dict[str, set]) -> None:
    if isinstance(labels, dict):
        for head, tensor in labels.items():
            uniques.setdefault(head, set())
            _update_uniques(tensor, uniques[head])
    else:
        uniques.setdefault("main", set())
        _update_uniques(labels, uniques["main"])


def _update_uniques(tensor: torch.Tensor, dest: set) -> None:
    if not isinstance(tensor, torch.Tensor):
        return
    if tensor.ndim != 1 or tensor.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        return
    dest.update(tensor.tolist())


def _log_label_preview(sampled: int, uniques: Dict[str, set]) -> None:
    for head in sorted(uniques.keys()):
        values = sorted(list(uniques[head]))
        preview = ", ".join(map(str, values[:10]))
        if len(values) > 10:
            preview += " …"
        log.info(
            f"Preflight head '{head}': {len(values)} unique label(s) across {sampled} batch(es): [{preview}]"
        )
