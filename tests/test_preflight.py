import pytest
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset

from src.utils.preflight import check_label_diversity


class _TensorDataModule(LightningDataModule):
    def __init__(self, labels, label_mode="classification"):
        super().__init__()
        self.save_hyperparameters()
        self._dataset = TensorDataset(torch.zeros(len(labels), 1, 28, 28), labels)

    def prepare_data(self):
        return None

    def setup(self, stage=None):  # pragma: no cover - simple no-op
        pass

    def train_dataloader(self):
        return DataLoader(self._dataset, batch_size=2, shuffle=False)


class _RegressionDataModule(_TensorDataModule):
    def __init__(self):
        super().__init__(labels=torch.zeros(4, dtype=torch.float32), label_mode="regression")
        self.hparams.label_mode = "regression"


def test_check_label_diversity_passes_with_varied_labels():
    labels = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    dm = _TensorDataModule(labels)
    check_label_diversity(dm, max_batches=2)


def test_check_label_diversity_fails_on_constant_labels():
    labels = torch.zeros(4, dtype=torch.long)
    dm = _TensorDataModule(labels)
    with pytest.raises(ValueError):
        check_label_diversity(dm, max_batches=2)


def test_check_label_diversity_skips_regression_mode(caplog):
    import logging
    caplog.set_level(logging.INFO)
    dm = _RegressionDataModule()
    check_label_diversity(dm)
    assert any("regression label mode" in record.message for record in caplog.records)
