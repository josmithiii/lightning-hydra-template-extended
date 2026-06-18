from pathlib import Path

import pytest
import torch

from src.data.mnist_datamodule import MNISTDataModule
from src.data.mnist_vit_995_datamodule import MNISTViTDataModule


@pytest.mark.parametrize("batch_size", [32, 128])
def test_mnist_datamodule(batch_size: int) -> None:
    """Tests `MNISTDataModule` to verify that it can be downloaded correctly, that the necessary
    attributes were created (e.g., the dataloader objects), and that dtypes and batch sizes
    correctly match.

    :param batch_size: Batch size of the data to be loaded by the dataloader.
    """
    data_dir = "data/"

    dm = MNISTDataModule(data_dir=data_dir, batch_size=batch_size)
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert Path(data_dir, "MNIST").exists()
    assert Path(data_dir, "MNIST", "raw").exists()

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    assert num_datapoints == 70_000

    batch = next(iter(dm.train_dataloader()))
    x, y = batch
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64


def test_mnist_multihead_datamodule() -> None:
    """Tests `MNISTDataModule` in multihead mode end-to-end: `setup()` wraps the data with
    `MultiheadMNISTDataset` and the dataloader yields `(images, dict-of-label-tensors)` via the
    custom collate function.
    """
    dm = MNISTDataModule(data_dir="data/", batch_size=64, multihead=True)
    dm.prepare_data()
    dm.setup()

    assert dm.num_classes == {"digit": 10, "thickness": 5, "smoothness": 3}

    x, y = next(iter(dm.train_dataloader()))
    assert x.shape == (64, 1, 28, 28)
    assert isinstance(y, dict)
    assert set(y.keys()) == {"digit", "thickness", "smoothness"}
    for labels in y.values():
        assert labels.shape == (64,)
        assert labels.dtype == torch.int64


def test_mnist_vit_datamodule() -> None:
    """Tests `MNISTViTDataModule`: split sizes, batch shapes/dtypes, and -- crucially -- that the
    val/test splits are DETERMINISTIC.

    The val/test determinism check is the regression guard for the concat-then-split transform
    bug: train-time `RandomCrop` augmentation must follow the split (train only), not the source
    MNIST train/test dataset. Under the old code, val/test samples sourced from the MNIST train
    set were randomly cropped on every access.
    """
    dm = MNISTViTDataModule(data_dir="data/", batch_size=64)
    dm.prepare_data()
    dm.setup()

    assert dm.data_train and dm.data_val and dm.data_test
    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    assert num_datapoints == 70_000

    x, y = next(iter(dm.train_dataloader()))
    assert x.shape == (64, 1, 28, 28)
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64

    # val/test transforms have no randomness, so repeated access of the same index must be
    # identical. Check several indices since only ~86% of each split is MNIST-train-sourced.
    for ds in (dm.data_val, dm.data_test):
        for i in range(10):
            assert torch.equal(
                ds[i][0], ds[i][0]
            ), "val/test must be deterministic (no train-time augmentation)"
