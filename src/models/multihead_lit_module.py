"""Multi-Head Lightning Module for General-Purpose ML Tasks.

This module provides a flexible LightningModule that supports multiple classification
or regression heads, making it suitable for multi-task learning scenarios.

Features:
- Dynamic head configuration from dataset metadata
- Support for both classification and regression outputs
- Backward compatibility with single-head models
- Configurable loss functions and weights per head
- Automatic metric tracking for each head
"""

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

# Import the actual multihead dataset base class
from ..data.multihead_dataset_base import MultiheadDatasetBase


class MultiheadLitModule(LightningModule):
    """Generic Lightning module for multihead classification and regression tasks.

    This module supports:
    - Multiple classification heads with different numbers of classes
    - Multiple regression heads with different output ranges
    - Dynamic head configuration from dataset metadata
    - Configurable loss functions and weights per head
    - Proper metrics tracking for each head
    - Backward compatibility with single-head models

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        criterion: Optional[torch.nn.Module] = None,
        criteria: Optional[Dict[str, torch.nn.Module]] = None,
        loss_weights: Optional[Dict[str, float]] = None,
        compile: bool = False,
        auto_configure_from_dataset: bool = True,
        output_mode: str = "classification",
    ) -> None:
        """Initialize a `MultiheadLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param criterion: The loss function to use for training (backward compatibility).
        :param criteria: Dict of loss functions for multihead training.
        :param loss_weights: Optional weights for combining losses from different heads.
        :param compile: Whether to compile the model.
        :param auto_configure_from_dataset: Whether to auto-configure heads from dataset.
        :param output_mode: Output mode - "classification" or "regression".
        """
        super().__init__()

        # Store for later use in setup
        self.auto_configure_from_dataset = auto_configure_from_dataset
        self._initial_criteria = criteria
        self._initial_criterion = criterion
        self._initial_loss_weights = loss_weights
        self.output_mode = output_mode

        # Backward compatibility handling
        if criteria is None and criterion is not None:
            criteria = {"head_0": criterion}
        elif criteria is None:
            # Will be configured later in setup() if auto_configure_from_dataset is True
            if not auto_configure_from_dataset:
                raise ValueError(
                    "Must provide either 'criterion' or 'criteria' or set auto_configure_from_dataset=True"
                )
            criteria = {}

        # If auto_configure_from_dataset is True but we have a network with heads_config,
        # initialize criteria based on network heads
        if auto_configure_from_dataset and not criteria and hasattr(net, "heads_config"):
            criteria = {
                head_name: torch.nn.CrossEntropyLoss() for head_name in net.heads_config.keys()
            }

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(
            logger=False, ignore=["net", "criterion", "criteria", "scheduler"]
        )

        self.net = net
        self.criteria = criteria
        self.loss_weights = loss_weights or {name: 1.0 for name in criteria.keys()}
        self.is_multihead = len(criteria) > 1

        # Store scheduler separately since it's not in hparams
        self.scheduler = scheduler

        # Will be initialized in setup()
        self.train_metrics = None
        self.val_metrics = None
        self.test_metrics = None
        self.train_loss = None
        self.val_loss = None
        self.test_loss = None
        self.val_acc_best = None

    def _setup_metrics(self) -> None:
        """Setup metrics based on current network configuration."""
        # Get head configurations
        if hasattr(self.net, "heads_config"):
            head_configs = self.net.heads_config
        else:
            # Fallback for backward compatibility
            head_configs = {"head_0": 10}

        # Metrics for each head
        self.train_metrics = torch.nn.ModuleDict()
        self.val_metrics = torch.nn.ModuleDict()
        self.test_metrics = torch.nn.ModuleDict()

        for head_name, num_classes in head_configs.items():
            if self.output_mode == "regression":
                # For regression, we'll use MAE as the primary metric instead of accuracy
                from torchmetrics.regression import MeanAbsoluteError

                self.train_metrics[f"{head_name}_mae"] = MeanAbsoluteError()
                self.val_metrics[f"{head_name}_mae"] = MeanAbsoluteError()
                self.test_metrics[f"{head_name}_mae"] = MeanAbsoluteError()
            else:
                self.train_metrics[f"{head_name}_acc"] = Accuracy(
                    task="multiclass", num_classes=num_classes
                )
                self.val_metrics[f"{head_name}_acc"] = Accuracy(
                    task="multiclass", num_classes=num_classes
                )
                self.test_metrics[f"{head_name}_acc"] = Accuracy(
                    task="multiclass", num_classes=num_classes
                )

        # Loss tracking
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_acc_best = MaxMetric()

    def _setup_criteria(self) -> None:
        """Setup loss criteria based on current network configuration."""
        if hasattr(self.net, "heads_config"):
            head_configs = self.net.heads_config
        else:
            # Fallback for backward compatibility
            head_configs = {"head_0": 10}

        # Initialize criteria if not already set
        if not self.criteria:
            self.criteria = {}
            for head_name in head_configs.keys():
                if self.output_mode == "regression":
                    self.criteria[head_name] = torch.nn.MSELoss()
                else:
                    self.criteria[head_name] = torch.nn.CrossEntropyLoss()

        # Initialize loss weights if not already set
        if not self.loss_weights:
            self.loss_weights = {name: 1.0 for name in self.criteria.keys()}

        # Update multihead flag
        self.is_multihead = len(self.criteria) > 1

    def _auto_configure_from_dataset(self, dataset: MultiheadDatasetBase) -> None:
        """Auto-configure heads and input channels from dataset metadata.

        :param dataset: The dataset to configure from
        """
        heads_config = dataset.get_heads_config()

        # Auto-configure input channels from dataset image shape
        if hasattr(self.net, "input_channels") and hasattr(dataset, "image_shape"):
            dataset_channels = dataset.image_shape[0] if dataset.image_shape else 3
            if self.net.input_channels != dataset_channels:
                print(
                    f"Auto-configuring network input channels: {self.net.input_channels} -> {dataset_channels}"
                )
                self.net.input_channels = dataset_channels

                # If the network has already been built (has conv layers), we need to rebuild the first layer
                if hasattr(self.net, "conv_layers") and hasattr(self.net.conv_layers, "0"):
                    import torch.nn as nn

                    first_conv = self.net.conv_layers[0]
                    if hasattr(first_conv, "in_channels"):
                        # Rebuild the first convolutional layer with correct input channels
                        new_first_conv = nn.Conv2d(
                            in_channels=dataset_channels,
                            out_channels=first_conv.out_channels,
                            kernel_size=first_conv.kernel_size,
                            stride=first_conv.stride,
                            padding=first_conv.padding,
                            bias=first_conv.bias is not None,
                        )
                        self.net.conv_layers[0] = new_first_conv

        # Update network heads configuration
        if hasattr(self.net, "heads_config"):
            self.net.heads_config = heads_config

            # Update parameter_names if the network has this attribute (needed for regression mode)
            if hasattr(self.net, "parameter_names"):
                self.net.parameter_names = list(heads_config.keys())

            # If the network has a _build_heads method, use it to rebuild heads
            # Call for networks that need dynamic head rebuilding (VisionTransformer, SimpleCNN in regression mode, TestMultiheadNet)
            if hasattr(self.net, "_build_heads") and callable(getattr(self.net, "_build_heads")):
                network_name = type(self.net).__name__
                if (
                    network_name == "VisionTransformer"
                    or (
                        network_name == "SimpleCNN"
                        and getattr(self.net, "output_mode", None) == "regression"
                    )
                    or network_name == "TestMultiheadNet"
                ):
                    self.net._build_heads(heads_config)
                    # Update the network's multihead flag after rebuilding
                    if hasattr(self.net, "is_multihead"):
                        self.net.is_multihead = len(heads_config) > 1
        else:
            # If network doesn't have heads_config, log a warning
            print(
                f"Warning: Network {type(self.net).__name__} doesn't have heads_config attribute"
            )

        # Auto-configure JND-based loss weights if available
        if hasattr(dataset, 'metadata_format') and dataset.metadata_format:
            jnd_weights = self._compute_jnd_weights(dataset.metadata_format)
            if jnd_weights:
                print(f"Auto-configuring JND-based loss weights: {jnd_weights}")
                self.loss_weights.update(jnd_weights)

        # Update criteria if using auto-configuration
        if self.auto_configure_from_dataset:
            # Check if we have hardcoded placeholder heads that need replacement
            placeholder_heads = ["digit", "synth_param1"]
            has_hardcoded_placeholder = False
            for placeholder in placeholder_heads:
                if (
                    self.criteria
                    and placeholder in self.criteria
                    and placeholder not in heads_config
                ):
                    has_hardcoded_placeholder = True
                    break

            if self.criteria and not has_hardcoded_placeholder:
                # If criteria were pre-configured (and not hardcoded placeholder), preserve them and update with parameter ranges
                self._update_criteria_with_parameter_ranges(dataset)
            else:
                # No pre-configured criteria or hardcoded placeholder - replace with dataset heads
                self.criteria = {}
                for head_name in heads_config.keys():
                    if self.output_mode == "regression":
                        # For regression mode, we need parameter ranges
                        param_range = self._get_param_range_for_head(dataset, head_name)
                        self.criteria[head_name] = torch.nn.MSELoss()
                    else:
                        self.criteria[head_name] = torch.nn.CrossEntropyLoss()

            # Update loss weights - reset them if we replaced criteria due to hardcoded placeholder
            if not self.loss_weights or has_hardcoded_placeholder:
                self.loss_weights = {name: 1.0 for name in self.criteria.keys()}

        # Update multihead flag
        self.is_multihead = len(self.criteria) > 1

    def _update_criteria_with_parameter_ranges(self, dataset: MultiheadDatasetBase) -> None:
        """Update existing criteria with parameter ranges from dataset metadata.

        :param dataset: The dataset containing parameter range information
        """
        # This is a placeholder for datasets that have parameter range information
        # In a real implementation, this would update criteria with dataset-specific ranges
        pass

    def _get_param_range_for_head(
        self, dataset: MultiheadDatasetBase, head_name: str
    ) -> Tuple[float, float]:
        """Get parameter range for a specific head from dataset metadata.

        :param dataset: The dataset containing parameter range information
        :param head_name: Name of the parameter head
        :return: Tuple of (min, max) parameter range
        """
        # Default fallback range for regression tasks
        return (0.0, 1.0)

    def _compute_jnd_weights(self, metadata_format: dict) -> dict:
        """Compute JND-based loss weights from dataset metadata.

        Weight each head's loss by the number of JNDs (Just Noticeable Differences).
        For classification: JNDs = 1 + (max-min)/step, where step=1 for discrete classes
        For regression: JNDs = 1 + (max-min)/step, using provided step size

        :param metadata_format: Dataset metadata containing parameter mappings
        :return: Dictionary of head names to JND-based weights
        """
        jnd_weights = {}

        if 'parameter_mappings' not in metadata_format:
            return jnd_weights

        param_mappings = metadata_format['parameter_mappings']

        for param_name, param_info in param_mappings.items():
            param_min = param_info.get('min', 0)
            param_max = param_info.get('max', 1)
            param_step = param_info.get('step', None)
            param_type = param_info.get('type', 'classification')

            # For classification, step is implicitly 1 (each class is one JND)
            if param_type == 'classification' or param_step is None:
                param_step = 1.0

            # Compute number of JNDs: 1 + (max-min)/step
            # The +1 accounts for the fact that with N steps, you have N+1 values
            if param_step > 0:
                num_jnds = 1 + (param_max - param_min) / param_step
                jnd_weights[param_name] = float(num_jnds)
            else:
                # If step is 0, parameter is constant (min == max)
                jnd_weights[param_name] = 1.0

        return jnd_weights

    def _compute_predictions(
        self, logits: torch.Tensor, criterion, head_name: str
    ) -> torch.Tensor:
        """Compute predictions based on loss function type."""
        if self.output_mode == "regression":
            # For regression mode, logits are already the predicted values
            preds = logits.squeeze(-1) if logits.dim() > 1 else logits
        else:
            # Classification: use argmax
            preds = torch.argmax(logits, dim=1)

        return preds

    def forward(self, x: torch.Tensor, auxiliary: Optional[torch.Tensor] = None):
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :param auxiliary: Optional auxiliary tensor.
        :return: A tensor of logits (single head) or dict of logits (multihead).
        """
        if hasattr(self.net, "forward") and "auxiliary" in self.net.forward.__code__.co_varnames:
            return self.net(x, auxiliary)
        else:
            return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        if self.val_loss is not None:
            self.val_loss.reset()
        if self.val_metrics is not None:
            for metric in self.val_metrics.values():
                metric.reset()
        if self.val_acc_best is not None:
            self.val_acc_best.reset()

    def model_step(self, batch):
        """Perform a single model step on a batch of data.

        :param batch: A batch of data containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A dict of predictions per head.
            - A dict of target labels per head.
        """
        if len(batch) == 3:
            x, y, auxiliary = batch
        else:
            # Backward compatibility
            x, y = batch
            auxiliary = None

        logits = self.forward(x, auxiliary)

        if self.is_multihead:
            # Multihead case: y is dict, logits should be dict
            if isinstance(logits, dict) and isinstance(y, dict):
                # True multihead: both logits and targets are dicts
                losses = {}
                for head_name in self.criteria.keys():
                    if head_name in y and head_name in logits:
                        losses[head_name] = self.criteria[head_name](
                            logits[head_name], y[head_name]
                        )
                    else:
                        print(f"Warning: Head {head_name} not found in targets or logits")
            elif not isinstance(logits, dict) and isinstance(y, dict):
                # Regression mode with single tensor output but dict targets: split logits by parameter
                # logits shape: [batch_size, num_parameters]
                losses = {}
                logits_dict = {}
                param_names = list(self.criteria.keys())

                for i, head_name in enumerate(param_names):
                    if i < logits.shape[1] and head_name in y:
                        logits_dict[head_name] = logits[:, i : i + 1]  # Keep shape [batch_size, 1]
                        losses[head_name] = self.criteria[head_name](
                            logits_dict[head_name].squeeze(-1), y[head_name]
                        )
                    else:
                        print(f"Warning: Head {head_name} not found in targets or logits")

                # Update logits to dict format for consistency
                logits = logits_dict
            else:
                # Fallback: if we have multihead criteria but tensor inputs, use first head
                print(
                    f"Warning: Multihead model but tensor inputs - falling back to single head mode"
                )
                head_name = next(iter(self.criteria.keys()))
                if isinstance(logits, dict):
                    logits_single = logits[head_name]
                else:
                    logits_single = logits

                if isinstance(y, dict):
                    y_single = y[head_name]
                else:
                    y_single = y

                loss = self.criteria[head_name](logits_single, y_single)
                preds = {
                    head_name: self._compute_predictions(
                        logits_single, self.criteria[head_name], head_name
                    )
                }
                return loss, preds, {head_name: y_single}

            # Calculate total loss and predictions for multihead case
            if not losses:
                # If no losses were computed (all heads failed), return zero tensor
                device = next(self.parameters()).device
                total_loss = torch.tensor(0.0, device=device, requires_grad=True)
            else:
                total_loss = sum(self.loss_weights[name] * loss for name, loss in losses.items())

            preds = {
                head_name: self._compute_predictions(
                    logits_head, self.criteria[head_name], head_name
                )
                for head_name, logits_head in logits.items()
                if head_name in self.criteria
            }

            # Return targets in dict format
            if isinstance(y, dict):
                return total_loss, preds, y
            else:
                # Convert tensor targets to dict format for consistency
                head_name = next(iter(self.criteria.keys()))
                return total_loss, preds, {head_name: y}
        else:
            # Single head case: y is tensor, logits is tensor (backward compatibility)
            head_name = next(iter(self.criteria.keys()))
            if isinstance(y, dict):
                # Extract single head from dict
                y_single = y[head_name]
            else:
                y_single = y

            if isinstance(logits, dict):
                # Extract single head from dict
                logits_single = logits[head_name]
            else:
                logits_single = logits

            loss = self.criteria[head_name](logits_single, y_single)
            preds = self._compute_predictions(logits_single, self.criteria[head_name], head_name)

            return loss, {head_name: preds}, {head_name: y_single}

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data containing the input tensor of images and target labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        # Ensure metrics are initialized
        if self.train_loss is None:
            self._setup_metrics()

        loss, preds_dict, targets_dict = self.model_step(batch)

        # Update metrics
        self.train_loss(loss)
        for head_name in preds_dict.keys():
            if self.output_mode == "regression":
                if f"{head_name}_mae" in self.train_metrics:
                    self.train_metrics[f"{head_name}_mae"](
                        preds_dict[head_name], targets_dict[head_name]
                    )
            else:
                if f"{head_name}_acc" in self.train_metrics:
                    self.train_metrics[f"{head_name}_acc"](
                        preds_dict[head_name], targets_dict[head_name]
                    )

        # Log metrics
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        for head_name in preds_dict.keys():
            if self.output_mode == "regression":
                if f"{head_name}_mae" in self.train_metrics:
                    metric_name = f"train/{head_name}_mae" if self.is_multihead else "train/mae"
                    self.log(
                        metric_name,
                        self.train_metrics[f"{head_name}_mae"],
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True,
                    )
            else:
                if f"{head_name}_acc" in self.train_metrics:
                    metric_name = f"train/{head_name}_acc" if self.is_multihead else "train/acc"
                    self.log(
                        metric_name,
                        self.train_metrics[f"{head_name}_acc"],
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True,
                    )

        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch, batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data containing the input tensor of images and target labels.
        :param batch_idx: The index of the current batch.
        """
        # Ensure metrics are initialized
        if self.val_loss is None:
            self._setup_metrics()

        loss, preds_dict, targets_dict = self.model_step(batch)

        # Update metrics
        self.val_loss(loss)
        for head_name in preds_dict.keys():
            if self.output_mode == "regression":
                if f"{head_name}_mae" in self.val_metrics:
                    self.val_metrics[f"{head_name}_mae"](
                        preds_dict[head_name], targets_dict[head_name]
                    )
            else:
                if f"{head_name}_acc" in self.val_metrics:
                    self.val_metrics[f"{head_name}_acc"](
                        preds_dict[head_name], targets_dict[head_name]
                    )

        # Log metrics
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        for head_name in preds_dict.keys():
            if self.output_mode == "regression":
                if f"{head_name}_mae" in self.val_metrics:
                    metric_name = f"val/{head_name}_mae" if self.is_multihead else "val/mae"
                    self.log(
                        metric_name,
                        self.val_metrics[f"{head_name}_mae"],
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True,
                    )
            else:
                if f"{head_name}_acc" in self.val_metrics:
                    metric_name = f"val/{head_name}_acc" if self.is_multihead else "val/acc"
                    self.log(
                        metric_name,
                        self.val_metrics[f"{head_name}_acc"],
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True,
                    )

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        if self.output_mode == "regression":
            # For regression mode, track best MAE (lower is better) of primary task
            primary_head = next(iter(self.criteria.keys()))
            if f"{primary_head}_mae" in self.val_metrics:
                mae = self.val_metrics[f"{primary_head}_mae"].compute()
                # Note: for MAE, we want to track the minimum (best), so we negate it
                self.val_acc_best(
                    -mae
                )  # Store negative MAE so MaxMetric tracks the best (lowest) MAE
            self.log("val/mae_best", -self.val_acc_best.compute(), sync_dist=True, prog_bar=True)
        else:
            # For classification mode, track best accuracy
            if self.is_multihead:
                # For multihead, compute average accuracy across all heads
                head_accs = []
                for head_name in self.criteria.keys():
                    if f"{head_name}_acc" in self.val_metrics:
                        acc = self.val_metrics[f"{head_name}_acc"].compute()
                        head_accs.append(acc)

                if head_accs:
                    # Compute weighted average using loss weights
                    weighted_sum = sum(
                        acc * self.loss_weights.get(head_name, 1.0)
                        for acc, head_name in zip(head_accs, self.criteria.keys())
                        if f"{head_name}_acc" in self.val_metrics
                    )
                    total_weight = sum(
                        self.loss_weights.get(head_name, 1.0)
                        for head_name in self.criteria.keys()
                        if f"{head_name}_acc" in self.val_metrics
                    )
                    avg_acc = weighted_sum / total_weight if total_weight > 0 else 0.0
                    self.val_acc_best(avg_acc)

                    # Also log simple average for comparison
                    simple_avg = sum(head_accs) / len(head_accs)
                    self.log("val/avg_acc", simple_avg, sync_dist=True, prog_bar=True)
                    self.log("val/weighted_avg_acc", avg_acc, sync_dist=True, prog_bar=True)
            else:
                # For single head, track the only accuracy
                head_name = next(iter(self.criteria.keys()))
                if f"{head_name}_acc" in self.val_metrics:
                    acc = self.val_metrics[f"{head_name}_acc"].compute()
                    self.val_acc_best(acc)
            self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch, batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data containing the input tensor of images and target labels.
        :param batch_idx: The index of the current batch.
        """
        # Ensure metrics are initialized
        if self.test_loss is None:
            self._setup_metrics()

        loss, preds_dict, targets_dict = self.model_step(batch)

        # Update metrics
        self.test_loss(loss)
        for head_name in preds_dict.keys():
            if self.output_mode == "regression":
                if f"{head_name}_mae" in self.test_metrics:
                    self.test_metrics[f"{head_name}_mae"](
                        preds_dict[head_name], targets_dict[head_name]
                    )
            else:
                if f"{head_name}_acc" in self.test_metrics:
                    self.test_metrics[f"{head_name}_acc"](
                        preds_dict[head_name], targets_dict[head_name]
                    )

        # Log metrics
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        for head_name in preds_dict.keys():
            if self.output_mode == "regression":
                if f"{head_name}_mae" in self.test_metrics:
                    metric_name = f"test/{head_name}_mae" if self.is_multihead else "test/mae"
                    self.log(
                        metric_name,
                        self.test_metrics[f"{head_name}_mae"],
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True,
                    )
            else:
                if f"{head_name}_acc" in self.test_metrics:
                    metric_name = f"test/{head_name}_acc" if self.is_multihead else "test/acc"
                    self.log(
                        metric_name,
                        self.test_metrics[f"{head_name}_acc"],
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True,
                    )

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), test, or
        predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        # Auto-configure from dataset if enabled
        if self.auto_configure_from_dataset and hasattr(self.trainer, "datamodule"):
            if (
                hasattr(self.trainer.datamodule, "data_train")
                and self.trainer.datamodule.data_train is not None
            ):
                dataset = self.trainer.datamodule.data_train
                if isinstance(dataset, MultiheadDatasetBase):
                    self._auto_configure_from_dataset(dataset)

        # Setup criteria and metrics
        self._setup_criteria()
        self._setup_metrics()

        # Compile model if requested
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def get_heads_config(self) -> Dict[str, int]:
        """Get the current heads configuration.

        :return: Dictionary mapping head names to number of classes
        """
        if hasattr(self.net, "heads_config"):
            return self.net.heads_config
        return {}

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the dataset configuration.

        :return: Dictionary with dataset information
        """
        info = {
            "heads_config": self.get_heads_config(),
            "is_multihead": self.is_multihead,
            "auto_configure_from_dataset": self.auto_configure_from_dataset,
            "criteria_keys": list(self.criteria.keys()) if self.criteria else [],
            "loss_weights": self.loss_weights,
        }
        return info


if __name__ == "__main__":
    # Basic test
    print("MultiheadLitModule class created successfully")
    print("Use this module for generic multihead classification and regression tasks")
