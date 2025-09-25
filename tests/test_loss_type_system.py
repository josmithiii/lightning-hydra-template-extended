"""Tests for the enhanced loss_type system in VIMHLitModule."""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock

from src.models.components.simple_cnn import SimpleCNN
from src.models.losses import (
    NormalizedRegressionLoss,
    OrdinalRegressionLoss,
    QuantizedRegressionLoss,
    WeightedCrossEntropyLoss,
)
from src.models.soft_target_loss import SoftTargetLoss
from src.models.vimh_lit_module import VIMHLitModule


class MockDataset:
    """Mock dataset for testing auto-configuration."""

    def __init__(self, param_ranges=None, metadata=None):
        self.param_ranges = param_ranges or {"param_0": 10.0, "param_1": 5.0}
        self._metadata = metadata or {
            "parameter_mappings": {
                "param_0": {"min": 0, "max": 10},
                "param_1": {"min": 5, "max": 10},
            }
        }

    def get_heads_config(self):
        return {"param_0": 256, "param_1": 128}


class TestLossTypeSystem:
    """Test the enhanced loss_type system."""

    def setup_method(self):
        """Setup test components."""
        self.net = SimpleCNN(
            input_channels=3,
            conv1_channels=32,
            conv2_channels=64,
            fc_hidden=128,
            heads_config={"param_0": 256, "param_1": 128},
            input_size=32,
        )
        self.optimizer = torch.optim.Adam
        self.scheduler = lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

    def test_create_loss_function_cross_entropy(self):
        """Test creating cross entropy loss function."""
        module = VIMHLitModule(
            net=self.net,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            loss_type="cross_entropy",
            auto_configure_from_dataset=False,
        )

        loss_fn = module._create_loss_function("cross_entropy", num_classes=10, param_range=1.0)

        assert isinstance(loss_fn, nn.CrossEntropyLoss)

    def test_create_loss_function_ordinal_regression(self):
        """Test creating ordinal regression loss function."""
        module = VIMHLitModule(
            net=self.net,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            loss_type="ordinal_regression",
            auto_configure_from_dataset=False,
        )

        loss_fn = module._create_loss_function("ordinal_regression", num_classes=256, param_range=10.0)

        assert isinstance(loss_fn, OrdinalRegressionLoss)
        assert loss_fn.num_classes == 256
        assert loss_fn.param_range == 10.0
        assert loss_fn.regression_loss == "l1"
        assert loss_fn.alpha == 0.1

    def test_create_loss_function_quantized_regression(self):
        """Test creating quantized regression loss function."""
        module = VIMHLitModule(
            net=self.net,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            loss_type="quantized_regression",
            auto_configure_from_dataset=False,
        )

        loss_fn = module._create_loss_function("quantized_regression", num_classes=256, param_range=5.0)

        assert isinstance(loss_fn, QuantizedRegressionLoss)
        assert loss_fn.num_classes == 256
        assert loss_fn.param_range == 5.0
        assert loss_fn.loss_type == "l1"

    def test_create_loss_function_weighted_cross_entropy(self):
        """Test creating weighted cross entropy loss function."""
        module = VIMHLitModule(
            net=self.net,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            loss_type="weighted_cross_entropy",
            auto_configure_from_dataset=False,
        )

        loss_fn = module._create_loss_function("weighted_cross_entropy", num_classes=128, param_range=1.0)

        assert isinstance(loss_fn, WeightedCrossEntropyLoss)
        assert loss_fn.num_classes == 128
        assert loss_fn.distance_power == 2.0
        assert loss_fn.base_weight == 1.0

    def test_create_loss_function_soft_target(self):
        """Test creating soft target loss function."""
        module = VIMHLitModule(
            net=self.net,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            loss_type="soft_target",
            auto_configure_from_dataset=False,
        )

        loss_fn = module._create_loss_function("soft_target", num_classes=256, param_range=1.0)

        assert isinstance(loss_fn, SoftTargetLoss)
        assert loss_fn.num_classes == 256
        assert loss_fn.mode == "triangular"
        assert loss_fn.width == 2

    def test_create_loss_function_normalized_regression(self):
        """Test creating normalized regression loss function."""
        module = VIMHLitModule(
            net=self.net,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            loss_type="normalized_regression",
            auto_configure_from_dataset=False,
        )

        loss_fn = module._create_loss_function("normalized_regression", num_classes=256, param_range=10.0)

        assert isinstance(loss_fn, NormalizedRegressionLoss)
        assert loss_fn.param_min == 0.0
        assert loss_fn.param_max == 1.0  # Default range, updated later by auto-configuration
        assert loss_fn.loss_type == "mse"
        assert loss_fn.return_perceptual_units is True

    def test_create_loss_function_invalid_type(self):
        """Test error handling for invalid loss type."""
        module = VIMHLitModule(
            net=self.net,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            loss_type="cross_entropy",
            auto_configure_from_dataset=False,
        )

        with pytest.raises(ValueError, match="Unknown loss_type: invalid_type"):
            module._create_loss_function("invalid_type", num_classes=256, param_range=1.0)

    def test_loss_type_initialization(self):
        """Test initialization with different loss types."""
        # Test each loss type
        loss_types = [
            "cross_entropy",
            "ordinal_regression",
            "quantized_regression",
            "weighted_cross_entropy",
            "soft_target",
            "normalized_regression",
        ]

        for loss_type in loss_types:
            module = VIMHLitModule(
                net=SimpleCNN(input_channels=3, heads_config={"param_0": 10}),
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                loss_type=loss_type,
                auto_configure_from_dataset=False,
            )

            assert module.loss_type == loss_type
            # Check output_mode mapping
            if loss_type == "normalized_regression":
                assert module.output_mode == "regression"
            else:
                assert module.output_mode == "classification"

    def test_backward_compatibility_output_mode_regression(self):
        """Test backward compatibility with output_mode='regression'."""
        module = VIMHLitModule(
            net=self.net,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            output_mode="regression",
            auto_configure_from_dataset=False,
        )

        assert module.loss_type == "normalized_regression"
        assert module.output_mode == "regression"

    def test_backward_compatibility_output_mode_classification(self):
        """Test backward compatibility with output_mode='classification'."""
        module = VIMHLitModule(
            net=self.net,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            output_mode="classification",
            auto_configure_from_dataset=False,
        )

        assert module.loss_type == "cross_entropy"
        assert module.output_mode == "classification"

    def test_loss_type_overrides_output_mode(self):
        """Test that loss_type parameter overrides output_mode when both are provided."""
        module = VIMHLitModule(
            net=self.net,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            loss_type="soft_target",
            output_mode="regression",  # This should be overridden
            auto_configure_from_dataset=False,
        )

        assert module.loss_type == "soft_target"
        assert module.output_mode == "classification"  # Should be classification for soft_target


class TestParameterRangeDetection:
    """Test the _get_param_range_for_head method."""

    def setup_method(self):
        """Setup test components."""
        self.net = SimpleCNN(
            input_channels=3,
            heads_config={"param_0": 256, "param_1": 128},
            input_size=32,
        )
        self.optimizer = torch.optim.Adam
        self.scheduler = lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

    def test_get_param_range_from_dataset_param_ranges(self):
        """Test getting parameter range from dataset param_ranges attribute."""
        module = VIMHLitModule(
            net=self.net,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            loss_type="ordinal_regression",
            auto_configure_from_dataset=False,
        )

        dataset = MockDataset(param_ranges={"param_0": 15.0, "param_1": 8.0})

        range_0 = module._get_param_range_for_head(dataset, "param_0")
        range_1 = module._get_param_range_for_head(dataset, "param_1")

        assert range_0 == 15.0
        assert range_1 == 8.0

    def test_get_param_range_from_dataset_metadata(self):
        """Test getting parameter range from dataset metadata."""
        module = VIMHLitModule(
            net=self.net,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            loss_type="ordinal_regression",
            auto_configure_from_dataset=False,
        )

        # Create dataset without param_ranges attribute so it falls back to metadata
        dataset = MockDataset()
        # Remove param_ranges attribute to force fallback to metadata
        delattr(dataset, 'param_ranges')
        dataset._metadata = {
            "parameter_mappings": {
                "param_0": {"min": 0, "max": 20},  # range = 20
                "param_1": {"min": 5, "max": 15},  # range = 10
            }
        }

        range_0 = module._get_param_range_for_head(dataset, "param_0")
        range_1 = module._get_param_range_for_head(dataset, "param_1")

        assert range_0 == 20.0
        assert range_1 == 10.0

    def test_get_param_range_tuple_format(self):
        """Test parameter range in tuple format (min, max)."""
        module = VIMHLitModule(
            net=self.net,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            loss_type="ordinal_regression",
            auto_configure_from_dataset=False,
        )

        dataset = MockDataset(param_ranges={"param_0": (0, 25), "param_1": (10, 20)})

        range_0 = module._get_param_range_for_head(dataset, "param_0")
        range_1 = module._get_param_range_for_head(dataset, "param_1")

        assert range_0 == 25.0  # 25 - 0
        assert range_1 == 10.0  # 20 - 10

    def test_get_param_range_fallback_default(self):
        """Test fallback to default when parameter not found."""
        module = VIMHLitModule(
            net=self.net,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            loss_type="ordinal_regression",
            auto_configure_from_dataset=False,
        )

        dataset = MockDataset(param_ranges={"other_param": 5.0})

        # Should return default 1.0 when parameter not found
        range_missing = module._get_param_range_for_head(dataset, "missing_param")
        assert range_missing == 1.0

    def test_get_param_range_invalid_format(self):
        """Test handling of invalid parameter range format."""
        module = VIMHLitModule(
            net=self.net,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            loss_type="ordinal_regression",
            auto_configure_from_dataset=False,
        )

        dataset = MockDataset(param_ranges={"param_0": "invalid_format"})

        # Should return default 1.0 for invalid format
        range_invalid = module._get_param_range_for_head(dataset, "param_0")
        assert range_invalid == 1.0


class TestAutoConfigurationWithLossType:
    """Test auto-configuration with the loss_type system."""

    def setup_method(self):
        """Setup test components."""
        self.net = SimpleCNN(
            input_channels=3,
            conv1_channels=32,
            conv2_channels=64,
            fc_hidden=128,
            heads_config={"param_0": 100, "param_1": 100},  # Wrong initial config
            input_size=32,
        )
        self.optimizer = torch.optim.Adam
        self.scheduler = lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

    def test_auto_configure_creates_correct_loss_functions(self):
        """Test that auto-configuration creates correct loss functions based on loss_type."""
        # Create a net with empty initial criteria - will be auto-configured
        net_with_placeholder = SimpleCNN(
            input_channels=3,
            conv1_channels=32,
            conv2_channels=64,
            fc_hidden=128,
            heads_config={"placeholder": 10},  # Will be replaced
            input_size=32,
        )

        module = VIMHLitModule(
            net=net_with_placeholder,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            loss_type="soft_target",
            auto_configure_from_dataset=True,
        )

        dataset = MockDataset(
            param_ranges={"param_0": 12.0, "param_1": 8.0}
        )

        # Simulate auto-configuration
        module._auto_configure_from_dataset(dataset)

        # Check that criteria were created with correct loss type
        assert len(module.criteria) == 2
        assert isinstance(module.criteria["param_0"], SoftTargetLoss)
        assert isinstance(module.criteria["param_1"], SoftTargetLoss)

        # Check configuration parameters (should use the dataset's heads_config)
        assert module.criteria["param_0"].num_classes == 256
        assert module.criteria["param_1"].num_classes == 128

    def test_auto_configure_with_ordinal_regression(self):
        """Test auto-configuration with ordinal regression loss type."""
        module = VIMHLitModule(
            net=self.net,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            loss_type="ordinal_regression",
            auto_configure_from_dataset=True,
        )

        dataset = MockDataset(param_ranges={"param_0": 20.0, "param_1": 15.0})

        module._auto_configure_from_dataset(dataset)

        assert len(module.criteria) == 2
        assert isinstance(module.criteria["param_0"], OrdinalRegressionLoss)
        assert isinstance(module.criteria["param_1"], OrdinalRegressionLoss)

        # Check parameter ranges were applied correctly
        assert module.criteria["param_0"].param_range == 20.0
        assert module.criteria["param_1"].param_range == 15.0

    def test_auto_configure_with_weighted_cross_entropy(self):
        """Test auto-configuration with weighted cross entropy loss type."""
        # Create a net with placeholder heads that will be auto-configured
        net_with_placeholder = SimpleCNN(
            input_channels=3,
            conv1_channels=32,
            conv2_channels=64,
            fc_hidden=128,
            heads_config={"placeholder": 10},  # Will be replaced
            input_size=32,
        )

        module = VIMHLitModule(
            net=net_with_placeholder,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            loss_type="weighted_cross_entropy",
            auto_configure_from_dataset=True,
        )

        dataset = MockDataset()
        module._auto_configure_from_dataset(dataset)

        assert len(module.criteria) == 2
        assert isinstance(module.criteria["param_0"], WeightedCrossEntropyLoss)
        assert isinstance(module.criteria["param_1"], WeightedCrossEntropyLoss)

        # Check default parameters (should use dataset's heads_config)
        assert module.criteria["param_0"].num_classes == 256
        assert module.criteria["param_1"].num_classes == 128

    def test_loss_weights_auto_configuration(self):
        """Test that loss weights are auto-configured when not provided."""
        module = VIMHLitModule(
            net=self.net,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            loss_type="cross_entropy",
            auto_configure_from_dataset=True,
        )

        dataset = MockDataset()
        module._auto_configure_from_dataset(dataset)

        # Check that default loss weights were set
        assert module.loss_weights == {"param_0": 1.0, "param_1": 1.0}

    def test_preserve_existing_criteria(self):
        """Test that existing criteria are preserved and updated with parameter ranges."""
        # Create module with pre-existing criteria
        existing_criteria = {
            "param_0": OrdinalRegressionLoss(num_classes=256, param_range=1.0),
            "param_1": OrdinalRegressionLoss(num_classes=128, param_range=1.0),
        }

        module = VIMHLitModule(
            net=self.net,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            criteria=existing_criteria,
            auto_configure_from_dataset=True,
        )

        dataset = MockDataset(param_ranges={"param_0": 25.0, "param_1": 18.0})
        module._auto_configure_from_dataset(dataset)

        # Check that criteria were preserved but updated
        assert len(module.criteria) == 2
        assert isinstance(module.criteria["param_0"], OrdinalRegressionLoss)
        assert isinstance(module.criteria["param_1"], OrdinalRegressionLoss)

        # Parameter ranges should be updated
        assert module.criteria["param_0"].param_range == 25.0
        assert module.criteria["param_1"].param_range == 18.0


class TestIntegrationWithExistingCode:
    """Test integration with existing VIMHLitModule functionality."""

    def test_model_step_with_new_loss_types(self):
        """Test that model_step works correctly with new loss types."""
        module = VIMHLitModule(
            net=SimpleCNN(input_channels=3, heads_config={"param_0": 10, "param_1": 5}),
            optimizer=torch.optim.Adam,
            scheduler=lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, step_size=10),
            loss_type="soft_target",
            auto_configure_from_dataset=False,
        )

        # Create dummy batch
        batch_size = 4
        dummy_input = torch.randn(batch_size, 3, 32, 32)
        dummy_labels = {
            "param_0": torch.randint(0, 10, (batch_size,)),
            "param_1": torch.randint(0, 5, (batch_size,)),
        }
        batch = (dummy_input, dummy_labels)

        # Test model step
        loss, preds, targets = module.model_step(batch)

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert not torch.isnan(loss)
        assert isinstance(preds, dict)
        assert "param_0" in preds and "param_1" in preds

    def test_training_step_with_loss_types(self):
        """Test training step with different loss types."""
        loss_types_to_test = ["ordinal_regression", "weighted_cross_entropy", "soft_target"]

        for loss_type in loss_types_to_test:
            module = VIMHLitModule(
                net=SimpleCNN(input_channels=3, heads_config={"param_0": 10}),
                optimizer=torch.optim.Adam,
                scheduler=lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, step_size=10),
                loss_type=loss_type,
                auto_configure_from_dataset=False,
            )

            # Create dummy batch
            batch_size = 4
            dummy_input = torch.randn(batch_size, 3, 32, 32)
            dummy_labels = {"param_0": torch.randint(0, 10, (batch_size,))}
            batch = (dummy_input, dummy_labels)

            # Test training step
            loss = module.training_step(batch, 0)

            assert isinstance(loss, torch.Tensor)
            assert loss.requires_grad
            assert not torch.isnan(loss), f"Loss is NaN for loss_type: {loss_type}"

    def test_output_mode_inference_for_metrics(self):
        """Test that output_mode is correctly inferred for metrics setup."""
        # Test classification mode
        module_classification = VIMHLitModule(
            net=SimpleCNN(input_channels=3, heads_config={"param_0": 10}),
            optimizer=torch.optim.Adam,
            scheduler=lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, step_size=10),
            loss_type="soft_target",
            auto_configure_from_dataset=False,
        )

        assert module_classification.output_mode == "classification"

        # Test regression mode
        module_regression = VIMHLitModule(
            net=SimpleCNN(input_channels=3, heads_config={"param_0": 1}),
            optimizer=torch.optim.Adam,
            scheduler=lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, step_size=10),
            loss_type="normalized_regression",
            auto_configure_from_dataset=False,
        )

        assert module_regression.output_mode == "regression"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
