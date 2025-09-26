# Lightning-Hydra-Template-Extended (LHTE) Experiment Overview

This document summarizes all experiment configurations in the Lightning-Hydra-Template-Extended (LHTE) experiments directory `configs/experiment/`.

The table was created by
[../scripts/extract_logs.py](../scripts/extract_logs.py)
after running
[../scripts/run_all_experiments.sh](../scripts/run_all_experiments.sh)

## Experiment Summary Table

| Experiment Name | Loss Type | Aggregate Metric | texture | fine_label | coarse_label | Batch Size | Num Epochs | Runtime | Parameters |
|-----------------|-----------|------------------|------------------|------------|--------------|------------|------------|---------|------------|
| cifar100_benchmark_cnn         | cross_entropy | 1.7670*↓         | N/A              | N/A            | N/A            | N/A        | 201        | 26m0.852s | 1.2 M |
| cifar100_benchmark_cnn_improved | cross_entropy | 2.1922*↓         | N/A              | N/A            | N/A            | N/A        | 201        | 49m25.965s | 2.6 M |
| cifar100_benchmark_convnext    | cross_entropy | 2.6950*↓         | N/A              | N/A            | N/A            | N/A        | 101        | 30m23.386s | 296 K |
| cifar100_cnn                   | cross_entropy | 1.9128*↓         | N/A              | N/A            | N/A            | N/A        | 101        | 13m40.061s | 1.2 M |
| cifar100_coarse_cnn            | cross_entropy | 1.9353*↓         | N/A              | N/A            | N/A            | N/A        | 61         | 8m14.151s | 1.1 M |
| cifar100mh_cnn                 | cross_entropy | 0.2240↑          | 0.1276↑          | 0.2879↑        | 0.2564↑        | N/A        | 101        | 15m8.166s | 1.2 M |
| cifar100mh_convnext            | cross_entropy | 0.2621↑          | 0.1244↑          | 0.3424↑        | 0.3194↑        | N/A        | 101        | 29m55.743s | 299 K |
| cifar100mh_efficientnet        | cross_entropy | 0.2561↑          | 0.1261↑          | 0.3103↑        | 0.3318↑        | N/A        | 90         | 75m30.767s | 7.3 M |
| cifar100mh_vit                 | cross_entropy | 0.1138↑          | 0.1258↑          | 0.0937↑        | 0.1220↑        | N/A        | 101        | 161m46.319s | 14.7 M |
| cifar10_benchmark_cnn          | cross_entropy | 0.6169*↓         | N/A              | N/A            | N/A            | N/A        | 51         | 6m9.671s | 1.1 M |
| cifar10_benchmark_convnext     | cross_entropy | 0.8473*↓         | N/A              | N/A            | N/A            | N/A        | 51         | 13m3.261s | 288 K |
| cifar10_benchmark_efficientnet | cross_entropy | 0.5822*↓         | N/A              | N/A            | N/A            | N/A        | 51         | 28m1.196s | 5.0 M |
| cifar10_benchmark_vit          | cross_entropy | 0.7681*↓         | N/A              | N/A            | N/A            | N/A        | 51         | 11m53.129s | 213 K |
| cifar10_cnn                    | cross_entropy | 0.6129*↓         | N/A              | N/A            | N/A            | N/A        | 51         | 6m15.131s | 1.1 M |
| cifar10_cnn_cpu                | cross_entropy | 0.5842*↓         | N/A              | N/A            | N/A            | N/A        | 51         | 29m2.232s | 1.1 M |
| cifar10_convnext_128k_optimized | cross_entropy | 0.7492*↓         | N/A              | N/A            | N/A            | N/A        | 51         | 12m50.950s | 309 K |
| cifar10_convnext_64k_optimized | cross_entropy | 0.8127*↓         | N/A              | N/A            | N/A            | N/A        | 51         | 10m56.824s | 122 K |
| cnn_mnist                      | cross_entropy | 0.0271*↓         | N/A              | N/A            | N/A            | N/A        | 11         | 2m37.585s | 421 K |
| convnext_mnist                 | cross_entropy | 0.0471*↓         | N/A              | N/A            | N/A            | N/A        | 21         | 9m5.772s | 288 K |
| convnext_v2_official_tiny_benchmark | cross_entropy | 0.5267*↓         | N/A              | N/A            | N/A            | N/A        | 67         | 29m42.055s | 288 K |
| example                        | cross_entropy | 0.0751*↓         | N/A              | N/A            | N/A            | N/A        | 11         | 2m40.227s | 151 K |
| multihead_cnn_cifar10          | Incomplete  | N/A↑             | N/A              | N/A            | N/A            | N/A        | N/A        | 0m44.565s | 1.1 M |
| multihead_cnn_mnist            | Incomplete  | N/A↑             | N/A              | N/A            | N/A            | N/A        | N/A        | 0m28.123s | 422 K |
| vimh_cnn                       | Incomplete  | N/A↑             | N/A              | N/A            | N/A            | N/A        | N/A        | 0m2.586s | N/A |
| vimh_cnn_16kdss                | cross_entropy | 0.2501↑          | 0.1214↑          | 0.3193↑        | 0.3096↑        | N/A        | 101        | 31m35.290s | 1.5 M |
| vimh_cnn_16kdss_ordinal        | ordinal_regression | 0.0000↑          | 0.0000↑          | 0.0001↑        | 0.0000↑        | N/A        | 17         | 4m36.903s | 1.5 M |
| vimh_cnn_16kdss_regression     | unknown     | 30.1922↓         | 0.3275↓          | 79.0091↓       | 11.2399↓       | N/A        | 17         | 3m57.254s | 1.1 M |
| vit_mnist                      | cross_entropy | 0.0606*↓         | N/A              | N/A            | N/A            | N/A        | 21         | 7m12.446s | 210 K |
| vit_mnist_995                  | Incomplete  | N/A↑             | N/A              | N/A            | N/A            | N/A        | N/A        | 0m3.454s | N/A |

Notes:
- Loss Type shows the configured loss function from model config (e.g., cross_entropy, normalized_regression, ordinal).
- Classification models (cross_entropy, ordinal) use JND-weighted accuracy metrics; regression models use MSE/MAE loss functions.
- Arrows indicate optimization direction: ↑ for higher-is-better (accuracies), ↓ for lower-is-better (losses/errors).
- Aggregate Metric is the mean of the available per-head test metrics from detected heads (falls back to test/loss when heads are missing).
- Values marked with * indicate fallback to test/loss due to missing head metrics.
- Head columns show metrics for different dataset types: texture/fine_label/coarse_label (VIMH), digit (MNIST), main (CIFAR-10); values are rounded to 4 decimals.
- Batch Size is parsed from the Hydra data configuration line.
- Num Epochs shows actual epochs completed when available (from training completion log), otherwise falls back to configured max_epochs.
- Runtime uses the shell `real` timer when present (falls back to log timestamps otherwise); Parameters come from the Lightning model summary output.

## Architecture Categories

### CNN Architectures
- **Standard CNNs**: Various parameter counts (64K, 421K, 1M)
- **Simple Dense Net**: 68K params (original template example)

### Modern Architectures
- **ConvNeXt**: 64K, 128K, 210K parameter variants
- **EfficientNet**: 210K parameter variants
- **Vision Transformer (ViT)**: 210K params, plus ultra-optimized 995-param variant

### Specialized Architectures
- **Multihead Models**: For simultaneous multi-task prediction
- **VIMH Models**: Variable Image MultiHead for spectral/audio analysis
- **Ordinal Regression**: Distance-aware loss functions
- **Pure Regression**: Continuous parameter prediction

## Dataset Categories

### Standard Vision Datasets
- **MNIST**: 28x28 grayscale handwritten digits
- **CIFAR-10**: 32x32 color images, 10 classes
- **CIFAR-100**: 32x32 color images, 100 fine-grained classes
- **CIFAR-100 Coarse**: 32x32 color images, 20 coarse-grained classes

### Specialized Datasets
- **VIMH**: Variable Image MultiHead format for audio spectral analysis
- **Multihead variants**: Extended datasets with multiple prediction targets

## Training Configurations

### Performance Optimizations
- **CPU Training**: For MPS compatibility issues
- **Mixed Precision**: 16-bit training for memory efficiency
- **Batch Size Variations**: 64-128 depending on model complexity
- **MPS Optimizations**: num_workers=0, pin_memory adjustments

### Specialized Training
- **Benchmark Configs**: Standardized evaluation setups
- **Long Training**: Up to 200 epochs for complex tasks
- **Quick Testing**: 1-10 epochs for rapid iteration
- **Official Reproductions**: Exact hyperparameter matching

## Notable Features

1. **Comprehensive Benchmarking**: Multiple architectures tested on same datasets
2. **Parameter Count Variants**: Models scaled from 995 params to 1M+ params
3. **Multi-task Learning**: Extensive multihead configurations
4. **Audio/Spectral Focus**: VIMH experiments for neural spectral modeling
5. **Hardware Compatibility**: CPU, GPU, and MPS (Apple Silicon) support
6. **Loss Function Diversity**: Classification, regression, and ordinal losses
7. **Reproducibility**: Fixed seeds and deterministic training options

## Summary Statistics

- **Total Experiments**: 29 (2 failed)
- **Loss Types**: CrossEntropy (23), JND-weighted (5), MSE/MAE (1)
- **CNN Experiments**: 15
- **ConvNeXt Experiments**: 6
- **Vision Transformer Experiments**: 4
- **EfficientNet Experiments**: 2
- **Multihead Experiments**: 8
- **VIMH Experiments**: 4
- **Benchmark Experiments**: 8
- **MNIST Experiments**: 8 (best: 99.44% accuracy)
- **CIFAR-10 Experiments**: 7 (best: 78.76% accuracy)
- **CIFAR-100 Experiments**: 8 (best: 58.16% accuracy)
- **Parameter Range**: 122K - 14.7M parameters
- **Runtime Range**: 46s - 339m35s

This comprehensive experiment suite provides extensive coverage of modern deep learning architectures, training configurations, and specialized applications for both standard computer vision tasks and neural spectral modeling for audio processing.
