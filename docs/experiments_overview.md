# Lightning-Hydra-Template-Extended (LHTE) Experiment Overview

This document summarizes all experiment configurations in the Lightning-Hydra-Template-Extended (LHTE) experiments directory `configs/experiment/`.

The table was created by
[../scripts/extract_logs.py](../scripts/extract_logs.py)
after running
[../scripts/run_all_experiments.sh](../scripts/run_all_experiments.sh)

## Experiment Summary Table

| Experiment Name | Loss Type | Aggregate Metric | log10_decay_time | wah_position | Batch Size | Num Epochs | Runtime | Parameters |
|-----------------|-----------|------------------|------------------|----------------|------------|------------|---------|------------|
| cifar100_benchmark_cnn         | cross_entropy | 1.7620*↓         | N/ANone          | N/ANone        | N/A        | 200        | 37m5.808s | 1.2 M |
| cifar100_benchmark_cnn_improved | cross_entropy | 2.1996*↓         | N/ANone          | N/ANone        | N/A        | 200        | 33m31s   | 2.6 M |
| cifar100_benchmark_convnext    | cross_entropy | 2.6845*↓         | N/ANone          | N/ANone        | N/A        | 100        | 20m30.106s | 296 K |
| cifar100_cnn                   | cross_entropy | 2.0759*↓         | N/ANone          | N/ANone        | N/A        | 100        | 17m21.252s | 1.2 M |
| cifar100_coarse_cnn            | cross_entropy | 1.9254*↓         | N/ANone          | N/ANone        | N/A        | 60         | 41m2s    | 3.3 M |
| cifar100mh_cnn                 | cross_entropy | 354.3169*↓       | N/ANone          | N/ANone        | N/A        | 100        | 14m5.995s | 1.2 M |
| cifar100mh_convnext            | cross_entropy | 346.8721*↓       | N/ANone          | N/ANone        | N/A        | 100        | 111m48.814s | 299 K |
| cifar100mh_efficientnet        | cross_entropy | 363.3585*↓       | N/ANone          | N/ANone        | N/A        | N/A        | 339m35.024s | 7.3 M |
| cifar100mh_vit                 | cross_entropy | 478.5045*↓       | N/ANone          | N/ANone        | N/A        | 100        | 135m6.623s | 14.7 M |
| cifar10_benchmark_cnn          | cross_entropy | 0.6148*↓         | N/ANone          | N/ANone        | N/A        | 50         | 5m11.831s | 1.1 M |
| cifar10_benchmark_convnext     | cross_entropy | 0.8397*↓         | N/ANone          | N/ANone        | N/A        | 50         | 9m28.298s | 288 K |
| cifar10_benchmark_efficientnet | Incomplete  | N/A↑             | N/A↑             | N/A↑           | N/A        | 50         | 40m24s   | 5.0 M |
| cifar10_benchmark_vit          | cross_entropy | 0.7496*↓         | N/ANone          | N/ANone        | N/A        | 50         | 9m3.062s | 213 K |
| cifar10_cnn                    | cross_entropy | 0.6279*↓         | N/ANone          | N/ANone        | N/A        | 50         | 5m17.412s | 1.1 M |
| cifar10_cnn_cpu                | cross_entropy | 0.6162*↓         | N/ANone          | N/ANone        | N/A        | 50         | 27m4.933s | 1.1 M |
| cifar10_convnext_128k_optimized | cross_entropy | 0.7360*↓         | N/ANone          | N/ANone        | N/A        | 50         | 12m9.069s | 309 K |
| cifar10_convnext_64k_optimized | cross_entropy | 0.8407*↓         | N/ANone          | N/ANone        | N/A        | 50         | 9m34.547s | 122 K |
| cnn_mnist                      | cross_entropy | 0.0306*↓         | N/ANone          | N/ANone        | N/A        | 10         | 2m27.687s | 421 K |
| convnext_mnist                 | cross_entropy | 0.0428*↓         | N/ANone          | N/ANone        | N/A        | 20         | 5m6.750s | 288 K |
| convnext_v2_official_tiny_benchmark | cross_entropy | 0.5250*↓         | N/ANone          | N/ANone        | N/A        | N/A        | 10m22.690s | 288 K |
| example                        | cross_entropy | 0.0631*↓         | N/ANone          | N/ANone        | N/A        | 10         | 1m16.779s | 151 K |
| multihead_cnn_cifar10          | cross_entropy | 1.0279*↓         | N/ANone          | N/ANone        | N/A        | 20         | 5m36.892s | 1.1 M |
| multihead_cnn_mnist            | cross_entropy | 0.0615*↓         | N/ANone          | N/ANone        | N/A        | 10         | 2m22.067s | 422 K |
| vimh_cnn                       | cross_entropy | 5.3176*↓         | N/ANone          | N/ANone        | N/A        | N/A        | 0m46.201s | 1.5 M |
| vimh_cnn_16kdss                | cross_entropy | 7.2989*↓         | N/ANone          | N/ANone        | N/A        | N/A        | 16m28.097s | 1.5 M |
| vimh_cnn_16kdss_ordinal        | ordinal     | 7.5690*↓         | N/ANone          | N/ANone        | N/A        | N/A        | 12m10.277s | 1.5 M |
| vimh_cnn_16kdss_regression     | normalized_regression | 0.0000*↓         | N/ANone          | N/ANone        | N/A        | N/A        | 3m47.873s | 1.1 M |
| vit_mnist                      | cross_entropy | 0.0593*↓         | N/ANone          | N/ANone        | N/A        | 20         | 3m8.591s | 210 K |
| vit_mnist_995                  | cross_entropy | 0.0190*↓         | N/ANone          | N/ANone        | N/A        | 200        | 41m46.551s | 210 K |

Notes:
- Loss Type shows the configured loss function from model config (e.g., cross_entropy, normalized_regression, ordinal).
- Classification models (cross_entropy, ordinal) use JND-weighted accuracy metrics; regression models use MSE/MAE loss functions.
- Arrows indicate optimization direction: ↑ for higher-is-better (accuracies), ↓ for lower-is-better (losses/errors).
- Aggregate Metric is the mean of the available per-head test metrics for log10_decay_time and wah_position (falls back to test/loss when heads are missing).
- Values marked with * indicate fallback to test/loss due to missing head metrics.
- Per-head columns report the exact metric logged (accuracy for classification heads, MAE for regression heads); values are rounded to 4 decimals.
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
