# Lightning-Hydra-Template-Extended (LHTE) Experiment Overview

This document summarizes all experiment configurations in the Lightning-Hydra-Template-Extended (LHTE) project directory `configs/experiment/`.

## Experiment Summary Table

| Experiment | Architecture | Loss Type | Acc/Loss | Runtime | #Params | Special Notes |
|------------|-------------|-----------|----------|---------|---------|---------------|
| cifar10_benchmark_cnn | cifar10_cnn_64k | CrossEntropy | 78.71% | 5m12s | 1.1M | CIFAR-10 CNN benchmark, optimized training config |
| cifar10_benchmark_convnext | cifar10_convnext_210k | CrossEntropy | 72.21% | 9m28s | 288K | CIFAR-10 ConvNeXt benchmark |
| cifar10_benchmark_efficientnet | cifar10_efficientnet_210k | CrossEntropy | Failed | - | 5.0M | CIFAR-10 EfficientNet benchmark |
| cifar10_benchmark_vit | cifar10_vit_210k | CrossEntropy | 73.77% | 9m3s | 213K | CIFAR-10 Vision Transformer benchmark |
| cifar10_cnn_cpu | cifar10_cnn_64k | CrossEntropy | 78.76% | 27m5s | 1.1M | **CPU-only training** due to MPS compatibility issues |
| cifar10_cnn | cifar10_cnn_64k | CrossEntropy | 78.07% | 5m17s | 1.1M | CIFAR-10 CNN baseline, standard training |
| cifar10_convnext_128k_optimized | cifar10_convnext_128k | CrossEntropy | 76.43% | 12m9s | 309K | CIFAR-10 specific optimizations |
| cifar10_convnext_64k_optimized | cifar10_convnext_64k | CrossEntropy | 71.76% | 9m35s | 122K | CIFAR-10 specific optimizations |
| cifar100_benchmark_cnn_improved | cifar100_cnn_1m_improved | CrossEntropy | 58.16% | - | 2.6M | **Improved version** with data augmentation |
| cifar100_benchmark_cnn | cifar100_cnn_1m_multistep | CrossEntropy | 52.08% | 37m6s | 1.2M | CIFAR-100 CNN benchmark, multistep scheduler |
| cifar100_benchmark_convnext | cifar100_convnext_210k | CrossEntropy | 39.68% | 20m30s | 296K | CIFAR-100 ConvNeXt benchmark |
| cifar100_coarse_cnn | cifar100_coarse_cnn_64k | CrossEntropy | 41.94% | - | 3.3M | **Coarse-grained classification** (20 classes) |
| cifar100_cnn | cifar100_cnn_1m_original | CrossEntropy | 44.96% | 17m21s | 1.2M | **Fine-grained classification** (100 classes) |
| cifar100mh_cnn | cifar100mh_cnn_64k | JND-weighted | 22.40% | 14m6s | 1.2M | **Multihead CNN** with real labels |
| cifar100mh_convnext | cifar100mh_convnext_210k | JND-weighted | 26.21% | 111m49s | 299K | **Multihead ConvNeXt** with cosine annealing |
| cifar100mh_efficientnet | cifar100mh_efficientnet_210k | JND-weighted | 25.61% | 339m35s | 7.3M | **Multihead EfficientNet** with cosine annealing |
| cifar100mh_vit | cifar100mh_vit_210k | JND-weighted | Failed | - | 14.7M | **Multihead Vision Transformer**, smaller batch size |
| cnn_mnist | mnist_cnn_421k | CrossEntropy | 99.10% | 2m28s | 421K | Short training (1-10 epochs) |
| convnext_mnist | mnist_convnext_210k | CrossEntropy | 98.88% | 5m7s | 288K | Cosine annealing scheduler |
| convnext_v2_official_tiny_benchmark | mnist_convnext_210k | CrossEntropy | 99.15% | 10m23s | 288K | **Official ConvNeXt V2-Tiny benchmark**, mixed precision |
| example | mnist_sdn_68k | CrossEntropy | 98.24% | 1m17s | 151K | Original Lightning-Hydra-Template example, 10 epochs only |
| multihead_cnn_cifar10 | cifar10_mh_cnn_64k | JND-weighted | 84.49% | 5m37s | 1.1M | **Multihead CNN** demo with custom monitoring |
| multihead_cnn_mnist | mnist_mh_cnn_422k | JND-weighted | 99.01% | 2m22s | 422K | **Multihead CNN** demo |
| vimh_cnn_16kdss | vimh_cnn_64k | JND-weighted | 23.54% | 16m28s | 1.5M | **VIMH** with 16K dataset samples, resonarium dataset |
| vimh_cnn_16kdss_ordinal | vimh_cnn_64k_ordinal | JND-weighted | 21.27% | 12m10s | 1.5M | **VIMH ordinal regression**, distance-aware loss |
| vimh_cnn_16kdss_regression | vimh_cnn_64k_regression | MSE/MAE | 8.79 MAE | 3m48s | 1.1M | **VIMH pure regression** heads, monitors MAE |
| vimh_cnn | vimh_cnn_64k | JND-weighted | 23.50% | 46s | 1.5M | VIMH CNN baseline, MPS optimizations |
| vit_mnist_995 | mnist_vit_995 | CrossEntropy | 99.44% | 41m47s | 210K | **Ultra-optimized tiny model**, 200 epochs |
| vit_mnist | mnist_vit_210k | CrossEntropy | 98.24% | 3m9s | 210K | Cosine annealing scheduler |

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
