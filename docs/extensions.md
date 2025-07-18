# Lightning-Hydra-Template-Extended

## Overview

This [Lightning-Hydra-Template-Extended](https://github.com/josmithiii/lightning-hydra-template-extended.git) project extends the original [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template) with powerful new capabilities for deep learning research while maintaining full backward compatibility.

## 🎯 Key Extensions

### 1. Multiple Neural Network Architectures
- **SimpleCNN**: Convolutional neural networks with multihead support
- **ConvNeXt-V2**: Modern CNN with Global Response Normalization
- **Vision Transformer**: Attention-based learning on image patches
- **EfficientNet**: Highly efficient CNN architecture
- **Configurable parameters**: Easy switching via Hydra configuration

### 2. CIFAR Benchmark Suite
- **CIFAR-10 & CIFAR-100**: Comprehensive computer vision benchmarks
- **Multiple architectures**: CNN, ConvNeXt, ViT, EfficientNet support
- **Automated benchmarks**: Systematic performance comparison
- **Literature-competitive baselines**: 85-95% CIFAR-10, 55-75% CIFAR-100

### 3. Advanced Multihead Classification (VIMH)
- **Variable Image MultiHead (VIMH)**: Next-generation multihead dataset format
- **Variable dimensions**: Support for 32x32x3, 28x28x1, and other image formats
- **Self-describing metadata**: JSON-based dataset configuration with parameter mappings
- **8-bit quantization**: Efficient storage and handling of continuous parameters
- **Auto-configuration**: Models automatically configure from dataset metadata
- **Performance optimized**: 10x faster initialization with efficient dimension detection
- **Comprehensive testing**: 27 tests covering all VIMH functionality
- **Real-world applications**: Audio synthesis, computer vision, scientific computing

### 4. Configurable Loss Functions
- **Hydra-managed losses**: No more hardcoded loss functions
- **Multiple criteria**: Support for different loss functions per task
- **Easy experimentation**: Switch losses via configuration

### 5. Enhanced Make Targets
- **Convenient shortcuts**: `make trc` (train CNN), `make cb10c` (CIFAR-10 benchmark)
- **Quick testing**: `make tqa` (test all architectures), `make cbqa` (quick CIFAR tests)
- **Systematic comparison**: `make ca` (compare architectures), `make cbs` (benchmark suite)

## 📚 Documentation

This comprehensive documentation has been organized into focused, navigable files:

### 🚀 Features
- **[features.md](features.md)** - High-level overview and key features summary

### 🏗️ Technical Details
- **[architectures.md](architectures.md)** - Detailed architecture documentation, parameter comparisons, and usage guides
- **[benchmarks.md](benchmarks.md)** - CIFAR benchmark system, expected performance, and automated testing
- **[multihead.md](multihead.md)** - Multihead classification system, synthetic label generation, and multi-task learning
- **[vimh.md](vimh.md)** - VIMH dataset format specification, usage guide, and implementation details

### 🛠️ Usage and Reference
- **[makefile.md](makefile.md)** - Complete make targets reference with abbreviations and workflows
- **[configuration.md](configuration.md)** - Configuration patterns, experiment system, and best practices

### 👩‍💻 Development
- **[development.md](development.md)** - Further-development guide, extension patterns, and integration approach

## 🚀 Quick Start

### 1. Environment Setup
```bash
source .venv/bin/activate  # or: conda activate myenv
```

### 2. Test All Architectures (Quick)
```bash
make tqa      # Test SimpleDenseNet, CNN, ConvNeXt (~3 minutes total)
```

### 3. CIFAR Quick Validation
```bash
make cbqa     # Quick CIFAR tests across architectures (~15 minutes)
```

### 4. Architecture Comparison
```bash
make ca       # Systematic 3-epoch comparison (~10 minutes)
```

### 5. Full Benchmarks
```bash
make cbs10    # Complete CIFAR-10 benchmark suite
make cbs100   # Complete CIFAR-100 benchmark suite
```

### 6. VIMH Multihead Training
```bash
# Train with VIMH dataset
python src/train.py experiment=vimh_cnn

# Complete training example with analysis
python examples/vimh_training.py

# Quick demo with visualizations
python examples/vimh_training.py --demo --save-plots
```

## 🎯 Example Workflows

### Research Workflow
```bash
# 1. Quick exploration
make tqa && make cbqa

# 2. Focused comparison
make cb10c && make cb10cn

# 3. Full evaluation
make cbs
```

### Development Workflow
```bash
# 1. Code quality
make f && make t

# 2. Quick validation
make tq

# 3. Architecture test
make tqc
```

### Experiment Workflow
```bash
# 1. Run baseline experiments
make example && make evit && make excn

# 2. Custom configurations
python src/train.py experiment=cifar10_benchmark_cnn trainer.max_epochs=100

# 3. Multihead experiments
make emhcm && make emhcc10
```

## 🔗 Integration with Original Template

### ✅ Preserved Features
- All original Lightning-Hydra template functionality works unchanged
- Original SimpleDenseNet and MNIST configurations preserved
- Existing make targets, workflows, and documentation remain functional
- Complete backward compatibility for existing users

### ✨ Enhanced Features
- **Architecture diversity**: 5 neural network types vs. 1 original
- **Dataset variety**: MNIST + CIFAR-10/100 vs. MNIST only
- **Benchmark capabilities**: Systematic performance evaluation
- **Configuration flexibility**: Configurable losses, multihead support
- **Developer convenience**: 50+ make targets with abbreviations

## 📊 Performance Summary

| Architecture | MNIST (quick) | CIFAR-10 (full) | CIFAR-100 (full) | Parameters |
|-------------|---------------|------------------|-------------------|------------|
| SimpleDenseNet | ~56.6% | N/A | N/A | 68K |
| SimpleCNN | ~74.8% | 85-92% | 55-70% | 421K-3.3M |
| ConvNeXt-V2 | ~68.3% | 90-95% | 70-80% | 73K-725K |
| Vision Transformer | TBD | 88-93% | 65-75% | 210K-821K |
| EfficientNet | TBD | 89-94% | 67-77% | 210K-7M |

*Quick results from 1 epoch; full results from complete training*

## 🎨 Key Design Principles

### Non-Destructive Extensions
- **Add, don't modify**: New files added, existing files preserved
- **Zero risk**: Original workflows continue unchanged
- **Easy rollback**: Extensions can be removed without affecting base template
- **Incremental adoption**: Users can adopt new features gradually

### Configuration-Driven Development
- **No code changes**: Most experiments achievable through configuration
- **Reproducible research**: Version-controlled experiment configurations
- **Systematic comparison**: Fair evaluation across architectures and datasets
- **Best practices**: Established patterns for common research tasks

## 🏆 Ready for Research

This extended template provides a comprehensive platform for:
- **Computer vision research**: Modern architectures and standard benchmarks
- **Multi-task learning**: Multihead classification capabilities
- **Architecture comparison**: Systematic evaluation across multiple datasets
- **Reproducible experiments**: Version-controlled configurations and fixed seeds
- **Literature-competitive baselines**: Performance matching published results

**Get started with modern deep learning research!** 🚀

---

*For the original Lightning-Hydra-Template documentation, see [README.md](../README.md)*
