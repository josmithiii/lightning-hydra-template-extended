# Lightning-Hydra-Template-Extended Documentation

Expert-focused documentation for power users. Get up to speed quickly on what's here and how to use it.

## ⚡ Quick Start (5 min)
```bash
source .venv/bin/activate          # Setup environment
make h                            # See all make targets
make tqa                          # Test all architectures (3 min)
make cbqa                         # Quick CIFAR validation (15 min)
make ca                           # Compare architectures (10 min)
```

## 🎯 Core Extensions
- **5 Architectures**: CNN, ConvNeXt, ViT, EfficientNet, MLP (8K-7M params)
- **CIFAR Benchmarks**: Literature-competitive baselines (85-95% CIFAR-10)
- **VIMH Multihead**: Variable Image MultiHead format with auto-config
- **50+ Make Targets**: `tq` `cb10c` `evit` `cbs` - abbreviated expert shortcuts

## 📚 Documentation Map

### Essential (Read First)
- **[architectures.md](architectures.md)** - 5 architectures, params, usage patterns
- **[vimh.md](vimh.md)** - VIMH dataset format, multihead classification

### Benchmarking & Research
- **[benchmarks.md](benchmarks.md)** - CIFAR system, expected performance, automation
- **[configuration.md](configuration.md)** - Hydra patterns, experiments, best practices

### Advanced Usage
- **[multihead.md](multihead.md)** - Multi-task learning details and implementation
- **[development.md](development.md)** - Extension patterns, integration guide

## 🔧 Expert Navigation
| Goal | Command | Doc | Time |
|------|---------|-----|------|
| Architecture comparison | `make ca` | architectures.md | 10 min |
| CIFAR benchmarks | `make cbs10` | benchmarks.md | 2h |
| VIMH training | `make evimh` | vimh.md | 15 min |
| Custom experiment | `python src/train.py experiment=X` | configuration.md | varies |
| Add architecture | Edit configs/model/ | development.md | 30 min |

## 📖 Reference
- **[quickref.md](quickref.md)** - Expert cheat sheet (start here)
- **[../README.md](../README.md)** - Main project documentation
- **[extensions.md](extensions.md)** - Complete feature overview
- **Makefile** - 50+ targets with h, tq, cb, e prefixes
