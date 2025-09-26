# Lightning-Hydra-Template-Extended Documentation

Expert-focused documentation for power users. Get up to speed quickly on what's here and how to use it.

## ⚡ Quick Start (5 min)
```bash
bash setup.bash                     # Create & install deps with uv (once)
source .venv/bin/activate       # Activate virtualenv for this shell
make h                          # See all make targets
make tqa                        # Test all architectures (3 min)
make cbqa                       # Quick CIFAR validation (15 min)
make ca                         # Compare architectures (10 min)
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
- **[onboarding.md](onboarding.md)** - 15-minute onboarding guide

### Benchmarking & Research
- **[experiments_overview.md](experiments_overview.md)** - Complete experiment results table with performance metrics
- **[benchmarks.md](benchmarks.md)** - CIFAR system, expected performance, automation
- **[configuration.md](configuration.md)** - Hydra patterns, experiments, best practices

### Advanced Usage
- **[multihead.md](multihead.md)** - Multi-task learning details and implementation
- **[vimh_cookbook.md](vimh_cookbook.md)** - Practical recipes for VIMH datasets
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
- **[experiments_overview.md](experiments_overview.md)** - Performance results for all 29 experiments
- Hydra overrides → see section in [quickref.md](quickref.md)
- Config Group Map → in [quickref.md](quickref.md#config-group-map) and README
- **[../README.md](../README.md)** - Main project documentation
- **[extensions.md](extensions.md)** - Complete feature overview
- **[features.md](features.md)** - High-level summary of LHTE capabilities
- **[benchmark_snapshots.md](benchmark_snapshots.md)** - Latest wall-clock + accuracy samples
- **[presentation/README.md](presentation/README.md)** - Legacy slide deck archive (LaTeX)
- **Makefile** - 50+ targets with h, tq, cb, e prefixes

## ⚠️ Common Pitfalls
- Use `trainer=mps` on macOS; set `num_workers: 0` for VIMH on MPS.
- First run may download datasets; allow time and network.
- Checkpoints must be local (remote URL loading is disabled).
