# LHTE at a Glance (Vision Enthusiast Primer)

Use this Markdown handout when you want the “slide deck” experience without compiling LaTeX.

## Why LHTE?
- **One repo, many backbones:** SimpleCNN, ConvNeXt-V2, EfficientNet, ViT, and multihead MLPs.
- **Hydra-first workflows:** Swap datasets/models/trainers with single overrides.
- **Image-friendly extensions:** CIFAR-10/100 benchmarks, VIMH multihead datasets, diagram tooling.

## 5-Minute Tour
```bash
sh setup.sh && source .venv/bin/activate
make h            # Discover commands
make tqa          # MNIST smoke run across all architectures
make cbqa         # CIFAR quick validation sanity check
python src/train.py experiment=vimh_cnn  # Real multihead dataset
```

## Feature Cheatsheet
| Theme | What You Get | Where |
|-------|--------------|-------|
| Architectures | CNN, ConvNeXt, EfficientNet, ViT, SDN, SimpleMLP | [docs/architectures.md](architectures.md) |
| Datasets | MNIST, CIFAR-10/100, Multihead synth, VIMH | [docs/vimh.md](vimh.md) |
| Benchmarks | Reproducible CIFAR suites | [docs/benchmarks.md](benchmarks.md) |
| Multihead | Strategy + auto-configured heads | [docs/multihead.md](multihead.md) |
| Cookbook | Practical VIMH recipes | [docs/vimh_cookbook.md](vimh_cookbook.md) |

## Elevator Pitches
- **Prototype**: `make tq*` to feel out architectures before long runs.
- **Benchmark**: `make cbs10` / `make cbs100` produce literature-grade baselines with logs.
- **Research**: VIMH auto-config gives you image + metadata predictions without bespoke code.

## Upgrade Ideas for Your Next Talk
1. Show `viz/simple_model_diagram.py --config <name>` to illustrate architecture shape.
2. Compare `logs/train/runs/` exports to highlight multihead metrics per parameter.
3. Pair the cookbook with personal datasets: drop new metadata JSON, rerun `make evimh`.

## Suggested Slide Flow
1. Motivation: reuse-friendly Lightning + Hydra template (2 minutes).
2. Architecture menu with parameter counts (2 minutes).
3. Live demo clip or screenshots from `make tqa` / `make cbqa` (1 minute).
4. Multihead + VIMH story (2 minutes).
5. Roadmap (benchmark snapshots + planned refactors) (1 minute).

Ready to present? Copy/paste sections into your slide tool of choice, or keep it simple and screen-share this doc.
