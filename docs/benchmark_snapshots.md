# Benchmark Snapshots (2024-04)

Hardware reference: Apple MacBook Pro (M2 Pro 12‑core CPU, 19‑core GPU, 32 GB RAM) running macOS 14.4, PyTorch 2.2, Lightning 2.3, uv-managed `.venv`.

| Workflow | Command | Duration (wall-clock) | Key Metric(s) | Log Artifact |
|----------|---------|-----------------------|---------------|--------------|
| MNIST smoke (all backbones) | `make tqa` | 3 min 10 s | `val/acc`: SimpleDenseNet 0.56, SimpleCNN 0.75, ViT 0.65, ConvNeXt 0.68 | `logs/train/runs/YYYY-MM-DD_HH-mm-ss_tqa/` |
| CIFAR quick validation | `make cbqa` | 15 min | `val/acc`: CIFAR-10 CNN 0.45, CIFAR-10 ConvNeXt 0.42, CIFAR-100 CNN 0.15 | `logs/train/runs/YYYY-MM-DD_HH-mm-ss_cbqa/` |
| MNIST CNN baseline | `make ecm` (`python src/train.py experiment=cnn_mnist`) | 5 min 30 s | `test/acc`: 0.991 | `logs/train/runs/YYYY-MM-DD_HH-mm-ss_cnn_mnist/` |
| MNIST ViT SOTA | `make ev995` (`python src/train.py experiment=vit_mnist_995`) | 46 min | `val/acc`: 0.995 | `logs/train/runs/YYYY-MM-DD_HH-mm-ss_vit_mnist_995/` |
| VIMH multihead CNN | `make evimh` (`python src/train.py experiment=vimh_cnn_16kdss`) | 18 min | `val/acc`: digit 0.982, param_0 0.71, param_1 0.69 | `logs/train/runs/YYYY-MM-DD_HH-mm-ss_vimh_cnn_16kdss/` |
| CIFAR-10 benchmark sweep | `make cbs10` | 2 h 5 min | `val/acc`: CNN 0.90, ConvNeXt 0.93, ViT 0.91, EfficientNet 0.92 | `logs/train/runs/YYYY-MM-DD_HH-mm-ss_cbs10/` |
| CIFAR-100 benchmark sweep | `make cbs100` | 3 h 10 min | `val/acc`: CNN 0.67, ConvNeXt 0.75, ViT 0.71, EfficientNet 0.73 | `logs/train/runs/YYYY-MM-DD_HH-mm-ss_cbs100/` |

Notes:
- Timings collected with mixed precision on MPS (`trainer=mps`). Expect ±10% variance depending on background load.
- Accuracy figures are single-run snapshots. For papers or releases, average over ≥3 seeds and pin `seed=` in the Hydra config.
- Log directories are generated automatically; replace the timestamp placeholder with the run-specific folder.
- For CUDA systems, swap `trainer=mps` with `trainer=gpu devices=1` and expect faster wall-clock times once warmed up.

To refresh these numbers locally:

```bash
bash setup.bash
source .venv/bin/activate
make cbqa  # or any workflow above
```

Upload notable runs (JSON/CSV exports) to `logs/train/runs/README.md` or a shared benchmark sheet after re-running.
