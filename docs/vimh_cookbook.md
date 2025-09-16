# VIMH Cookbook

Hands-on recipes for working with Variable Image MultiHead (VIMH) datasets in Lightning-Hydra-Template-Extended. Pair this with the format reference in [vimh.md](vimh.md) and the architecture deep dives in [multihead_data_architecture.md](multihead_data_architecture.md).

## 1. Inspect an Existing Dataset

```bash
# Activate environment if needed
sh setup.sh && source .venv/bin/activate

# List generated datasets
ls data-vimh/

# Display metadata + random samples
make dv               # or: python display_vimh.py data-vimh/<dataset-name>

# Dump metadata JSON for quick review
cat data-vimh/<dataset-name>/vimh_dataset_info.json | jq
```

Key fields in `vimh_dataset_info.json`:
- `parameter_names`: Ordered list of varying parameters. Used to auto-wire heads.
- `parameter_mappings`: Min/max, scaling, and descriptions for each parameter.
- `image_size`: Dimensions (height×width×channels).

## 2. Train with Auto-Configured Heads

VIMH-ready configs declare `auto_configure_from_dataset: true`. The training entry point reads metadata and builds heads/losses automatically.

```bash
# CNN baseline with classification heads
devices=cpu python src/train.py experiment=vimh_cnn  tags='[vimh,baseline]'

# CNN on the 16K dataset with ordinal head example
python src/train.py experiment=vimh_cnn_16kdss_ordinal trainer.max_epochs=30

# EfficientNet variant
python src/train.py data=vimh model=vimh_cnn_64k trainer=gpu devices=1

# Vision Transformer variant
python src/train.py data=vimh model=cifar100_vit_210k trainer=mps
```

Tips:
- Use `trainer.max_epochs` and `+trainer.limit_*` overrides to iterate quickly.
- Inspect `log.info` output—`src/train.py` prints detected parameter names, head shapes, and loss weights.

## 3. Add Synthetic Heads to MNIST/CIFAR Datasets

Any LightningDataModule with `multihead: true` will generate synthetic heads on the fly.

```bash
# MNIST multihead example
python src/train.py experiment=multihead_cnn_mnist tags='[mnist,multihead]'

# CIFAR-10 multihead example
python src/train.py experiment=multihead_cnn_cifar10 tags='[cifar10,multihead]'
```

To tweak synthetic strategies:
1. Edit the appropriate strategy in `src/data/multihead_dataset.py`.
2. Add regression tests in `tests/test_multihead_datasets.py` to cover new heuristics.

## 4. Create a Custom VIMH Dataset

```bash
# Generate from existing audio/image assets (see scripts/examples)
python src/data/vimh_generator.py \
    --output data-vimh/my_dataset \
    --samples 1000 \
    --height 64 --width 64 --channels 3 \
    --param note_number 40 88 linear \
    --param note_velocity 0 127 linear

# Verify
python display_vimh.py data-vimh/my_dataset
```

Recommendations:
- Keep `samples` divisible by train/val/test splits used downstream (e.g., 80/10/10).
- Provide descriptive names in `--param` to ease auto-configuration.

## 5. Wire a New Model Architecture

```bash
# Copy an existing config as a starting point
cp configs/model/vimh_cnn_64k.yaml configs/model/vimh_efficientnet_small.yaml

# Edit net target and hyperparameters
vim configs/model/vimh_efficientnet_small.yaml

# Update experiment config
cp configs/experiment/vimh_cnn.yaml configs/experiment/vimh_efficientnet_small.yaml

# Run smoke test
python src/train.py experiment=vimh_efficientnet_small +trainer.limit_train_batches=5
```

Checklist:
- Ensure `model.auto_configure_from_dataset: true`.
- Set `net.input_channels: 0` (placeholder) so auto-config can adjust from metadata.
- Leave `criteria` empty when relying on auto-configured regression/classification heads.

## 6. Debug Auto-Configuration

Common failure points and fixes:

| Symptom | Fix |
|---------|-----|
| `KeyError: 'parameter_names'` | Confirm `vimh_dataset_info.json` includes `parameter_names`. Regenerate metadata or add manually. |
| `ValueError: Label preflight failed` | Run `python src/train.py ... preflight.label_diversity_batches=0` to bypass, then inspect dataset with `display_vimh.py`. |
| Wrong input channels | Delete cached `.hydra` run directory and ensure metadata reports correct `channels`. |
| Regression heads using CrossEntropy | Verify `model.output_mode` is set to `regression` in the config. |

Enable debug logging:
```bash
python src/train.py experiment=vimh_cnn logger=rich +trainer.log_every_n_steps=1
```

## 7. Export Predictions for Analysis

```bash
python src/train.py experiment=vimh_cnn test=true trainer.accelerator=cpu

# Grab latest run directory (Hydra sets it in .hydra/)
RUN_DIR=$(ls -td logs/train/runs/*vimh_cnn* | head -n 1)
python scripts/export_predictions.py --run $RUN_DIR --format csv
```

Use exports to chart per-parameter performance or feed into downstream evaluation notebooks.

## 8. Cross-Linking & Further Reading

- [vimh.md](vimh.md) – Detailed format specification
- [multihead_data_architecture.md](multihead_data_architecture.md) – Design rationale and classes
- [benchmark_snapshots.md](benchmark_snapshots.md) – Timing reference for VIMH experiments
- [tests/test_vimh_datasets.py](../tests/test_vimh_datasets.py) – Unit tests and fixtures

Want to contribute a recipe? Add a section here and include the exact command sequence plus expected metrics/logs.
