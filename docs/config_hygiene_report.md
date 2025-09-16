# Config Hygiene

Run the automated audit to verify that Hydra defaults point to real YAML files.

```bash
python scripts/config_audit.py            # train/eval + all experiment configs
python scripts/config_audit.py configs/experiment/multihead_cnn_mnist.yaml  # single file
```

The script reports missing overrides in a CI-friendly format and exits non-zero when issues are detected.

Example output:
```
✓ All referenced Hydra configs resolved successfully.
```

Add this check to your workflow (e.g., pre-commit, CI pipeline) to catch renamed or deleted config files before they break experiments.
