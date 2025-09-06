# Onboarding in 15 Minutes

Fast path for power users to validate the setup, explore capabilities, and run a meaningful experiment.

## 0–3 min: Setup

- Activate env: `source .venv/bin/activate` (run `sh setup.sh` first if needed).
- Discover commands: `make h` and list configs: `make lc`.

## 3–6 min: Validate

- Fast tests: `make t` (excludes slow tests).
- Smoke-train all core archs: `make tqa` (1 epoch, limited batches).
- Launch TensorBoard: `make tensorboard` and open http://localhost:6006.

## 6–10 min: Compare Architectures

- Short comparison run: `make zca` (3 epochs each; check logs under `logs/train/runs/`).
- Tip: Use `trainer=mps` on macOS; `trainer=gpu` if you have CUDA.

## 10–13 min: Run a Reproducible Experiment

- MNIST CNN baseline: `make ecm` (or `python src/train.py experiment=cnn_mnist`).
- CIFAR quick validation suite: `make cbqa` (optional, ~15 min).

## 13–15 min: Customize with Hydra

- Switch model: `python src/train.py model=mnist_vit_38k trainer.max_epochs=1`.
- Override hyperparams: `python src/train.py model.optimizer.lr=1e-3 data.batch_size=128`.
- Add quick limits: `python src/train.py +trainer.limit_train_batches=10 +trainer.limit_val_batches=5`.

## Expectations & Notes

- MNIST CNN reaches ~99.1% (full run); ViT can reach 99.5% (longer).
- MNIST/CIFAR auto-download on first use; allow time.
- Checkpoints must be local (remote URL loading is disabled).
- VIMH on macOS MPS: set `num_workers: 0` if you see DataLoader issues.
