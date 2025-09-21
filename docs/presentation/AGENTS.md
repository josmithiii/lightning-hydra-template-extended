# Repository Guidelines

## Project Structure & Module Organization
- Source in `src/`; key entrypoints are `src/train.py` for training runs and `src/eval.py` for evaluation.
- Data utilities live under `src/data/`, models in `src/models/`, and shared helpers in `src/utils/`.
- Configuration lives in `configs/` with Hydra YAML grouped by `data/`, `model/`, `trainer/`, and `experiment/`.
- Tests reside in `tests/` (pytest), while datasets, logs, figures, and docs are stored in `data/`, `logs/`, `viz/`, and `docs/` respectively.

## Build, Test, and Development Commands
- `make tq` — smoke-train a tiny MNIST run to verify the pipeline.
- `python src/train.py experiment=<name>` — launch a Hydra experiment such as `experiment=cnn_mnist`.
- `python src/train.py model=mnist_vit_38k data=mnist trainer=gpu` — combine config overrides for targeted runs.
- `make test` (or `pytest -k "not slow"`) — execute the fast test suite; use `make test-all` for the full matrix.
- `make format` — run black, isort, flake8, bandit, and docformatter to enforce style and security checks.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indents and black line length 99; ensure public APIs carry type hints.
- Modules, functions, and variables use `snake_case`; classes are `CamelCase`; constants are `UPPER_CASE`.
- Config names and Hydra group choices use lower_snake (e.g., `mnist_vit_38k`).

## Testing Guidelines
- Tests live in `tests/test_*.py`; name new tests `test_*` and mark long runs with `@pytest.mark.slow`.
- Keep assertions meaningful; avoid network usage or heavyweight downloads in unit tests.
- Run `pytest -q` locally before commits, then `make test-all` when altering training behavior.

## Commit & Pull Request Guidelines
- Write commits in imperative mood (e.g., `train: add vit scheduler`) with concise subjects.
- PRs must describe changes, list Hydra commands used, attach sample logs from `logs/train/runs/`, and link issues.
- Check that tests pass, `make format` is clean, and configs are reproducible; pin overrides in the PR body.

## Security & Configuration Tips
- Load checkpoints only from local paths; remote URLs are blocked.
- Use `.env.example`, `requirements.txt`, and `environment.yaml` as references for local setup.
- Prefer Hydra overrides (e.g., `trainer.max_epochs=3`) over editing YAML files directly.
