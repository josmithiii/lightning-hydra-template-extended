from pathlib import Path

import hydra
import pytest
import rootutils
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

EXPERIMENT_DIR = Path(__file__).parent.parent / "configs" / "experiment"
EXPERIMENT_NAMES = sorted(p.stem for p in EXPERIMENT_DIR.glob("*.yaml"))


@pytest.mark.parametrize("experiment", EXPERIMENT_NAMES)
def test_experiment_config_instantiates(experiment: str) -> None:
    """Compose each `configs/experiment/*.yaml` and instantiate its data module and model.

    This is a fast, non-slow guard against back-port drift in the Hydra wiring: dangling
    `_target_` paths, renamed/removed config files referenced in `defaults`, and invalid override
    keys. It complements `tests/test_sweeps.py::test_experiments`, which actually trains each
    experiment and is therefore gated behind `@pytest.mark.slow` and `@RunIf(sh=True)`.
    """
    try:
        with initialize(version_base="1.3", config_path="../configs"):
            cfg = compose(
                config_name="train.yaml",
                return_hydra_config=True,
                overrides=[f"experiment={experiment}"],
            )
            # `paths.root_dir` interpolates `${oc.env:PROJECT_ROOT}`, which rootutils sets at app
            # startup but not in a bare test; resolve it explicitly like the conftest fixtures do.
            with open_dict(cfg):
                cfg.paths.root_dir = str(rootutils.find_root(indicator=".project-root"))

        assert cfg.data
        assert cfg.model

        HydraConfig().set_config(cfg)

        # Builds the data module and the LightningModule (including its `net`); optimizer/scheduler
        # are `_partial_` and are not invoked here. No training or data download occurs.
        hydra.utils.instantiate(cfg.data)
        hydra.utils.instantiate(cfg.model)
    finally:
        GlobalHydra.instance().clear()
