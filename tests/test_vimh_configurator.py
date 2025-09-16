import json

from omegaconf import OmegaConf

from src.utils.vimh_config import configure_model_from_metadata


def _write_metadata(tmp_path):
    metadata = {
        "height": 32,
        "width": 32,
        "channels": 3,
        "parameter_names": ["note_number", "note_velocity"],
        "parameter_mappings": {
            "note_number": {"min": 40.0, "max": 88.0},
            "note_velocity": {"min": 0.0, "max": 127.0},
        },
    }
    (tmp_path / "vimh_dataset_info.json").write_text(json.dumps(metadata))


def test_configure_model_from_metadata_classification(tmp_path):
    _write_metadata(tmp_path)
    cfg = OmegaConf.create(
        {
            "model": {
                "auto_configure_from_dataset": True,
                "output_mode": "classification",
                "loss_weights": {},
                "net": {"input_channels": 1, "heads_config": None},
            }
        }
    )

    configure_model_from_metadata(cfg, str(tmp_path))

    assert cfg.model.net.input_channels == 3
    assert cfg.model.net.heads_config == {"note_number": 256, "note_velocity": 256}
    assert cfg.model.loss_weights == {"note_number": 1.0, "note_velocity": 1.0}


def test_configure_model_from_metadata_regression(tmp_path):
    _write_metadata(tmp_path)
    cfg = OmegaConf.create(
        {
            "model": {
                "auto_configure_from_dataset": True,
                "output_mode": "regression",
                "loss_weights": {},
                "net": {
                    "input_channels": 1,
                    "heads_config": {"note_number": 10},
                    "parameter_names": [],
                },
            }
        }
    )

    configure_model_from_metadata(cfg, str(tmp_path))

    assert cfg.model.net.parameter_names == ["note_number", "note_velocity"]
    assert cfg.model.net.output_mode == "regression"
    assert cfg.model.net.heads_config is None
    for crit in cfg.model.criteria.values():
        assert crit["_target_"] == "src.models.losses.NormalizedRegressionLoss"
    assert set(cfg.model.criteria.keys()) == {"note_number", "note_velocity"}
