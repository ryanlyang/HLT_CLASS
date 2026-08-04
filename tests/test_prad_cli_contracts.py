from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import sys


def test_teacher_cli_arguments_bind_exact_engine_signature(
    tmp_path: Path, monkeypatch
) -> None:
    script_path = Path(__file__).parents[1] / "scripts" / "train_prad_teacher.py"
    module_spec = importlib.util.spec_from_file_location(
        "train_prad_teacher_contract", script_path
    )
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    real_engine = module.train_prad_teacher
    signature = inspect.signature(real_engine)
    assert "model_factory" in signature.parameters
    assert "model" not in signature.parameters
    captured = {}

    def validating_engine(**kwargs):
        signature.bind(**kwargs)
        captured.update(kwargs)
        return {"complete": True}

    monkeypatch.setattr(
        module,
        "load_json",
        lambda _path: {
            "teacher_offline": {"positive_weights": [1.0, 1.0, 1.0]}
        },
    )
    monkeypatch.setattr(module, "PradCacheDataset", lambda path: Path(path))
    monkeypatch.setattr(module, "train_prad_teacher", validating_engine)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script_path),
            "--train-paired-cache",
            str(tmp_path / "train_paired"),
            "--train-target-cache",
            str(tmp_path / "train_targets"),
            "--validation-paired-cache",
            str(tmp_path / "val_paired"),
            "--validation-target-cache",
            str(tmp_path / "val_targets"),
            "--semantic-weights",
            str(tmp_path / "weights.json"),
            "--output-dir",
            str(tmp_path / "output"),
            "--source-snapshot-sha256",
            "a" * 64,
            "--device",
            "cpu",
            "--amp-dtype",
            "none",
        ],
    )

    assert module.main() == 0
    assert callable(captured["model_factory"])
    assert captured["source_snapshot_sha256"] == "a" * 64
