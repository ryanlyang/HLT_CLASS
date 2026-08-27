from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from hlt_classification.scouting.evaluation import softmax
from hlt_classification.scouting.hcwdl_mhpe_tri60_graph import GRAPH_SHA256
from hlt_classification.scouting.hcwdl_mhpe_tri60_original_logit_roc import (
    MODEL_ORDER,
    REPORT_CONTRACT,
    SIGNALS,
    _align_labels,
    evaluate_original_logit_d000e_roc,
    validate_original_logit_d000e_roc_report,
)
from hlt_classification.scouting.hcwdl_representation_data import (
    canonical_identity_digests,
)


SHA = "a" * 64
COMMIT = "b" * 40


def _probabilities(labels: np.ndarray, strength: float, bias: int) -> np.ndarray:
    logits = np.full((len(labels), 15), -0.7, dtype=np.float32)
    logits[np.arange(len(labels)), labels] = strength
    logits[:, bias] += 0.15
    return np.ascontiguousarray(softmax(logits), dtype=np.float32)


def _bank_lineage(offset: int) -> dict[str, str]:
    return {
        "lock_sha256": f"{offset + 1:064x}",
        "manifest_sha256": f"{offset + 2:064x}",
        "stage_report_sha256": f"{offset + 3:064x}",
        "probabilities_sha256": f"{offset + 4:064x}",
    }


def test_original_logit_roc_aligns_labels_by_canonical_identity():
    keys = tuple(f"validation.root::{index}" for index in range(4))
    identities = canonical_identity_digests(keys)
    labels = np.array([0, 1, 2, 1], dtype=np.int64)
    order = np.array([2, 0, 3, 1])
    actual = _align_labels(identities, identities[order], labels[order])
    np.testing.assert_array_equal(actual, labels)


def test_original_logit_roc_evaluator_writes_exact_curves_and_figures(
    tmp_path: Path, monkeypatch,
):
    from hlt_classification.scouting import (
        hcwdl_mhpe_tri60_original_logit_roc as diagnostic,
    )

    labels = np.tile(np.array([0, 1, 2], dtype=np.int64), 30)
    keys = tuple(f"validation.root::{index}" for index in range(len(labels)))
    identities = canonical_identity_digests(keys)
    probabilities = {
        "M0CE60": _probabilities(labels, 1.0, 4),
        "LOGIT_D000E": _probabilities(labels, 1.8, 5),
        "U000": _probabilities(labels, 2.7, 6),
    }
    source_root = tmp_path / "source"
    spec_path = source_root / "campaign_spec.json"
    recipe_path = source_root / "recipe.json"
    spec = {
        "content_hash": SHA,
        "campaign_root": str(source_root),
        "artifact_paths": {"recipe": str(recipe_path)},
        "parents": {"graph": GRAPH_SHA256, "recipe": "6" * 64},
        "role_counts": {"validation": len(labels)},
        "ordinary_access_roles": ["train", "validation"],
        "ordinary_final_test_capability": False,
        "final_test_accessed": False,
    }
    recipe = {"content_hash": "6" * 64}

    def fake_load_json(path):
        candidate = Path(path)
        if candidate == spec_path.resolve():
            return spec
        if candidate == recipe_path:
            return recipe
        raise AssertionError(candidate)

    def fake_bank(*, root, distribution_id, spec):
        offset = 10 if distribution_id == "LOGIT_D000E" else 20
        order = np.arange(len(labels))[::-1] if distribution_id == "U000" else np.arange(len(labels))
        return (
            identities[order], probabilities[distribution_id][order],
            _bank_lineage(offset),
        )

    monkeypatch.setattr(diagnostic, "load_json", fake_load_json)
    monkeypatch.setattr(diagnostic, "validate_campaign", lambda *a, **k: SHA)
    monkeypatch.setattr(diagnostic, "_foundation", lambda value: {"fixture": True})
    monkeypatch.setattr(
        diagnostic, "_load_common",
        lambda value: ({}, "7" * 64, "8" * 64, {}, {}, {}),
    )
    monkeypatch.setattr(diagnostic, "validate_recipe", lambda value: "6" * 64)
    monkeypatch.setattr(diagnostic, "_load_validation_bank", fake_bank)
    monkeypatch.setattr(
        diagnostic, "_m0ce60_validation_probabilities",
        lambda **kwargs: (
            identities[::-1], probabilities["M0CE60"][::-1], labels[::-1],
            {
                "control_spec_sha256": "1" * 64,
                "report_sha256": "2" * 64,
                "checkpoint_sha256": "3" * 64,
                "probabilities_sha256": "4" * 64,
            },
        ),
    )

    output = tmp_path / "diagnostic"
    report = evaluate_original_logit_d000e_roc(
        campaign_spec_path=spec_path,
        ce_control_spec_path=tmp_path / "control/control_spec.json",
        output_dir=output,
        producer_commit=COMMIT,
        device="cpu",
    )

    assert report["contract"] == REPORT_CONTRACT
    assert validate_original_logit_d000e_roc_report(report) == report["content_hash"]
    assert report["models"] == list(MODEL_ORDER)
    assert report["signals"] == list(SIGNALS)
    assert report["persistent_prediction_arrays"] is False
    assert report["curve_arrays_only"] is True
    assert report["final_test_accessed"] is False
    assert all(
        count <= 4098
        for rows in report["stored_curve_point_counts"].values()
        for count in rows.values()
    )
    assert (output / "original_logit_d000e_hbb_hcc_curves.npz").is_file()
    assert (output / "original_logit_d000e_hbb_hcc_rejection.pdf").read_bytes().startswith(b"%PDF")
    assert (output / "original_logit_d000e_hbb_hcc_rejection.png").read_bytes().startswith(b"\x89PNG")
    stored = json.loads(
        (output / "original_logit_d000e_hbb_hcc_report.json").read_text()
    )
    assert stored["content_hash"] == report["content_hash"]
    assert set(stored["working_points"]) == {"Xbb", "Xcc"}
    assert set(stored["working_points"]["Xbb"]) == {"30pct", "50pct", "80pct"}


def test_original_logit_roc_cli_help_and_worker_is_isolated():
    root = Path(__file__).resolve().parents[1]
    cli = root / "scripts/plot_hcwdl_mhpe_tri60_original_logit_d000e_roc.py"
    result = subprocess.run(
        [sys.executable, str(cli), "--help"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--campaign-spec" in result.stdout
    assert "--ce-control-spec" in result.stdout
    assert "--output-dir" in result.stdout
    assert "--device" in result.stdout

    worker = (
        root / "sbatch/run_hcwdl_mhpe_tri60_original_logit_d000e_roc.sh"
    ).read_text()
    assert "--dependency" not in worker
    assert "scancel" not in worker
    assert "scontrol" not in worker
    assert "--device cuda" in worker
    assert "plot_hcwdl_mhpe_tri60_original_logit_d000e_roc.py" in worker
