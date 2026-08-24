from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from hlt_classification.data.cache_contracts import with_content_hash
from hlt_classification.scouting.hcwdl_mhpe_tri60_d000_ensemble import (
    COMPONENTS,
    PRIMARY_ENSEMBLE_ID,
    build_d000_cross_track_report,
    validate_d000_cross_track_report,
)
from hlt_classification.scouting.hcwdl_mhpe_tri60_graph import (
    GRAPH_SHA256, NODE_REGISTRY,
)
from hlt_classification.scouting.hcwdl_mhpe_tri60_contracts import (
    TRAINING_REPORT_CONTRACT, artifact,
)


SHA = "a" * 64


def _fixture_report() -> dict[str, object]:
    labels = np.tile(np.arange(15, dtype=np.int64), 4)
    identities = np.zeros((len(labels), 32), dtype=np.uint8)
    identities[:, -2:] = (
        np.arange(len(labels), dtype=">u2").view(np.uint8).reshape(-1, 2)
    )
    component_logits = {}
    for component_index, node_id in enumerate(COMPONENTS):
        logits = np.full((len(labels), 15), -1.0, dtype=np.float32)
        logits[np.arange(len(labels)), labels] = 2.0 + component_index * 0.1
        logits[:, (component_index + 1) % 15] += 0.15
        component_logits[node_id] = logits
    lineage = {
        node_id: {
            "report_sha256": f"{index + 1:064x}",
            "checkpoint_sha256": f"{index + 11:064x}",
            "logits_sha256": f"{index + 21:064x}",
        }
        for index, node_id in enumerate(COMPONENTS)
    }
    parents = {
        "campaign_spec": SHA,
        "graph": GRAPH_SHA256,
        "recipe": "b" * 64,
        "split_manifest": "c" * 64,
        "selection_manifest": "d" * 64,
    }
    for node_id in COMPONENTS:
        parents[f"component_report/{node_id}"] = lineage[node_id][
            "report_sha256"
        ]
        parents[f"component_checkpoint/{node_id}"] = lineage[node_id][
            "checkpoint_sha256"
        ]
    return build_d000_cross_track_report(
        component_logits=component_logits,
        labels=labels,
        identity_digests=identities,
        component_lineage=lineage,
        parents=parents,
        source_campaign_spec_path=Path("campaign_spec.json"),
        producer_commit="c" * 40,
        runtime_seconds=12.5,
    )


def test_d000_cross_track_report_is_fixed_uniform_validation_only():
    report = _fixture_report()
    assert validate_d000_cross_track_report(report) == report["content_hash"]
    assert report["component_order"] == list(COMPONENTS)
    assert report["primary_ensemble"]["ensemble_id"] == PRIMARY_ENSEMBLE_ID
    assert report["primary_ensemble"]["uniform_weight"] == [1, 4]
    assert report["primary_ensemble"]["accumulation_order"] == sorted(COMPONENTS)
    assert len(report["leave_one_out_diagnostics"]) == 4
    assert {
        row["omitted_node_id"] for row in report["leave_one_out_diagnostics"]
    } == set(COMPONENTS)
    assert report["fresh_fit_count"] == 0
    assert report["persistent_prediction_arrays"] is False
    assert report["selection_eligible"] is False
    assert report["final_test_accessed"] is False


def test_d000_cross_track_components_are_exact_hlt_deployable_nodes():
    assert len(COMPONENTS) == 4
    for node_id in COMPONENTS:
        node = NODE_REGISTRY[node_id]
        assert node.coordinate_name == "D000"
        assert node.deployable is True


def test_d000_cross_track_report_rejects_semantic_tampering():
    report = _fixture_report()
    report["primary_ensemble"]["uniform_weight"] = [3, 10]
    report = with_content_hash(report)
    with pytest.raises(ValueError):
        validate_d000_cross_track_report(report)


def test_d000_cross_track_cli_help():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/evaluate_hcwdl_mhpe_tri60_d000_ensemble.py"),
            "--help",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--campaign-spec" in result.stdout
    assert "--producer-commit" in result.stdout


def test_d000_cross_track_evaluator_needs_only_four_completed_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    import torch
    from hlt_classification.scouting import (
        hcwdl_mhpe_tri60_d000_ensemble as diagnostic,
    )

    campaign_root = tmp_path / "campaign"
    spec_path = campaign_root / "campaign_spec.json"
    recipe_path = campaign_root / "recipe.json"
    spec = {
        "content_hash": SHA,
        "campaign_root": str(campaign_root),
        "artifact_paths": {"recipe": str(recipe_path)},
        "role_counts": {"validation": 30},
        "replicate_seed": 1337,
        "source_commit": "e" * 40,
        "final_test_accessed": False,
    }
    reports = {}
    for index, node_id in enumerate(COMPONENTS):
        reports[node_id] = artifact({
            "complete": True,
            "node_id": node_id,
            "node_spec": NODE_REGISTRY[node_id].payload(),
            "graph_sha256": GRAPH_SHA256,
            "campaign_spec_sha256": SHA,
            "selected_checkpoint_sha256": f"{index + 31:064x}",
        }, contract=TRAINING_REPORT_CONTRACT)

    def fake_load_json(path):
        candidate = Path(path)
        if candidate == spec_path.resolve():
            return spec
        if candidate == recipe_path:
            return {"training": {"effective_batch_size": 256}}
        if candidate.name == "training_report.json":
            return reports[candidate.parent.name]
        raise AssertionError(candidate)

    class FakeModel:
        def __init__(self, bias: float):
            self.bias = bias

        def __call__(self, features, vectors, mask):
            labels = features[:, 0, 0].long()
            logits = torch.full(
                (len(labels), 15), -1.0, dtype=torch.float32,
                device=features.device,
            )
            logits[torch.arange(len(labels)), labels] = 2.0 + self.bias
            return logits

    def fake_load_model(path, *, device):
        node_id = Path(path).parent.name
        return FakeModel(float(COMPONENTS.index(node_id)) / 10), reports[node_id]

    labels = np.tile(np.arange(15, dtype=np.int64), 2)
    features = np.zeros((len(labels), 21, 3), dtype=np.float32)
    features[:, 0, 0] = labels
    batch = {
        "hlt": SimpleNamespace(
            features=features,
            vectors=np.ones((len(labels), 4, 3), dtype=np.float32),
            mask=np.ones((len(labels), 1, 3), dtype=np.bool_),
        ),
        "labels": labels,
        "identity_keys": np.asarray([
            f"validation.root::tree::{index}" for index in range(len(labels))
        ]),
    }

    monkeypatch.setattr(diagnostic, "load_json", fake_load_json)
    monkeypatch.setattr(diagnostic, "validate_campaign", lambda *a, **k: SHA)
    monkeypatch.setattr(diagnostic, "_configure_deterministic_backend", lambda: None)
    monkeypatch.setattr(diagnostic, "_foundation", lambda value: {"fixture": True})
    monkeypatch.setattr(
        diagnostic,
        "_load_common",
        lambda value: ({}, "1" * 64, "2" * 64, {}, {}, {}),
    )
    monkeypatch.setattr(diagnostic, "validate_recipe", lambda value: "3" * 64)
    monkeypatch.setattr(diagnostic, "load_tri60_model", fake_load_model)
    monkeypatch.setattr(diagnostic, "_stream", lambda **kwargs: iter((batch,)))

    output = tmp_path / "diagnostic" / "report.json"
    report = diagnostic.evaluate_d000_cross_track_ensemble(
        campaign_spec_path=spec_path,
        output=output,
        producer_commit="f" * 40,
        device="cpu",
    )
    assert output.is_file()
    assert report["validation_rows"] == 30
    assert report["fresh_fit_count"] == 0
    assert report["primary_ensemble"]["uniform_weight"] == [1, 4]
    assert report["final_test_accessed"] is False
