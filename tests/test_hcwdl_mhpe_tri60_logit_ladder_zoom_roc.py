from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from hlt_classification.scouting.evaluation import softmax
from hlt_classification.scouting.hcwdl_mhpe_tri60_dense_graph import (
    GRAPH_SHA256 as DENSE_GRAPH_SHA256,
)
from hlt_classification.scouting.hcwdl_mhpe_tri60_graph import (
    GRAPH_SHA256 as SOURCE_GRAPH_SHA256,
)
from hlt_classification.scouting.hcwdl_mhpe_tri60_logit_ladder_zoom_roc import (
    DISPLAY_ORDER,
    FIGURE_FILENAMES,
    LEGEND_D_ORDER,
    REPORT_CONTRACT,
    SOURCE_REGISTRY,
    X_RANGE,
    evaluate_logit_ladder_zoom_roc,
    validate_logit_ladder_zoom_roc_report,
)
from hlt_classification.scouting.hcwdl_representation_data import (
    canonical_identity_digests,
)


SOURCE_SHA = "1" * 64
DENSE_SHA = "2" * 64
COMMIT = "a" * 40


def _probabilities(labels: np.ndarray, strength: float, bias: int) -> np.ndarray:
    logits = np.full((len(labels), 15), -0.6, dtype=np.float32)
    logits[np.arange(len(labels)), labels] = strength
    logits[:, bias] += 0.1
    return np.ascontiguousarray(softmax(logits), dtype=np.float32)


def _lineage(seed: int, *, checkpoint: bool = False) -> dict[str, str]:
    names = (
        ("report_sha256", "checkpoint_sha256", "probabilities_sha256")
        if checkpoint else
        ("lock_sha256", "manifest_sha256", "stage_report_sha256", "probabilities_sha256")
    )
    return {name: f"{seed + index:064x}" for index, name in enumerate(names, 1)}


def test_zoom_registry_preserves_requested_display_aliases_and_order():
    assert DISPLAY_ORDER == (
        "Offline", "D100", "D080", "D060", "D040", "D020", "D000",
    )
    assert LEGEND_D_ORDER == ("D100", "D080", "D060", "D040", "D020", "D000")
    assert {
        name: row["artifact_id"] for name, row in SOURCE_REGISTRY.items()
    } == {
        "Offline": "U000",
        "D100": "LOGIT_U100E",
        "D080": "DX_LOGIT_D083E",
        "D060": "LOGIT_U100_from_U050E",
        "D040": "DX_LOGIT_D033E",
        "D020": "DX_LOGIT_D083_from_LOGIT_U100E",
        "D000": "LOGIT_D000E",
    }
    assert X_RANGE == (0.30, 0.50)


def test_zoom_evaluator_writes_combined_and_individual_figures(
    tmp_path: Path, monkeypatch,
):
    from hlt_classification.scouting import (
        hcwdl_mhpe_tri60_logit_ladder_zoom_roc as diagnostic,
    )

    labels = np.tile(np.arange(15, dtype=np.int64), 32)
    keys = tuple(f"validation.root::{index}" for index in range(len(labels)))
    identities = canonical_identity_digests(keys)
    probability = {
        alias: _probabilities(labels, 1.0 + 0.2 * index, (index + 3) % 15)
        for index, alias in enumerate(DISPLAY_ORDER)
    }

    source_root = tmp_path / "source"
    dense_root = tmp_path / "dense"
    source_path = source_root / "campaign_spec.json"
    dense_path = dense_root / "campaign_spec.json"
    recipe_path = source_root / "recipe.json"
    foundation_sha = "3" * 64
    recipe_sha = "4" * 64
    source = {
        "content_hash": SOURCE_SHA,
        "campaign_root": str(source_root),
        "artifact_paths": {"recipe": str(recipe_path)},
        "parents": {
            "graph": SOURCE_GRAPH_SHA256,
            "foundation": foundation_sha,
            "recipe": recipe_sha,
        },
        "replicate_seed": 7,
        "role_counts": {"train": 1000, "validation": len(labels)},
        "final_test_accessed": False,
    }
    dense = {
        "content_hash": DENSE_SHA,
        "campaign_root": str(dense_root),
        "artifact_paths": {},
        "parents": {
            "graph": DENSE_GRAPH_SHA256,
            "source_campaign": SOURCE_SHA,
            "foundation": foundation_sha,
            "source_recipe": recipe_sha,
        },
        "replicate_seed": 7,
        "role_counts": dict(source["role_counts"]),
        "final_test_accessed": False,
    }
    recipe = {
        "content_hash": recipe_sha,
        "training": {"effective_batch_size": 256},
    }

    def fake_json(path):
        candidate = Path(path)
        if candidate == source_path.resolve():
            return source
        if candidate == dense_path.resolve():
            return dense
        if candidate == recipe_path:
            return recipe
        raise AssertionError(candidate)

    source_alias_by_id = {
        row["artifact_id"]: alias for alias, row in SOURCE_REGISTRY.items()
        if row["origin"] == "source" and row["kind"] == "distribution"
    }
    dense_alias_by_id = {
        row["artifact_id"]: alias for alias, row in SOURCE_REGISTRY.items()
        if row["origin"] == "dense" and row["kind"] == "distribution"
    }

    def fake_source_bank(*, root, distribution_id, spec):
        alias = source_alias_by_id[distribution_id]
        order = np.arange(len(labels))[::-1] if alias == "D100" else np.arange(len(labels))
        return identities[order], probability[alias][order], _lineage(10 + len(alias))

    def fake_dense_bank(*, root, distribution_id, spec):
        alias = dense_alias_by_id[distribution_id]
        order = np.roll(np.arange(len(labels)), 11)
        return identities[order], probability[alias][order], _lineage(30 + len(alias))

    def fake_source_specialist(**kwargs):
        order = np.roll(np.arange(len(labels)), 7)
        return (
            identities[order], probability["D060"][order], labels[order],
            _lineage(50, checkpoint=True),
        )

    def fake_dense_specialist(**kwargs):
        order = np.roll(np.arange(len(labels)), 13)
        return (
            identities[order], probability["D020"][order], labels[order],
            _lineage(60, checkpoint=True),
        )

    monkeypatch.setattr(diagnostic, "load_json", fake_json)
    monkeypatch.setattr(
        diagnostic, "validate_source_campaign", lambda *a, **k: SOURCE_SHA,
    )
    monkeypatch.setattr(
        diagnostic, "validate_dense_campaign", lambda *a, **k: DENSE_SHA,
    )
    monkeypatch.setattr(diagnostic, "_foundation", lambda value: {"fixture": True})
    monkeypatch.setattr(
        diagnostic, "_load_common",
        lambda value: ({}, "5" * 64, "6" * 64, {}, {}, {}),
    )
    monkeypatch.setattr(diagnostic, "validate_recipe", lambda value: recipe_sha)
    monkeypatch.setattr(diagnostic, "_source_validation_bank", fake_source_bank)
    monkeypatch.setattr(diagnostic, "_dense_validation_bank", fake_dense_bank)
    monkeypatch.setattr(diagnostic, "_source_specialist", fake_source_specialist)
    monkeypatch.setattr(diagnostic, "_dense_specialist", fake_dense_specialist)

    output = tmp_path / "output"
    report = evaluate_logit_ladder_zoom_roc(
        source_campaign_spec_path=source_path,
        dense_campaign_spec_path=dense_path,
        output_dir=output,
        producer_commit=COMMIT,
        device="cpu",
    )

    assert report["contract"] == REPORT_CONTRACT
    assert validate_logit_ladder_zoom_roc_report(report) == report["content_hash"]
    assert report["display_order"] == list(DISPLAY_ORDER)
    assert report["legend_d_order"] == list(LEGEND_D_ORDER)
    assert report["signal_efficiency_range"] == [0.3, 0.5]
    assert report["source_registry"] == SOURCE_REGISTRY
    assert report["persistent_prediction_arrays"] is False
    assert report["final_test_accessed"] is False
    for name, filename in FIGURE_FILENAMES.items():
        payload = (output / filename).read_bytes()
        if name.endswith("png"):
            assert payload.startswith(b"\x89PNG")
        else:
            assert payload.startswith(b"%PDF")
    stored = json.loads((output / diagnostic.REPORT_FILENAME).read_text())
    assert stored["content_hash"] == report["content_hash"]
    assert (output / diagnostic.CURVE_FILENAME).is_file()


def test_zoom_cli_help_and_worker_are_independent():
    root = Path(__file__).resolve().parents[1]
    cli = root / "scripts/plot_hcwdl_mhpe_tri60_logit_ladder_zoom_roc.py"
    result = subprocess.run(
        [sys.executable, str(cli), "--help"], cwd=root,
        check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--source-campaign-spec" in result.stdout
    assert "--dense-campaign-spec" in result.stdout
    assert "--output-dir" in result.stdout
    assert "--device" in result.stdout

    worker = (
        root / "sbatch/run_hcwdl_mhpe_tri60_logit_ladder_zoom_roc.sh"
    ).read_text()
    assert "--dependency" not in worker
    assert "scancel" not in worker
    assert "scontrol" not in worker
    assert "HCWDL_UB_VIEW_SOURCE_BACKEND=process" in worker
    assert "--device cuda" in worker
