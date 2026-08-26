from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from hlt_classification.data.cache_contracts import with_content_hash
from hlt_classification.scouting.evaluation import classification_metrics, softmax
from hlt_classification.scouting.hcwdl_mhpe_tri60_d000_logit_rset_blend import (
    COMPONENTS,
    PRIMARY_ENSEMBLE_ID,
    REFERENCE_DISTRIBUTION,
    build_d000_logit_rset_blend_report,
    validate_d000_logit_rset_blend_report,
)
from hlt_classification.scouting.hcwdl_mhpe_tri60_graph import GRAPH_SHA256
from hlt_classification.scouting.hcwdl_representation_data import (
    canonical_identity_digests,
)


SHA = "a" * 64
COMMIT = "b" * 40


def _probabilities(labels: np.ndarray, strength: float, bias: int) -> np.ndarray:
    logits = np.full((len(labels), 15), -0.5, dtype=np.float32)
    logits[np.arange(len(labels)), labels] = strength
    logits[:, bias] += 0.2
    return np.ascontiguousarray(softmax(logits), dtype=np.float32)


def _lineage(offset: int) -> dict[str, str]:
    return {
        "lock_sha256": f"{offset + 1:064x}",
        "manifest_sha256": f"{offset + 2:064x}",
        "stage_report_sha256": f"{offset + 3:064x}",
        "probabilities_sha256": f"{offset + 4:064x}",
    }


def _fixture_report():
    labels = np.tile(np.arange(15, dtype=np.int64), 8)
    keys = tuple(f"validation.root::tree::{index}" for index in range(len(labels)))
    identities = canonical_identity_digests(keys)
    baseline_probability = _probabilities(labels, 1.2, 4)
    oracle_probability = _probabilities(labels, 3.0, 5)
    component_probability = {
        COMPONENTS[0]: _probabilities(labels, 1.8, 6),
        COMPONENTS[1]: _probabilities(labels, 2.0, 7),
    }
    baseline_metrics = classification_metrics(
        np.log(np.maximum(baseline_probability, 1e-30)), labels,
    )
    component_lineage = {
        COMPONENTS[0]: _lineage(10),
        COMPONENTS[1]: _lineage(20),
    }
    u000_lineage = _lineage(30)
    baseline_lineage = {
        "control_spec_sha256": "4" * 64,
        "report_sha256": "5" * 64,
    }
    parents = {
        "campaign_spec": SHA,
        "graph": GRAPH_SHA256,
        "recipe": "6" * 64,
        "split_manifest": "7" * 64,
        "selection_manifest": "8" * 64,
        "ce_control_spec": baseline_lineage["control_spec_sha256"],
        "ce_control_report": baseline_lineage["report_sha256"],
    }
    for node_id in (*COMPONENTS, REFERENCE_DISTRIBUTION):
        item = (
            u000_lineage if node_id == REFERENCE_DISTRIBUTION
            else component_lineage[node_id]
        )
        parents[f"probability_lock/{node_id}"] = item["lock_sha256"]
        parents[f"validation_manifest/{node_id}"] = item["manifest_sha256"]
        parents[f"stage_report/{node_id}"] = item["stage_report_sha256"]
    report = build_d000_logit_rset_blend_report(
        component_probabilities=component_probability,
        u000_probabilities=oracle_probability,
        m0ce60_metrics=baseline_metrics,
        labels=labels,
        identity_digests=identities,
        component_lineage=component_lineage,
        u000_lineage=u000_lineage,
        baseline_lineage=baseline_lineage,
        parents=parents,
        source_campaign_spec_path="source/campaign_spec.json",
        ce_control_spec_path="control/control_spec.json",
        producer_commit=COMMIT,
        runtime_seconds=4.5,
    )
    return report, labels, identities, baseline_metrics, oracle_probability, component_probability


def test_logit_rset_blend_is_exact_fixed_validation_only():
    report, *_ = _fixture_report()
    assert validate_d000_logit_rset_blend_report(report) == report["content_hash"]
    primary = report["primary_ensemble"]
    assert primary["ensemble_id"] == PRIMARY_ENSEMBLE_ID
    assert primary["component_order"] == list(COMPONENTS)
    assert primary["accumulation_order"] == sorted(COMPONENTS)
    assert primary["rational_weights"] == {
        node_id: [1, 2] for node_id in COMPONENTS
    }
    assert report["fresh_fit_count"] == 0
    assert report["persistent_prediction_arrays"] is False
    assert report["source_campaign_outputs_mutated"] is False
    assert report["scheduler_dependencies_created"] is False
    assert report["final_test_accessed"] is False


def test_logit_rset_blend_matches_direct_fp64_probability_average():
    report, labels, _, _, _, components = _fixture_report()
    expected_probability = np.ascontiguousarray(
        (
            components[COMPONENTS[0]].astype(np.float64)
            + components[COMPONENTS[1]].astype(np.float64)
        ) / 2,
        dtype=np.float32,
    )
    expected = classification_metrics(
        np.log(np.maximum(expected_probability, 1e-30)), labels,
    )
    actual = report["primary_ensemble"]["metrics"]
    assert actual["accuracy"] == expected["accuracy"]
    assert actual["macro_ovr_auc"] == expected["macro_ovr_auc"]
    assert actual["macro_mean_log_qcd_rejection_at_50pct_signal"] == expected[
        "macro_mean_log_qcd_rejection_at_50pct_signal"
    ]
    recovery = report["primary_ensemble"]["recovery_m0ce60_to_u000"]
    assert recovery["convention"] == "M0CE60_zero_U000_one_v1"
    assert set(recovery["per_class_r50_linear"]) == {
        "Xbb", "Xcc", "Xss", "Xqq", "Xbs", "Xgg", "Xee", "Xmm",
        "Xtauhtaue", "Xtauhtaum", "Xtauhtauh", "Xbc", "Xcs", "Xud",
    }


def test_logit_rset_blend_report_rejects_weight_tampering():
    report, *_ = _fixture_report()
    report["primary_ensemble"]["rational_weights"][COMPONENTS[0]] = [3, 5]
    report = with_content_hash(report)
    with pytest.raises(ValueError):
        validate_d000_logit_rset_blend_report(report)


def test_logit_rset_blend_cli_help():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(
                root
                / "scripts/evaluate_hcwdl_mhpe_tri60_d000_logit_rset_blend.py"
            ),
            "--help",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--campaign-spec" in result.stdout
    assert "--ce-control-spec" in result.stdout
    assert "--producer-commit" in result.stdout


def test_logit_rset_evaluator_reads_banks_and_writes_only_own_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    from hlt_classification.scouting import (
        hcwdl_mhpe_tri60_d000_logit_rset_blend as diagnostic,
    )

    labels = np.tile(np.arange(15, dtype=np.int64), 3)
    keys = tuple(f"validation.root::tree::{index}" for index in range(len(labels)))
    identities = canonical_identity_digests(keys)
    probability = {
        COMPONENTS[0]: _probabilities(labels, 1.8, 6),
        COMPONENTS[1]: _probabilities(labels, 2.0, 7),
        REFERENCE_DISTRIBUTION: _probabilities(labels, 3.0, 8),
    }
    baseline = classification_metrics(
        np.log(np.maximum(_probabilities(labels, 1.2, 4), 1e-30)), labels,
    )
    spec_path = tmp_path / "source" / "campaign_spec.json"
    recipe_path = tmp_path / "source" / "recipe.json"
    spec = {
        "content_hash": SHA,
        "campaign_root": str(tmp_path / "source"),
        "artifact_paths": {"recipe": str(recipe_path)},
        "parents": {"graph": GRAPH_SHA256, "recipe": "6" * 64},
        "role_counts": {"validation": len(labels)},
        "replicate_seed": 1337,
        "final_test_accessed": False,
    }

    def fake_load_json(path):
        candidate = Path(path)
        if candidate == spec_path.resolve():
            return spec
        if candidate == recipe_path:
            return {"training": {"effective_batch_size": 256}}
        raise AssertionError(candidate)

    def fake_bank(*, root, distribution_id, spec):
        return identities, probability[distribution_id], _lineage(
            40 + 10 * (*COMPONENTS, REFERENCE_DISTRIBUTION).index(distribution_id)
        )

    batch = {"identity_keys": np.asarray(keys), "labels": labels}
    monkeypatch.setattr(diagnostic, "load_json", fake_load_json)
    monkeypatch.setattr(diagnostic, "validate_campaign", lambda *a, **k: SHA)
    monkeypatch.setattr(diagnostic, "_foundation", lambda value: {"fixture": True})
    monkeypatch.setattr(
        diagnostic, "_load_common",
        lambda value: ({}, "7" * 64, "8" * 64, {}, {}, {}),
    )
    monkeypatch.setattr(diagnostic, "validate_recipe", lambda value: "6" * 64)
    monkeypatch.setattr(diagnostic, "_load_validation_bank", fake_bank)
    monkeypatch.setattr(diagnostic, "_stream", lambda **kwargs: iter((batch,)))
    monkeypatch.setattr(
        diagnostic, "_load_m0ce60_reference",
        lambda **kwargs: (
            baseline,
            {"control_spec_sha256": "4" * 64, "report_sha256": "5" * 64},
        ),
    )

    output = tmp_path / "diagnostic" / "validation_report.json"
    report = diagnostic.evaluate_d000_logit_rset_blend(
        campaign_spec_path=spec_path,
        ce_control_spec_path=tmp_path / "control" / "control_spec.json",
        output=output,
        producer_commit=COMMIT,
    )
    assert output.is_file()
    assert report["validation_rows"] == len(labels)
    assert report["primary_ensemble"]["rational_weights"] == {
        node_id: [1, 2] for node_id in COMPONENTS
    }
    assert list(tmp_path.rglob("*.json")) == [output]


def test_logit_rset_worker_has_no_gpu_dependency_or_campaign_mutation_commands():
    root = Path(__file__).resolve().parents[1]
    worker = (
        root / "sbatch/run_hcwdl_mhpe_tri60_d000_logit_rset_blend.sh"
    ).read_text()
    assert "--dependency" not in worker
    assert "scancel" not in worker
    assert "scontrol" not in worker
    assert "--device" not in worker
    assert "evaluate_hcwdl_mhpe_tri60_d000_logit_rset_blend.py" in worker
