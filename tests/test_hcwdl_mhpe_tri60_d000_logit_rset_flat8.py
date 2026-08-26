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
    FLAT8_DENOMINATOR,
    FLAT8_ENSEMBLE_ID,
    FLAT8_FAMILY_NUMERATORS,
    FLAT8_MEMBER_REGISTRY,
    REFERENCE_DISTRIBUTION,
    build_d000_logit_rset_flat8_report,
    evaluate_d000_logit_rset_flat8,
    validate_d000_logit_rset_flat8_report,
)
from hlt_classification.scouting.hcwdl_mhpe_tri60_graph import (
    ENSEMBLE_COMPONENTS, GRAPH_SHA256,
)
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
    components = {
        COMPONENTS[0]: _probabilities(labels, 1.8, 6),
        COMPONENTS[1]: _probabilities(labels, 2.0, 7),
    }
    baseline_metrics = classification_metrics(
        np.log(np.maximum(baseline_probability, 1e-30)), labels,
    )
    component_lineage = {
        COMPONENTS[0]: _lineage(10), COMPONENTS[1]: _lineage(20),
    }
    u000_lineage = _lineage(30)
    baseline_lineage = {
        "control_spec_sha256": "4" * 64, "report_sha256": "5" * 64,
    }
    parents = {
        "campaign_spec": SHA, "graph": GRAPH_SHA256, "recipe": "6" * 64,
        "split_manifest": "7" * 64, "selection_manifest": "8" * 64,
        "ce_control_spec": baseline_lineage["control_spec_sha256"],
        "ce_control_report": baseline_lineage["report_sha256"],
    }
    for node_id in (*COMPONENTS, REFERENCE_DISTRIBUTION):
        item = u000_lineage if node_id == REFERENCE_DISTRIBUTION else component_lineage[node_id]
        parents[f"probability_lock/{node_id}"] = item["lock_sha256"]
        parents[f"validation_manifest/{node_id}"] = item["manifest_sha256"]
        parents[f"stage_report/{node_id}"] = item["stage_report_sha256"]
    report = build_d000_logit_rset_flat8_report(
        component_probabilities=components,
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
    return report, labels, components


def test_flat8_registry_and_effective_weights_are_exact():
    report, *_ = _fixture_report()
    assert validate_d000_logit_rset_flat8_report(report) == report["content_hash"]
    assert FLAT8_MEMBER_REGISTRY == {
        node_id: ENSEMBLE_COMPONENTS[node_id] for node_id in COMPONENTS
    }
    assert [len(FLAT8_MEMBER_REGISTRY[name]) for name in COMPONENTS] == [5, 3]
    primary = report["primary_ensemble"]
    assert primary["ensemble_id"] == FLAT8_ENSEMBLE_ID
    assert primary["family_member_counts"] == FLAT8_FAMILY_NUMERATORS
    assert primary["effective_family_weights"] == {
        "LOGIT_D000E": [5, 8], "RSET_D000E": [3, 8],
    }
    assert primary["nominal_effective_underlying_member_weight"] == [1, 8]
    assert primary["family_bank_fp32_rounding_precedes_cross_family_blend"] is True
    assert primary["bitwise_identical_to_direct_raw_specialist_average"] is False
    assert primary["raw_specialist_reinference"] is False
    assert report["fresh_fit_count"] == 0
    assert report["scheduler_dependencies_created"] is False
    assert report["final_test_accessed"] is False


def test_flat8_matches_direct_family_bank_weighting_and_keeps_50_50_comparator():
    report, labels, components = _fixture_report()
    expected_probability = np.ascontiguousarray(
        (
            5 * components["LOGIT_D000E"].astype(np.float64)
            + 3 * components["RSET_D000E"].astype(np.float64)
        ) / FLAT8_DENOMINATOR,
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
    comparator = report["equal_family_comparator"]["primary_ensemble"]
    assert comparator["rational_weights"] == {
        node_id: [1, 2] for node_id in COMPONENTS
    }
    assert set(report["primary_delta"]) == {
        "equal_family_50_50", *COMPONENTS,
    }


def test_flat8_report_rejects_member_weight_tampering():
    report, *_ = _fixture_report()
    report["primary_ensemble"]["nominal_effective_underlying_member_weight"] = [
        1, 7,
    ]
    report = with_content_hash(report)
    with pytest.raises(ValueError):
        validate_d000_logit_rset_flat8_report(report)


def test_flat8_public_evaluator_selects_flat_builder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    from hlt_classification.scouting import (
        hcwdl_mhpe_tri60_d000_logit_rset_blend as diagnostic,
    )

    captured = {}

    def fake_evaluate(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(diagnostic, "_evaluate_d000_logit_rset", fake_evaluate)
    result = evaluate_d000_logit_rset_flat8(
        campaign_spec_path=tmp_path / "campaign.json",
        ce_control_spec_path=tmp_path / "control.json",
        output=tmp_path / "report.json",
        producer_commit=COMMIT,
    )
    assert result == {"ok": True}
    assert captured["report_builder"] is build_d000_logit_rset_flat8_report


def test_flat8_cli_help_and_worker_isolation():
    root = Path(__file__).resolve().parents[1]
    cli = root / "scripts/evaluate_hcwdl_mhpe_tri60_d000_logit_rset_flat8.py"
    result = subprocess.run(
        [sys.executable, str(cli), "--help"], cwd=root, check=False,
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--campaign-spec" in result.stdout
    assert "--ce-control-spec" in result.stdout
    worker = (
        root / "sbatch/run_hcwdl_mhpe_tri60_d000_logit_rset_flat8.sh"
    ).read_text()
    assert "--dependency" not in worker
    assert "scancel" not in worker
    assert "scontrol" not in worker
    assert "--device" not in worker
    assert "evaluate_hcwdl_mhpe_tri60_d000_logit_rset_flat8.py" in worker
