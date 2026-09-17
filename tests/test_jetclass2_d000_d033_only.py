from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from hlt_classification.data.cache_contracts import load_json, with_content_hash
from hlt_classification.jetclass2_delphes import d000_d033_only as study
from hlt_classification.jetclass2_delphes.contracts import artifact
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities


def _metrics(seed: int):
    labels = np.repeat(np.arange(11), 20)
    rng = np.random.default_rng(1907)
    raw = rng.random((len(labels), 11), dtype=np.float32)
    raw[np.arange(len(labels)), labels] += .35 * seed
    return evaluate_probabilities(labels, raw / raw.sum(axis=1, keepdims=True))


def _source(tmp_path: Path):
    d000 = dict(
        node_id=study.MT20_ENDPOINT_ID, coordinate="D000",
        teacher=study.TEACHER_ID, teachers=[], branch="COARSE",
        u=[1, 1], f=[1, 1], initialization_seed=123,
        sampler_seed=456, deployable=True,
    )
    source = dict(
        content_hash="a" * 64, source_commit="b" * 40,
        campaign_root=str((tmp_path / "source").resolve()),
        data_root=str((tmp_path / "data").resolve()),
        project_dir=str((tmp_path / "old-project").resolve()),
        source_lock={"foundation_root": str((tmp_path / "foundation").resolve())},
        foundation={"content_hash": "c" * 64}, runtime_profile={"workers": 8},
        scientific_plan={"nodes": [d000]},
    )
    names = ("M0HLT", "U000", study.TEACHER_ID, study.MT20_ENDPOINT_ID)
    references = [dict(
        node_id=name, task_sha256=f"{index + 1:x}" * 64,
        training_report_sha256=f"{index + 5:x}" * 64,
        validation=_metrics(index + 1), passes=60, selected_pass=20 + index,
    ) for index, name in enumerate(names)]
    evidence = dict(
        source_campaign_sha256=source["content_hash"],
        source_commit=source["source_commit"], references=references,
        reducer_task_sha256="9" * 64, teacher_report_sha256="8" * 64,
        teacher_bank_manifest_sha256="7" * 64,
        teacher_bank_path="attempts/reduce/train_bank",
        foundation_sha256=source["foundation"]["content_hash"],
        final_test_accessed=False,
    )
    return source, evidence


@pytest.fixture
def specification(tmp_path, monkeypatch):
    source, evidence = _source(tmp_path)
    monkeypatch.setattr(study, "_source", lambda *args: None)
    monkeypatch.setattr(study, "source_evidence", lambda path: (source, evidence))
    project = tmp_path / "project"
    root = tmp_path / "output"
    return study.create(
        source_spec=tmp_path / "source-spec.json", output_root=root,
        project=project, source_commit="d" * 40,
    )


def test_exact_scientific_delta_and_paired_d000_seed(specification):
    spec = specification
    assert spec["recipe"]["ce_weight"] == .25
    assert spec["recipe"]["kd_weight"] == .75
    assert spec["recipe"]["temperature"] == 2.
    assert spec["node"]["coordinate"] == "D000"
    assert spec["node"]["teacher"] == study.TEACHER_ID
    assert spec["node"]["teachers"] == [{
        "node_id": study.TEACHER_ID,
        "loss_weight": {"numerator": 3, "denominator": 4},
    }]
    assert spec["node"]["initialization_seed"] == 123
    assert spec["node"]["sampler_seed"] == 456
    assert spec["historical_teachers_included"] is False
    assert spec["existing_campaign_mutations"] is False


def test_one_job_dry_plan_and_explicit_tier3_route(specification, tmp_path, monkeypatch):
    command = study.command_plan(specification)["commands"][0]
    assert command["task_id"] == "fit" and command["dependencies"] == []
    assert "--partition=debug" in command["command"]
    assert "--gres=gpu:a100:1" in command["command"]
    assert "--cpus-per-task=8" in command["command"]
    assert "--mem=73728M" in command["command"]
    assert "--time=808" in command["command"]
    source, evidence = _source(tmp_path)
    monkeypatch.setattr(study, "source_evidence", lambda path: (source, evidence))
    tier3 = study.create(
        source_spec=tmp_path / "source-spec.json", output_root=tmp_path / "tier3",
        project=tmp_path / "tier3-project", source_commit="e" * 40, tier3=True,
    )
    assert "--partition=tier3" in study.command_plan(tier3)["commands"][0]["command"]


def test_dry_submission_never_calls_sbatch(specification, monkeypatch):
    from hlt_classification.jetclass2_delphes import submission

    monkeypatch.setattr(
        submission.subprocess, "run",
        lambda *args, **kwargs: pytest.fail("dry run called sbatch"),
    )
    result = study.submit(specification)
    assert result["dry_run"] and result["jobs"] == {"fit": "DRY_RUN_0000"}
    with pytest.raises(PermissionError):
        study.submit(specification, execute=True, authorization_phrase="wrong")


def test_spec_tamper_and_source_overlap_fail_closed(specification, tmp_path):
    changed = deepcopy(specification)
    changed["kd_weight"] = .8
    changed = with_content_hash(changed)
    with pytest.raises(ValueError, match="semantics"):
        study.validate_spec(changed)
    source, _ = _source(tmp_path)
    with pytest.raises(ValueError, match="overlaps"):
        study._isolated(Path(source["campaign_root"]) / "child", source, tmp_path / "project")


class _Cache:
    def __len__(self):
        return 4

    def batch(self, indexes):
        count = len(indexes)
        return dict(
            features=np.ones((count, 17, 3), np.float32),
            vectors=np.ones((count, 4, 3), np.float32),
            mask=np.ones((count, 1, 3), np.bool_),
            labels=np.arange(count, dtype=np.int64) % 11,
        )


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(17, 11))

    def forward(self, features, vectors, mask):
        del vectors, mask
        return features.mean(-1) @ self.weight


def test_acceptance_executes_single_teacher_exact_c25p75(specification, monkeypatch):
    calls = {}

    def loss(logits, labels, **kwargs):
        calls.update(kwargs)
        return torch.nn.functional.cross_entropy(logits, labels)

    monkeypatch.setattr(study, "DelphesParticleTransformer", _Model)
    monkeypatch.setattr(study, "distillation_loss", loss)
    teacher = np.full((4, 11), 1 / 11, np.float32)
    report = study._accept(specification, _Cache(), teacher, device="cpu")
    assert report["passed"] and report["teacher_count"] == 1
    assert calls["ce_weight"] == .25 and calls["kd_weight"] == .75
    assert calls["temperature"] == 2.
    assert calls["teacher_probabilities"].shape == (4, 11)


def test_cli_help_imports():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "scripts/jetclass2_d000_d033_only.py"), "--help"],
        check=True, capture_output=True, text=True,
    )
    assert "D033-only C25/P75" in result.stdout


def test_reference_rows_include_matched_source_comparators(specification):
    rows = study._reference_rows(specification)
    assert [row["node_id"] for row in rows] == list(study.REFERENCE_IDS)
    assert [row["kind"] for row in rows] == [
        "baseline", "oracle", "teacher", "mt20_endpoint",
    ]
    assert rows[0]["recovery"]["macro_ovr_auc"] == pytest.approx(0.)
    assert rows[1]["recovery"]["macro_ovr_auc"] == pytest.approx(100.)


def test_source_evidence_requires_exact_d033_bank_and_all_comparators(tmp_path, monkeypatch):
    source, _ = _source(tmp_path)
    source.update(
        scientific_plan={
            "split_profile": "TRAIN_500K",
            "role_counts": {"train": 500_000, "validation": 1_000_000,
                            "final_test": 1_000_000},
        },
        ordinary_final_test_capability=False, final_test_accessed=False,
    )
    monkeypatch.setattr(study, "load_json", lambda path: source)
    monkeypatch.setattr(
        study, "validate_campaign_snapshot", lambda value: value["content_hash"],
    )
    reports = {}
    for index, node_id in enumerate(study.REFERENCE_IDS):
        metrics = _metrics(index + 1)
        metrics["rows"] = 1_000_000
        report = artifact(
            "SALIENCE_MT20_TRAINING_REPORT",
            campaign_sha256=source["content_hash"], source_commit=source["source_commit"],
            node={"node_id": node_id}, final_test_accessed=False,
            kernel_training_report={
                "scientific_fit": True, "final_test_accessed": False,
                "validation": metrics, "passes": 60,
                "selected_pass": 20,
            },
        )
        reports[node_id] = report
    monkeypatch.setattr(
        study, "_training_report",
        lambda value, node_id: (
            {"content_hash": str(study.REFERENCE_IDS.index(node_id) + 1) * 64},
            reports[node_id],
        ),
    )
    reducer = {
        "content_hash": "9" * 64,
        "result": {
            "teacher_node": study.TEACHER_ID,
            "teacher_report_sha256": reports[study.TEACHER_ID]["content_hash"],
            "manifest_sha256": "7" * 64,
            "train_bank": "attempts/reduce/train_bank",
        },
    }
    monkeypatch.setattr(study, "completed_task", lambda value, task: reducer)
    actual_source, evidence = study.source_evidence(tmp_path / "source.json")
    assert actual_source is source
    assert evidence["teacher_report_sha256"] == reports[study.TEACHER_ID]["content_hash"]
    assert [row["node_id"] for row in evidence["references"]] == list(study.REFERENCE_IDS)
    broken = deepcopy(reducer)
    broken["result"]["teacher_node"] = study.MT20_ENDPOINT_ID
    monkeypatch.setattr(study, "completed_task", lambda value, task: broken)
    with pytest.raises(ValueError, match="bank lineage"):
        study.source_evidence(tmp_path / "source.json")
