from __future__ import annotations

import copy
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch
import torch.nn.functional as functional

from hlt_classification.data.cache_contracts import (
    canonical_sha256, with_content_hash, write_immutable_json,
)
from hlt_classification.provenance import (
    SOURCE_SNAPSHOT_CONTRACT,
    SOURCE_SNAPSHOT_SCHEMA_VERSION,
)
from hlt_classification.scouting.engine import (
    PmardTrainingConfig,
    PmardTrainingInterrupted,
    train_pmard,
)
from hlt_classification.scouting.hcwdl_parent_loss import (
    HCWDL_PARENT_BASE_LOSS_CONTRACT,
    HCWDL_PARENT_LOSS_POLICY_SHA256,
    HCWDL_PARENT_LOSS_SEMANTICS,
    build_parent_loss_attestation,
    build_parent_loss_attestation_from_reports,
    hcwdl_base_loss,
    parent_loss_runtime_fingerprint,
    validate_parent_loss_attestation,
)
from hlt_classification.scouting.hcwdl_ladder import NODE_REGISTRY
from hlt_classification.scouting.hcwdl_recipe import (
    CLASS_WEIGHT_POLICY,
    LEGACY_CLASS_WEIGHT_POLICY,
    LEGACY_RECIPE_CONTRACT,
    PRIMARY_RECIPE_PROFILE,
    RECIPE_CONTRACT,
    build_recipe,
    example_recipe,
)
from hlt_classification.scouting.hcwdl_training import (
    train_hcwdl_node,
    validate_hcwdl_training_report,
)
from hlt_classification.scouting.inputs import ParticleInputs
from hlt_classification.scouting.training import LossConfiguration, pmard_loss


def _configuration() -> LossConfiguration:
    return LossConfiguration.for_mixture(
        arm="HCWDL_TEST", ce=0.25, hlt_kd=0.40, privileged_kd=0.35,
        hlt_temperature=1.0, privileged_temperature=2.0,
    )


def _authorized_recipe() -> dict[str, object]:
    raw = example_recipe()
    payload = {
        key: value for key, value in raw.items()
        if key not in {
            "contract", "schema_version", "authorized_for_execution", "content_hash",
        }
    }
    payload["recipe_profile"] = PRIMARY_RECIPE_PROFILE
    payload["purpose"] = "hcwdl_primary_ladder"
    return build_recipe(payload, authorized=True)


def _alternate_authorized_recipe() -> dict[str, object]:
    raw = _authorized_recipe()
    payload = {
        key: copy.deepcopy(value) for key, value in raw.items()
        if key not in {
            "contract", "schema_version", "authorized_for_execution", "content_hash",
        }
    }
    payload["evidence"] = {"alternate_fixture": "9" * 64}
    return build_recipe(payload, authorized=True)


def _recipe_authority(recipe: dict[str, object]) -> dict[str, object]:
    return {
        "parent_recipe_contract": RECIPE_CONTRACT,
        "parent_recipe_sha256": recipe["content_hash"],
        "parent_recipe_profile": PRIMARY_RECIPE_PROFILE,
        "parent_class_weight_policy": CLASS_WEIGHT_POLICY,
    }


def _source_authority() -> dict[str, object]:
    registry = [
        {
            "logical_name": logical_name,
            "relative_path": relative_path,
            "sha256": f"{index:x}" * 64,
        }
        for index, (logical_name, relative_path) in enumerate((
            ("engine", "src/hlt_classification/scouting/engine.py"),
            ("parent_loss", "src/hlt_classification/scouting/hcwdl_parent_loss.py"),
            ("training", "src/hlt_classification/scouting/hcwdl_training.py"),
        ), start=1)
    ]
    commit = "4" * 40
    tree = "5" * 40
    tracked = "6" * 64
    snapshot = with_content_hash({
        "contract": SOURCE_SNAPSHOT_CONTRACT,
        "schema_version": SOURCE_SNAPSHOT_SCHEMA_VERSION,
        "git_commit": commit,
        "git_tree": tree,
        "tracked_files_sha256": tracked,
        "tracked_file_count": 3,
        "worktree_clean": True,
        "source_snapshot_sha256": canonical_sha256({
            "git_commit": commit,
            "git_tree": tree,
            "tracked_files_sha256": tracked,
        }),
    })
    return {
        "parent_campaign_spec_sha256": "7" * 64,
        "parent_campaign_authority": {
            "parent_campaign_contract": "HCWDL_CAMPAIGN_SPEC/v8",
            "parent_campaign_mode": "pilot",
            "parent_execution_scope": "parent_prefix_through_finalist_lock",
            "parent_endpoint_continuation": "preauthorized_automatic",
            "parent_training_passes": 60,
            "parent_validation_every_passes": 1,
            "parent_train_rows": 300000,
            "parent_terminal_task_id": "finalist_lock",
            "parent_execution_lock_authorized": False,
            "parent_final_test_access_authorized": False,
            "parent_registered_final_test_tasks": 0,
        },
        "parent_source_snapshot": snapshot,
        "runtime_source_registry": registry,
    }


def _runtime_source_sha256(source_authority: dict[str, object]) -> str:
    return canonical_sha256(source_authority["runtime_source_registry"])


def _parent_rows(
    recipe: dict[str, object], source_authority: dict[str, object],
) -> list[dict[str, object]]:
    batch = int(recipe["batching"]["effective_batch_size"])
    updates = 60 * int(np.ceil(300000 / batch))
    return [
        {
            "node_id": node_id,
            "training_report_sha256": "1" * 64,
            "checkpoint_sha256": "2" * 64,
            "final_checkpoint_sha256": "3" * 64,
            "execution_config_sha256": "4" * 64,
            "completed_updates": updates,
            "validation_record_count": 60,
            "loss_semantics_contract": HCWDL_PARENT_BASE_LOSS_CONTRACT,
            "producer_source_sha256": _runtime_source_sha256(source_authority),
            **_recipe_authority(recipe),
        }
        for node_id in sorted(NODE_REGISTRY)
    ]


def test_hcwdl_base_loss_weights_ce_but_not_either_kd_and_stays_fp32():
    student = torch.linspace(-1, 1, 60).reshape(4, 15).bfloat16().requires_grad_()
    labels = torch.tensor([0, 1, 2, 3])
    weights = torch.linspace(0.2, 3.0, 15)
    hlt = torch.linspace(1.0, -1.0, 60).reshape(4, 15)
    privileged = torch.cos(torch.arange(60, dtype=torch.float32)).reshape(4, 15)
    configuration = _configuration()
    result = hcwdl_base_loss(
        student, labels, class_weights=weights, configuration=configuration,
        hlt_teacher_logits=hlt, privileged_teacher_logits=privileged,
    )
    student_fp32 = student.float()
    ce_rows = functional.cross_entropy(student_fp32, labels, reduction="none")

    def kd_rows(teacher, tau):
        return functional.kl_div(
            functional.log_softmax(student_fp32 / tau, dim=-1),
            functional.softmax(teacher / tau, dim=-1), reduction="none",
        ).sum(-1) * tau * tau

    assert torch.equal(result["ce"], (weights[labels] * ce_rows).mean())
    assert torch.equal(result["hlt_kd"], kd_rows(hlt, 1.0).mean())
    assert torch.equal(result["privileged_kd"], kd_rows(privileged, 2.0).mean())
    assert all(value.dtype == torch.float32 for value in result.values())
    result["total"].backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()


def test_legacy_pmard_default_semantics_remain_class_weighted_and_distinct():
    torch.manual_seed(18)
    student = torch.randn(5, 15, requires_grad=True)
    labels = torch.tensor([0, 3, 4, 7, 14])
    weights = torch.linspace(0.25, 2.75, 15)
    hlt = torch.randn(5, 15)
    privileged = torch.randn(5, 15)
    configuration = _configuration()
    legacy = pmard_loss(
        student, labels, class_weights=weights, configuration=configuration,
        hlt_teacher_logits=hlt, privileged_teacher_logits=privileged,
    )
    corrected = hcwdl_base_loss(
        student, labels, class_weights=weights, configuration=configuration,
        hlt_teacher_logits=hlt, privileged_teacher_logits=privileged,
    )
    assert not torch.equal(legacy["hlt_kd"], corrected["hlt_kd"])
    assert not torch.equal(legacy["privileged_kd"], corrected["privileged_kd"])


def test_legacy_pmard_value_and_gradient_remain_exactly_the_historical_formula():
    torch.manual_seed(1818)
    logits = torch.randn(7, 15, dtype=torch.float32, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 6, 9, 11, 14])
    weights = torch.linspace(0.3, 2.8, 15)
    hlt = torch.randn(7, 15)
    privileged = torch.randn(7, 15)
    configuration = _configuration()

    observed = pmard_loss(
        logits, labels, class_weights=weights, configuration=configuration,
        hlt_teacher_logits=hlt, privileged_teacher_logits=privileged,
    )["total"]
    observed_gradient, = torch.autograd.grad(observed, logits, retain_graph=True)

    row_weight = weights[labels]
    ce = (
        row_weight
        * functional.cross_entropy(logits.float(), labels, reduction="none")
    ).mean()

    def historical_kd(teacher: torch.Tensor, tau: float) -> torch.Tensor:
        rows = functional.kl_div(
            functional.log_softmax(logits.float() / tau, dim=-1),
            functional.softmax(teacher.float() / tau, dim=-1),
            reduction="none",
        ).sum(-1) * tau * tau
        return (row_weight * rows).mean()

    expected = (
        0.25 * ce
        + 0.40 * historical_kd(hlt, 1.0)
        + 0.35 * historical_kd(privileged, 2.0)
    )
    expected_gradient, = torch.autograd.grad(expected, logits)
    assert torch.equal(observed, expected)
    assert torch.equal(observed_gradient, expected_gradient)


def test_parent_loss_attestation_rejects_old_semantics_and_tampering():
    recipe = _authorized_recipe()
    source_authority = _source_authority()
    rows = _parent_rows(recipe, source_authority)
    row = rows[0]
    report = build_parent_loss_attestation(
        parent_recipe=recipe,
        parent_artifacts=rows, **source_authority,
    )
    assert validate_parent_loss_attestation(
        report, parent_recipe=recipe,
    ) == report["content_hash"]
    assert parent_loss_runtime_fingerprint() == report["runtime_fingerprint"]

    old_rows = copy.deepcopy(rows)
    old = old_rows[0]
    old["loss_semantics_contract"] = "HCWDL_PARENT_BASE_LOSS/legacy_weighted_kd"
    with pytest.raises(ValueError, match="incompatible loss semantics"):
        build_parent_loss_attestation(
            parent_recipe=recipe,
            parent_artifacts=old_rows, **source_authority,
        )

    mixed_rows = copy.deepcopy(rows)
    mixed_source = mixed_rows[0]
    mixed_source["producer_source_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="producer source differs"):
        build_parent_loss_attestation(
            parent_recipe=recipe,
            parent_artifacts=mixed_rows, **source_authority,
        )

    tampered = copy.deepcopy(report)
    tampered["runtime_fingerprint"]["student_gradient_sha256"] = "f" * 64
    tampered = with_content_hash({
        key: value for key, value in tampered.items() if key != "content_hash"
    })
    with pytest.raises(ValueError, match="runtime fingerprint"):
        validate_parent_loss_attestation(tampered, parent_recipe=recipe)


def test_parent_loss_attestation_rejects_v3_non_one_and_mixed_recipe_authority():
    current = _authorized_recipe()
    source_authority = _source_authority()
    legacy = copy.deepcopy(current)
    legacy["contract"] = LEGACY_RECIPE_CONTRACT
    legacy["schema_version"] = 3
    counts = np.arange(1, 16, dtype=np.float64)
    inverse = 1.0 / np.sqrt(counts)
    weights = (counts.sum() / np.sum(counts * inverse) * inverse).astype(np.float32)
    legacy["class_weighting"] = {
        **legacy["class_weighting"],
        "policy": LEGACY_CLASS_WEIGHT_POLICY,
        "train_class_counts": [int(value) for value in counts],
    }
    legacy["class_weights"] = weights.tolist()
    legacy = with_content_hash({
        key: value for key, value in legacy.items() if key != "content_hash"
    })
    with pytest.raises(ValueError, match="HCWDL_RECIPE/v4"):
        build_parent_loss_attestation(
            parent_recipe=legacy,
            parent_artifacts=[],
            **source_authority,
        )

    non_one = copy.deepcopy(current)
    non_one["class_weights"][0] = 2.0
    non_one = with_content_hash({
        key: value for key, value in non_one.items() if key != "content_hash"
    })
    with pytest.raises(ValueError, match="class weights differ"):
        build_parent_loss_attestation(
            parent_recipe=non_one,
            parent_artifacts=[],
            **source_authority,
        )

    rows = _parent_rows(current, source_authority)
    with pytest.raises(ValueError, match="different parent recipe authority"):
        build_parent_loss_attestation(
            parent_recipe=_alternate_authorized_recipe(),
            parent_artifacts=rows,
            **source_authority,
        )


class _TinyParentModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(21, 15)

    def forward(self, features, vectors, mask):
        del vectors
        valid = mask.float()
        pooled = (features * valid).sum(-1) / valid.sum(-1).clamp_min(1)
        return self.linear(pooled)


def _engine_batch() -> dict[str, object]:
    features = np.linspace(-0.8, 1.2, 4 * 21 * 2, dtype=np.float32).reshape(4, 21, 2)
    vectors = np.zeros((4, 4, 2), np.float32)
    vectors[:, 3] = 1.0
    mask = np.ones((4, 1, 2), np.bool_)
    return {
        "hlt": ParticleInputs(features, vectors, mask, np.full(4, 2, np.int32)),
        "labels": np.asarray((0, 2, 7, 14), dtype=np.int64),
        "identity_keys": np.asarray([f"parent::{index}" for index in range(4)]),
    }


def _engine_config(*, updates: int = 1) -> PmardTrainingConfig:
    return PmardTrainingConfig(
        experiment_id="parent_loss_engine",
        loss=LossConfiguration.for_mixture(
            arm="HCWDL_PARENT_ENGINE_TEST", ce=0.25, hlt_kd=0.75,
            privileged_kd=0.0, hlt_temperature=1.0,
            privileged_temperature=2.0,
        ),
        total_updates=updates,
        effective_batch_size=4,
        peak_learning_rate=1e-3,
        validation_interval=1,
        logging_interval=1,
        amp_dtype="none",
    )


def test_train_pmard_requires_explicit_corrected_semantics_and_preserves_legacy_default(
    tmp_path: Path,
):
    torch.manual_seed(702)
    template = _TinyParentModel()
    initial = copy.deepcopy(template.state_dict())
    teacher = _TinyParentModel()
    batch = _engine_batch()
    weights = torch.linspace(0.2, 3.0, 15)
    labels = torch.as_tensor(batch["labels"])
    with torch.no_grad():
        features = torch.as_tensor(batch["hlt"].features)
        vectors = torch.as_tensor(batch["hlt"].vectors)
        mask = torch.as_tensor(batch["hlt"].mask)
        expected_student = template(features, vectors, mask)
        expected_teacher = teacher(features, vectors, mask)
    expected_legacy = pmard_loss(
        expected_student, labels, class_weights=weights,
        configuration=_engine_config().loss,
        hlt_teacher_logits=expected_teacher,
    )
    expected_corrected = hcwdl_base_loss(
        expected_student, labels, class_weights=weights,
        configuration=_engine_config().loss,
        hlt_teacher_logits=expected_teacher,
    )
    assert not torch.equal(expected_legacy["hlt_kd"], expected_corrected["hlt_kd"])

    legacy_model = _TinyParentModel()
    legacy_model.load_state_dict(initial)
    legacy = train_pmard(
        model=legacy_model, train_batches=lambda _epoch: iter((batch,)),
        validation_batches=lambda: iter((batch,)), class_weights=weights,
        config=_engine_config(), output_dir=tmp_path / "legacy",
        parents={"source": "a" * 64}, device="cpu", hlt_teacher=teacher,
    )
    assert legacy["training_history"][0]["mean_losses"]["hlt_kd"] == pytest.approx(
        float(expected_legacy["hlt_kd"]), rel=0, abs=1e-7,
    )
    assert "loss_semantics_contract" not in legacy

    corrected_model = _TinyParentModel()
    corrected_model.load_state_dict(initial)
    corrected = train_pmard(
        model=corrected_model, train_batches=lambda _epoch: iter((batch,)),
        validation_batches=lambda: iter((batch,)), class_weights=weights,
        config=_engine_config(), output_dir=tmp_path / "corrected",
        parents={"source": "a" * 64}, device="cpu", hlt_teacher=teacher,
        loss_semantics_contract=HCWDL_PARENT_BASE_LOSS_CONTRACT,
    )
    assert corrected["training_history"][0]["mean_losses"]["hlt_kd"] == pytest.approx(
        float(expected_corrected["hlt_kd"]), rel=0, abs=1e-7,
    )
    assert corrected["loss_semantics_contract"] == HCWDL_PARENT_BASE_LOSS_CONTRACT
    assert corrected["loss_semantics"] == HCWDL_PARENT_LOSS_SEMANTICS
    assert len(corrected["execution_config_sha256"]) == 64
    checkpoint = torch.load(
        tmp_path / "corrected" / corrected["selected_checkpoint"],
        map_location="cpu", weights_only=False,
    )
    assert checkpoint["loss_semantics_contract"] == HCWDL_PARENT_BASE_LOSS_CONTRACT
    assert checkpoint["loss_semantics"] == HCWDL_PARENT_LOSS_SEMANTICS
    assert checkpoint["execution_config_sha256"] == corrected["execution_config_sha256"]
    assert any(
        not torch.equal(legacy_model.state_dict()[name], corrected_model.state_dict()[name])
        for name in legacy_model.state_dict()
    )

    with pytest.raises(ValueError, match="require explicit engine authorization"):
        train_pmard(
            model=_TinyParentModel(), train_batches=lambda _epoch: iter((batch,)),
            validation_batches=lambda: iter((batch,)), class_weights=weights,
            config=_engine_config(), output_dir=tmp_path / "spoof",
            parents={"source": "a" * 64}, device="cpu", hlt_teacher=teacher,
            scientific_config={
                "loss_semantics_contract": HCWDL_PARENT_BASE_LOSS_CONTRACT,
            },
        )


def test_parent_attestation_rejects_v7_before_reusing_report_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    from hlt_classification.scouting import hcwdl_campaign

    monkeypatch.setattr(
        hcwdl_campaign, "validate_campaign_spec",
        lambda value, *, executable: value["content_hash"],
    )
    recipe = _authorized_recipe()
    recipe_path = tmp_path / "campaign_recipe.json"
    write_immutable_json(recipe_path, recipe)
    root = tmp_path / "M0"
    engine = train_pmard(
        model=_TinyParentModel(),
        train_batches=lambda _epoch: iter((_engine_batch(),)),
        validation_batches=lambda: iter((_engine_batch(),)),
        class_weights=torch.linspace(0.2, 3.0, 15),
        config=_engine_config(), output_dir=root,
        parents={"source": "a" * 64, "recipe": recipe["content_hash"]}, device="cpu",
        hlt_teacher=_TinyParentModel(),
        scientific_config={"recipe_sha256": recipe["content_hash"]},
        loss_semantics_contract=HCWDL_PARENT_BASE_LOSS_CONTRACT,
    )
    semantics = dict(HCWDL_PARENT_LOSS_SEMANTICS)
    wrapper = with_content_hash({
        "contract": "HCWDL_TRAINING_REPORT/v1", "schema_version": 1,
        "node_id": "M0", "complete": True,
        "recipe_sha256": recipe["content_hash"],
        "parents": {"source": "a" * 64, "recipe": recipe["content_hash"]},
        "pmard_engine_report_sha256": engine["content_hash"],
        "pmard_execution_config_sha256": engine["execution_config_sha256"],
        "selected_checkpoint_sha256": engine["selected_checkpoint_sha256"],
        "loss_semantics_contract": HCWDL_PARENT_BASE_LOSS_CONTRACT,
        "loss_semantics": semantics,
        "loss_semantics_sha256": canonical_sha256(semantics),
    })
    report_path = root / "hcwdl_training_report.json"
    write_immutable_json(report_path, wrapper)
    project_root = tmp_path / "parent-project"
    source_root = project_root / "src" / "hlt_classification" / "scouting"
    source_root.mkdir(parents=True)
    runtime_sources = {}
    for logical_name, file_name in {
        "engine": "engine.py", "parent_loss": "hcwdl_parent_loss.py",
        "training": "hcwdl_training.py",
    }.items():
        source = source_root / file_name
        source.write_text(f"SOURCE = {logical_name!r}\n", encoding="utf-8")
        runtime_sources[logical_name] = source
    subprocess.run(["git", "init"], cwd=project_root, check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=project_root, check=True)
    subprocess.run(
        [
            "git", "-c", "user.name=HCWDL Test", "-c",
            "user.email=hcwdl@example.invalid", "commit", "-m", "fixture",
        ],
        cwd=project_root, check=True, capture_output=True,
    )
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=project_root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    campaign = with_content_hash({
        "contract": "HCWDL_CAMPAIGN_SPEC/v7", "schema_version": 7,
        "source_commit": source_commit,
    })
    campaign_path = tmp_path / "parent_campaign_spec.json"
    write_immutable_json(campaign_path, campaign)
    with pytest.raises(ValueError, match="HCWDL_CAMPAIGN_SPEC/v8"):
        build_parent_loss_attestation_from_reports(
            parent_recipe_path=recipe_path,
            parent_campaign_spec_path=campaign_path,
            parent_reports={"M0": report_path}, runtime_source_paths=runtime_sources,
        )

    with pytest.raises(ValueError, match="file registry"):
        build_parent_loss_attestation_from_reports(
            parent_recipe_path=recipe_path,
            parent_campaign_spec_path=campaign_path,
            parent_reports={"M0": report_path},
            runtime_source_paths={"engine": runtime_sources["engine"]},
        )


def test_corrected_loss_semantics_are_bound_into_resume_lineage(tmp_path: Path):
    batch = _engine_batch()
    teacher = _TinyParentModel()
    common = dict(
        train_batches=lambda _epoch: iter((batch,)),
        validation_batches=lambda: iter((batch,)),
        class_weights=torch.linspace(0.2, 3.0, 15),
        config=_engine_config(updates=2), output_dir=tmp_path,
        parents={"source": "b" * 64}, device="cpu", hlt_teacher=teacher,
        scientific_config={"purpose": "resume-lineage-test"},
    )
    with pytest.raises(PmardTrainingInterrupted):
        train_pmard(
            model=_TinyParentModel(), stop_after_update=1,
            loss_semantics_contract=HCWDL_PARENT_BASE_LOSS_CONTRACT, **common,
        )
    with pytest.raises(ValueError, match="resume checkpoint lineage differs"):
        train_pmard(model=_TinyParentModel(), **common)
    resumed = train_pmard(
        model=_TinyParentModel(),
        loss_semantics_contract=HCWDL_PARENT_BASE_LOSS_CONTRACT, **common,
    )
    assert resumed["updates"] == 2


def test_parent_hcwdl_reruns_route_to_corrected_engine_and_old_reports_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}

    def fake_train_pmard(**kwargs):
        captured.update(kwargs)
        return {
            "content_hash": "1" * 64,
            "execution_config_sha256": "2" * 64,
            "selected_checkpoint_sha256": "3" * 64,
            "final_checkpoint_sha256": "4" * 64,
            "selected_update": 2,
            "validation_history": [{
                "update": 2, "macro_ovr_auc": 0.91, "cross_entropy": 0.62,
                "macro_mean_log_qcd_rejection_at_50pct_signal": 7.1,
            }],
        }

    monkeypatch.setattr(
        "hlt_classification.scouting.hcwdl_training.train_pmard",
        fake_train_pmard,
    )
    report = train_hcwdl_node(
        node_id="M0", recipe=_authorized_recipe(), train_rows=1,
        replicate_seed=19, model_factory=_TinyParentModel,
        train_batches=lambda _epoch: iter(()), validation_batches=lambda: iter(()),
        class_weights=np.ones(15, np.float32), output_dir=tmp_path,
        parents={"split_manifest_sha256": "5" * 64}, device="cpu", smoke=True,
    )
    assert captured["loss_semantics_contract"] == HCWDL_PARENT_BASE_LOSS_CONTRACT
    assert captured["scientific_config"]["loss_semantics"] == HCWDL_PARENT_LOSS_SEMANTICS
    assert validate_hcwdl_training_report(report) == report["content_hash"]

    old = {
        key: value for key, value in report.items()
        if key not in {
            "content_hash", "loss_semantics_contract", "loss_semantics",
            "loss_semantics_sha256",
        }
    }
    old = with_content_hash(old)
    with pytest.raises(ValueError, match="loss semantics differ"):
        validate_hcwdl_training_report(old)
