from __future__ import annotations

from pathlib import Path

import numpy as np

from hlt_classification.data.cache_contracts import (
    sha256_file, with_content_hash, write_immutable_json,
)
from hlt_classification.scouting.hcwdl_adjacent_output_handoff_contracts import (
    CONTROL_LOCK_CONTRACT, SOURCE_LOCK_CONTRACT, artifact,
)
from hlt_classification.scouting.hcwdl_adjacent_output_handoff_fusion import (
    centered_log_probabilities, distillation_target, evaluate_mixture_curve,
    mix_probabilities, paired_stratified_macro_auc_bootstrap,
)
from hlt_classification.scouting.hcwdl_adjacent_output_handoff_graph import (
    ENSEMBLE_IDS, FINAL_NODES, FIT_ORDER, GRAPH_SHA256, NODE_REGISTRY,
    SELECTION_IDS, TERMINAL_SEEDS, ensemble_components, selection_components,
    validate_graph,
)
from hlt_classification.scouting.hcwdl_adjacent_output_handoff_partition import (
    load_partition, partition_codes, publish_partition,
)
from hlt_classification.scouting.hcwdl_adjacent_output_handoff_probability import (
    HandoffProbabilityTargets, ROLES, load_probability_role,
    publish_probability_lock, publish_probability_role, validate_probability_lock,
)
from hlt_classification.scouting.evaluation import classification_metrics
from hlt_classification.scouting.hcwdl_adjacent_output_handoff_source import (
    build_control_lock, validate_control_lock,
)
from hlt_classification.scouting.hcwdl_mhpe_tri60_ce_control_contracts import (
    TRAINING_REPORT_CONTRACT as CE60_REPORT_CONTRACT,
)
from hlt_classification.scouting.hcwdl_mhpe_tri60_contracts import (
    TRAINING_REPORT_CONTRACT as TRI60_REPORT_CONTRACT,
)
from hlt_classification.scouting.hcwdl_recovery import (
    build_monitor_report, build_submission_ledger,
)


SHA = "a" * 64
COMMIT = "b" * 40


def _probabilities(seed: int = 3, per_class: int = 6):
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(15), per_class)
    logits = rng.normal(size=(len(labels), 15))
    logits[np.arange(len(labels)), labels] += 2.0
    logits -= logits.max(axis=1, keepdims=True)
    values = np.exp(logits); values /= values.sum(axis=1, keepdims=True)
    return values.astype(np.float32), labels


def _identities(rows: int):
    return np.asarray([
        np.frombuffer(index.to_bytes(32, "little"), dtype=np.uint8)
        for index in range(rows)
    ])


def test_graph_has_exact_26_fit_design_and_paired_seeds():
    assert validate_graph() == GRAPH_SHA256
    assert len(FIT_ORDER) == 26
    assert len(SELECTION_IDS) == 9
    assert len(ENSEMBLE_IDS) == 15
    assert len(FINAL_NODES) == 3
    assert selection_components("OUTPUT_MIX_D080") == (
        "SOURCE_U100", "OUTPUT_DIRECT_D080",
    )
    assert selection_components("OUTPUT_MIX_D000_S5") == (
        "OUTPUT_T_D020", "OUTPUT_DIRECT_D000_S5",
    )
    assert ensemble_components("OUTPUT_COMPRESSION_D000_E5") == tuple(
        f"OUTPUT_COMPRESSION_D000_S{index}" for index in range(1, 6)
    )
    for seed in TERMINAL_SEEDS:
        aliases = {
            NODE_REGISTRY[f"{prefix}_D000_{seed}"].seed_alias
            for prefix in ("OUTPUT_DIRECT", "OUTPUT_COMPRESSION", "CE")
        }
        assert len(aliases) == 1


def test_fusion_formulas_and_temperature_target_are_normalized():
    rich, _ = _probabilities()
    poor = np.roll(rich, 1, axis=1)
    arithmetic = mix_probabilities(
        rich, poor, alpha=.25, family="arithmetic_probability",
    )
    assert np.allclose(arithmetic, .75 * rich + .25 * poor, atol=2e-7)
    assert np.allclose(arithmetic.sum(1), 1)
    calibrated = mix_probabilities(
        rich, poor, alpha=.5, family="calibrated_centered_logit",
        rich_temperature=.8, poor_temperature=1.2,
    )
    assert np.isfinite(centered_log_probabilities(calibrated)).all()
    assert np.allclose(calibrated.sum(1), 1, atol=2e-6)
    target = distillation_target(calibrated, temperature=2.0)
    assert target.dtype == np.float32
    assert np.allclose(target.sum(1), 1, atol=2e-6)


def test_paired_stratified_bootstrap_point_matches_exact_macro_auc_delta():
    rich, labels = _probabilities(per_class=9)
    rng = np.random.default_rng(11)
    poor = rich.astype(np.float64) + rng.normal(0, .002, rich.shape)
    poor = np.maximum(poor, 1e-6); poor /= poor.sum(1, keepdims=True)
    result = paired_stratified_macro_auc_bootstrap(
        poor, rich, labels, samples=40, seed=19,
    )
    expected = (
        classification_metrics(np.log(poor), labels)["macro_ovr_auc"]
        - classification_metrics(np.log(rich), labels)["macro_ovr_auc"]
    )
    assert np.isclose(result["difference"], expected, atol=1e-12)
    assert result["samples"] == 40


def test_curve_registers_both_families_and_alpha_zero_fallback():
    rich, labels = _probabilities(per_class=5)
    curve, selected, chosen = evaluate_mixture_curve(
        rich_probabilities=rich, poor_probabilities=rich,
        labels=labels, rich_id="R", poor_id="P", transition_id="T",
        parents={"x": SHA}, bootstrap_seed=7, bootstrap_samples=8,
    )
    assert len(curve["candidates"]) == 82
    assert {row["family"] for row in curve["candidates"]} == {
        "calibrated_centered_logit", "arithmetic_probability",
    }
    assert selected["selected_alpha_numerator"] == 40
    assert selected["selected_family"] in {
        "calibrated_centered_logit", "arithmetic_probability",
    }
    assert selected["alpha_zero_fallback_registered"] is True
    assert np.allclose(chosen.sum(1), 1, atol=2e-6)


def test_validation_partition_is_deterministic_disjoint_and_stratified(tmp_path: Path):
    identities = _identities(90)
    labels = np.repeat(np.arange(15), 6)
    first = partition_codes(identities, labels)
    second = partition_codes(identities, labels)
    assert np.array_equal(first, second)
    assert set(first.tolist()) == {0, 1, 2}
    for class_id in range(15):
        assert np.bincount(first[labels == class_id], minlength=3).tolist() == [2, 2, 2]
    report = publish_partition(
        tmp_path / "partition.json", identity_digests=identities, labels=labels,
        parents={"campaign": SHA}, source_commit=COMMIT,
    )
    loaded, arrays = load_partition(tmp_path / "partition.json")
    assert loaded == report
    assert np.array_equal(arrays["partition"], first)


def test_probability_bundle_round_trip_is_compact_and_identity_joined(tmp_path: Path):
    assert ROLES == ("train", "V_checkpoint", "V_blend", "V_report")
    probabilities, _ = _probabilities(per_class=3)
    identities = _identities(len(probabilities))
    manifests = {}
    for role in ROLES:
        manifests[role] = publish_probability_role(
            tmp_path, distribution_id="D", role=role,
            identity_digests=identities, probabilities=probabilities,
            component_order=("M",), component_lineage={"M": {"report": SHA}},
            consumers=("C",), parents={"campaign": SHA}, producer_commit=COMMIT,
            target_temperature=2.0 if role == "train" else 1.0,
        )
    lock = publish_probability_lock(
        tmp_path / "lock.json", distribution_id="D", manifests=manifests,
        consumers=("C",), parents={"campaign": SHA},
    )
    checked, _ = validate_probability_lock(tmp_path / "lock.json", distribution_id="D")
    assert checked == lock
    _, loaded_ids, loaded_probabilities = load_probability_role(
        tmp_path / "V_report_manifest.json", distribution_id="D", role="V_report",
    )
    assert np.array_equal(loaded_ids, identities)
    assert np.array_equal(loaded_probabilities, probabilities)
    targets = HandoffProbabilityTargets.load(
        tmp_path / "train_manifest.json", distribution_id="D",
    )
    assert np.allclose(
        targets.join(identities[::-1]),
        distillation_target(probabilities[::-1], temperature=2.0),
        atol=2e-7,
    )
    assert not list(tmp_path.glob("*particle*"))


def test_reporting_control_lock_binds_selected_checkpoints_and_u000_campaign(tmp_path: Path):
    m0_dir = tmp_path / "m0" / "training" / "M0CE60"
    u000_root = tmp_path / "u000"
    u000_dir = u000_root / "training" / "U000"
    m0_dir.mkdir(parents=True); u000_dir.mkdir(parents=True)
    m0_checkpoint = m0_dir / "selected.pt"
    u000_checkpoint = u000_dir / "selected.pt"
    m0_checkpoint.write_bytes(b"m0")
    u000_checkpoint.write_bytes(b"u000")
    campaign = with_content_hash({
        "contract": "TEST_U000_CAMPAIGN/v1", "schema_version": 1,
    })
    write_immutable_json(u000_root / "campaign_spec.json", campaign)
    m0 = with_content_hash({
        "contract": CE60_REPORT_CONTRACT, "schema_version": 1,
        "node_id": "M0CE60", "selected_checkpoint": m0_checkpoint.name,
        "selected_checkpoint_sha256": sha256_file(m0_checkpoint),
        "validation": {"macro_ovr_auc": .5}, "final_test_accessed": False,
    })
    u000 = with_content_hash({
        "contract": TRI60_REPORT_CONTRACT, "schema_version": 1,
        "node_id": "U000", "selected_checkpoint": u000_checkpoint.name,
        "selected_checkpoint_sha256": sha256_file(u000_checkpoint),
        "campaign_spec_sha256": campaign["content_hash"],
        "validation": {"macro_ovr_auc": .6}, "final_test_accessed": False,
    })
    write_immutable_json(m0_dir / "training_report.json", m0)
    write_immutable_json(u000_dir / "training_report.json", u000)
    lock = build_control_lock(
        m0ce60_training_report=m0_dir / "training_report.json",
        pure_offline_u000_training_report=u000_dir / "training_report.json",
    )
    assert lock["m0ce60_checkpoint_path"] == str(m0_checkpoint.resolve())
    assert lock["pure_offline_u000_checkpoint_path"] == str(u000_checkpoint.resolve())
    assert validate_control_lock(lock) == lock["content_hash"]


def test_campaign_dag_is_isolated_and_uses_exact_resources(tmp_path: Path, monkeypatch):
    from hlt_classification.scouting import hcwdl_adjacent_output_handoff_campaign as campaign
    source = artifact({
        "parents": {"foundation": SHA, "assignment_lock": "c" * 64},
        "foundation_spec_path": str(tmp_path / "foundation.json"),
        "support_audit_path": str(tmp_path / "support.json"),
        "replicate_seed": 17, "role_counts": {"train": 30, "validation": 30},
        "final_test_accessed": False,
    }, contract=SOURCE_LOCK_CONTRACT)
    controls = artifact({"parents": {"x": SHA}, "final_test_accessed": False}, contract=CONTROL_LOCK_CONTRACT)
    monkeypatch.setattr(campaign, "build_source_lock", lambda **_: source)
    monkeypatch.setattr(campaign, "build_control_lock", lambda **_: controls)
    monkeypatch.setattr(campaign, "validate_source_lock", lambda value: value["content_hash"])
    monkeypatch.setattr(campaign, "validate_control_lock", lambda value: value["content_hash"])
    root = tmp_path / "campaign"
    spec = campaign.create_campaign(
        source_campaign_spec=tmp_path / "s.json", u100_training_report=tmp_path / "r.json",
        u100_selected_checkpoint=tmp_path / "c.pt", m0ce60_training_report=tmp_path / "m.json",
        pure_offline_u000_training_report=tmp_path / "u.json", campaign_root=root,
        project_dir=tmp_path, source_commit=COMMIT, authorize_live_submission=True,
        authorization_phrase=campaign.CREATION_PHRASE,
    )
    assert spec["fresh_fit_count"] == 26
    assert len(spec["tasks"]) == 85
    assert spec["source_campaign_dependencies"] == []
    assert spec["durable_particle_views"] is False
    assert spec["rolling_resume"] is False
    assert campaign.validate_campaign(spec) == spec["content_hash"]
    plan = {row["task_id"]: row for row in campaign.command_plan(spec)["commands"]}
    assert "--gres=gpu:gh200:1" in plan["train_OUTPUT_DIRECT_D080"]["command"]
    assert plan["train_OUTPUT_DIRECT_D080"]["dependencies"] == ["reduce_SOURCE_U100"]
    assert plan["train_CE_D000_S5"]["dependencies"] == ["preflight"]
    assert all(row["external_dependencies"] == [] for row in plan.values())
    gate = campaign.command_plan(spec, stage="gate")
    science = campaign.command_plan(spec, stage="science")
    assert [row["task_id"] for row in gate["commands"]] == [
        "authenticate", "partition_validation", "audit_sources_and_storage", "preflight",
    ]
    assert science["commands"][0]["task_id"] == "reduce_SOURCE_U100"
    assert science["commands"][0]["dependencies"] == []

    from hlt_classification.scouting.hcwdl_adjacent_output_handoff_recovery import (
        create_recovery,
    )
    commands = {
        row["task_id"]: row["command"] for row in science["commands"]
    }
    jobs = {
        task_id: str(95000 + index) for index, task_id in enumerate(commands)
    }
    ledger = build_submission_ledger(
        campaign_spec_sha256=spec["content_hash"], jobs=jobs,
        commands=commands, dry_run=False,
    )
    ledger_path = tmp_path / "science_ledger.json"
    write_immutable_json(ledger_path, ledger)
    monitor = build_monitor_report(
        ledger, states_by_job_id={job: "FAILED" for job in jobs.values()},
        artifact_validity={task_id: False for task_id in jobs},
    )
    monitor_path = tmp_path / "science_monitor.json"
    write_immutable_json(monitor_path, monitor)
    recovery_root = tmp_path / "recovery"
    recovery = create_recovery(
        campaign_spec=root / "campaign_spec.json", submission_ledger=ledger_path,
        monitor_report=monitor_path, recovery_root=recovery_root,
        project_dir=tmp_path, source_commit=COMMIT,
    )
    assert set(recovery["retry_tasks"]) == set(jobs)
    recovery_plan = __import__("json").loads(
        (recovery_root / "command_plan.json").read_text(),
    )
    assert recovery_plan["commands"][0]["task_id"] == "reduce_SOURCE_U100"
    assert recovery_plan["commands"][0]["dependencies"] == []
