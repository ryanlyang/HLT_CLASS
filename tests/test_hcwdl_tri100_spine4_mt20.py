from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from hlt_classification.data.cache_contracts import (
    load_json, with_content_hash, write_immutable_json,
)
from hlt_classification.scouting.hcwdl_recovery import build_submission_ledger
from hlt_classification.scouting.hcwdl_tri100_spine4_mt20_graph import (
    ANCHOR_NODE_ID, BRANCH_NODES, BRANCH_ORDER, FIT_ORDER, KD_WEIGHT,
    NODE_REGISTRY, PROBABILITY_COMPONENTS, REDUCER_ORDER,
    SOURCE_DISTRIBUTION, TEACHER_DISTRIBUTIONS, TEACHER_NODES,
    TEACHER_WEIGHTS, mt20_teacher_weights, recipe_payload, validate_graph,
)


def test_mt20_graph_has_four_exact_all_prior_spines() -> None:
    assert validate_graph()
    assert tuple(len(BRANCH_NODES[name]) for name in BRANCH_ORDER) == (1, 5, 8, 15)
    assert len(FIT_ORDER) == 30 and len(REDUCER_ORDER) == 26
    assert PROBABILITY_COMPONENTS[SOURCE_DISTRIBUTION] == (ANCHOR_NODE_ID,)
    assert all(len(value) == 1 for value in PROBABILITY_COMPONENTS.values())
    assert recipe_payload()["loss"] == {
        "kind": "constant_ce_weighted_probability_kd_v1",
        "ce_weight": .20, "kd_weight": .80, "temperature": 2.0,
    }
    for branch in BRANCH_ORDER:
        ancestry = [ANCHOR_NODE_ID]
        for node_id in BRANCH_NODES[branch]:
            assert TEACHER_NODES[node_id] == tuple(reversed(ancestry))
            assert len(TEACHER_DISTRIBUTIONS[node_id]) == len(ancestry)
            assert sum(TEACHER_WEIGHTS[node_id], Fraction()) == KD_WEIGHT
            assert NODE_REGISTRY[node_id].ce_weight == .20
            assert NODE_REGISTRY[node_id].kd_weight == .80
            ancestry.append(node_id)


def test_mt20_teacher_weights_match_frozen_examples() -> None:
    assert mt20_teacher_weights(1) == (Fraction(4, 5),)
    assert mt20_teacher_weights(3) == (
        Fraction(1, 2), Fraction(1, 5), Fraction(1, 10),
    )
    assert mt20_teacher_weights(4) == (
        Fraction(1, 2), Fraction(6, 35), Fraction(3, 35), Fraction(3, 70),
    )
    assert mt20_teacher_weights(5) == (
        Fraction(1, 2), Fraction(4, 25), Fraction(2, 25),
        Fraction(1, 25), Fraction(1, 50),
    )


def test_mt20_preflight_derives_cache_identities_from_stream_keys() -> None:
    from hlt_classification.scouting.hcwdl_representation_data import (
        canonical_identity_digests,
    )
    from hlt_classification.scouting.hcwdl_tri100_spine4_mt20_execution import (
        _batch_identity_digests,
    )

    keys = np.asarray(["source-a::tree::12", "source-b::tree::7"])
    observed = _batch_identity_digests({"identity_keys": keys})
    expected = canonical_identity_digests(tuple(map(str, keys)))
    assert observed.dtype == np.uint8
    assert observed.shape == (2, 32)
    assert np.array_equal(observed, expected)
    with pytest.raises(ValueError, match="identity keys"):
        _batch_identity_digests({})


def test_mt20_preparation_registry_contains_only_timings() -> None:
    from hlt_classification.scouting.hcwdl_tri100_spine4_mt20_runner import (
        _preparation_metrics,
    )

    observed = _preparation_metrics(
        student_view_cache_seconds=12.5,
        pre_training_total_seconds=14.0,
    )
    assert observed == {
        "student_view_cache_seconds": 12.5,
        "pre_training_total_seconds": 14.0,
    }
    assert all(name.endswith("_seconds") for name in observed)


def test_ram_mixture_is_exact_ordered_and_not_published(tmp_path: Path) -> None:
    from hlt_classification.scouting.hcwdl_tri100_spine4_mt20_probability import (
        materialize_ram_mixture, validate_mixture_registry,
    )

    node_id = next(name for name in FIT_ORDER if len(TEACHER_NODES.get(name, ())) == 3)
    identities = np.zeros((3, 32), np.uint8)
    identities[:, 0] = np.arange(3)
    banks = {}
    locks = {}
    expected = np.zeros((3, 15), np.float64)
    for index, (distribution, contribution) in enumerate(zip(
        TEACHER_DISTRIBUTIONS[node_id], TEACHER_WEIGHTS[node_id], strict=True,
    )):
        probabilities = np.full((3, 15), .001, np.float32)
        probabilities[:, index] = .986
        probabilities /= probabilities.sum(1, keepdims=True)
        manifest = with_content_hash({"temperature": 2.0})
        lock = with_content_hash({"distribution_id": distribution})
        banks[distribution] = (manifest, identities, probabilities)
        locks[distribution] = lock
        expected += float(contribution / KD_WEIGHT) * probabilities
    targets, registry = materialize_ram_mixture(
        node_id=node_id, banks=banks, locks=locks,
        parents={"campaign_spec": "a" * 64},
    )
    assert np.allclose(targets.probabilities, expected.astype(np.float32), atol=2e-7)
    assert np.array_equal(targets.join(identities[[2, 0]]), targets.probabilities[[2, 0]])
    assert registry["durable_mixture_path"] is None
    assert validate_mixture_registry(registry, node_id=node_id)
    assert list(tmp_path.iterdir()) == []
    drifted = dict(registry)
    drifted["teacher_banks"] = [dict(row) for row in registry["teacher_banks"]]
    drifted["teacher_banks"][0]["conditional_mixture_weight"] = {
        "numerator": 1, "denominator": 1,
    }
    drifted = with_content_hash({
        key: value for key, value in drifted.items() if key != "content_hash"
    })
    with pytest.raises(ValueError):
        validate_mixture_registry(drifted, node_id=node_id)
    reversed_banks = dict(reversed(tuple(banks.items())))
    with pytest.raises(ValueError, match="bank order"):
        materialize_ram_mixture(
            node_id=node_id, banks=reversed_banks, locks=locks,
            parents={"campaign_spec": "a" * 64},
        )


def _fake_campaign(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from hlt_classification.scouting import hcwdl_tri100_spine4_mt20_campaign as campaign
    from hlt_classification.scouting.hcwdl_mhpe_tri60_ce_control_contracts import (
        TRAINING_REPORT_CONTRACT as CE_CONTRACT,
    )
    from hlt_classification.scouting.hcwdl_tri100_spine4_mt20_contracts import (
        SOURCE_LOCK_CONTRACT, artifact,
    )

    source = artifact({
        "parents": {
            "source_campaign": "1" * 64, "foundation_lock": "2" * 64,
            "foundation_spec": "3" * 64, "assignment_lock": "4" * 64,
            "matcher_spec": "5" * 64,
        },
        "foundation_spec_path": str(tmp_path / "foundation.json"),
        "replicate_seed": 1337,
        "role_counts": {
            "train": 2_777_855, "validation": 957_541, "final_test": 899_779,
        },
    }, contract=SOURCE_LOCK_CONTRACT)
    monkeypatch.setattr(campaign, "build_source_lock", lambda path: source)
    monkeypatch.setattr(campaign, "validate_source_lock", lambda value: value["content_hash"])
    immediate = with_content_hash({
        "campaign_root": str(tmp_path / "immediate"),
        "parents": {"foundation": "2" * 64},
        "support_policy": campaign.PERSISTENT_HLT_SUPPORT_POLICY,
        "role_counts": dict(source["role_counts"]), "replicate_seed": 1337,
        "final_test_accessed": False,
    })
    immediate_path = tmp_path / "immediate.json"
    write_immutable_json(immediate_path, immediate)
    monkeypatch.setattr(
        campaign, "validate_immediate_campaign", lambda value: value["content_hash"],
    )
    baseline = with_content_hash({
        "contract": CE_CONTRACT, "schema_version": 1, "node_id": "M0CE60",
        "validation": {"accuracy": .8, "macro_ovr_auc": .94},
        "final_test_accessed": False,
    })
    baseline_path = tmp_path / "m0.json"
    write_immutable_json(baseline_path, baseline)
    root = tmp_path / "campaign"
    spec = campaign.create_campaign(
        foundation_spec=tmp_path / "foundation.json",
        immediate_campaign_spec=immediate_path,
        m0ce60_report=baseline_path, campaign_root=root, project_dir=tmp_path,
        source_commit="a" * 40, authorize_live_submission=True,
        authorization_phrase=campaign.CREATION_PHRASE,
    )
    return campaign, spec, root


def test_mt20_campaign_is_independent_exact_61_task_dag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign, spec, root = _fake_campaign(tmp_path, monkeypatch)
    assert campaign.validate_campaign(spec, executable=True) == spec["content_hash"]
    assert len(spec["tasks"]) == 61
    assert spec["fresh_fit_count"] == 30 and spec["reducer_count"] == 26
    assert spec["existing_campaign_dependencies"] == []
    assert spec["existing_campaign_outputs_mutated"] is False
    assert spec["ram_only_teacher_mixtures"] is True
    assert spec["durable_teacher_mixture_arrays"] is False
    assert spec["combined_intervention"] == [
        "c20p80", "all_prior_same_spine_teachers",
    ]
    tasks = {row["task_id"]: row for row in spec["tasks"]}
    coarse_d033 = BRANCH_NODES["COARSE"][3]
    assert tasks[f"train_{coarse_d033}"]["dependencies"] == [
        f"reduce_{name}" for name in TEACHER_DISTRIBUTIONS[coarse_d033]
    ]
    plan = load_json(root / "command_plan.json")
    assert len(plan["commands"]) == 61
    assert all(not row["external_dependencies"] for row in plan["commands"])
    fit = next(
        row["command"] for row in plan["commands"]
        if row["task_id"] == f"train_{coarse_d033}"
    )
    assert "--cpus-per-task=72" in fit and "--mem=320G" in fit
    assert "--gres=gpu:gh200:1" in fit
    assert any("HCWDL_MT20_SPEC=" in item for item in fit)


def test_mt20_restart_zero_recovery_closes_over_exact_dag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, spec, root = _fake_campaign(tmp_path, monkeypatch)
    from hlt_classification.scouting import hcwdl_tri100_spine4_mt20_recovery as recovery

    plan = load_json(root / "command_plan.json")
    commands = {row["task_id"]: row["command"] for row in plan["commands"]}
    jobs = {name: str(10000 + index) for index, name in enumerate(commands)}
    ledger = build_submission_ledger(
        campaign_spec_sha256=spec["content_hash"], jobs=jobs,
        commands=commands, dry_run=False,
    )
    ledger_path = tmp_path / "ledger.json"
    write_immutable_json(ledger_path, ledger)
    monitor = recovery.build_monitor(
        spec=spec, ledger=ledger,
        states_by_job_id={job: "FAILED" for job in jobs.values()},
    )
    monitor_path = tmp_path / "monitor.json"
    write_immutable_json(monitor_path, monitor)
    value = recovery.create_recovery(
        subject_spec=root / "campaign_spec.json", subject_ledger=ledger_path,
        monitor_report=monitor_path, recovery_root=tmp_path / "recovery",
        project_dir=tmp_path, source_commit="b" * 40,
    )
    assert recovery.validate_recovery(value) == value["content_hash"]
    assert len(value["retry_tasks"]) == 61
    recovery_plan = load_json(tmp_path / "recovery/command_plan.json")
    assert len(recovery_plan["commands"]) == 61
    assert all(not row["external_dependencies"] for row in recovery_plan["commands"])
