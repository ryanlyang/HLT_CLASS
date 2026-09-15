from fractions import Fraction
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from hlt_classification.data.cache_contracts import canonical_sha256, write_immutable_json
from hlt_classification.jetclass2_delphes.contracts import artifact
from hlt_classification.jetclass2_delphes.salience_mt20_campaign import (
    BRANCH_ORDER, build_campaign_plan, teacher_loss_weights,
)
from hlt_classification.jetclass2_delphes.salience_mt20_probability import (
    mix_probabilities,
)
from hlt_classification.jetclass2_delphes.salience_mt20_production import (
    GATE_TASKS, command_plan, task_graph,
)
from hlt_classification.jetclass2_delphes.execution import execution_site


def _foundation():
    memberships = {
        role: {"content_hash": canonical_sha256({"role": role})}
        for role in ("train", "validation", "final_test")
    }
    return {
        "content_hash": "a" * 64,
        "matcher": {"content_hash": "b" * 64},
        "candidate": "SALIENCE_PT_LINEAR",
        "inputs": {"content_hash": "c" * 64},
        "inventory": {"content_hash": "f" * 64},
        "views": {
            "content_hash": "9" * 64,
            "support": "persistent_hlt_skeleton_offline_tail_v1",
        },
        "splits": {
            "profile": "TRAIN_500K", "content_hash": "d" * 64,
            "registry_sha256": "e" * 64,
            "role_counts": {"train": 500_000, "validation": 1_000_000,
                            "final_test": 1_000_000},
            "memberships": memberships,
        },
    }


def test_teacher_weights_are_exact_nearest_first_loss_contributions():
    assert teacher_loss_weights(1) == (Fraction(4, 5),)
    assert teacher_loss_weights(2) == (Fraction(1, 2), Fraction(3, 10))
    assert teacher_loss_weights(4) == (
        Fraction(1, 2), Fraction(6, 35), Fraction(3, 35), Fraction(3, 70),
    )
    for count in range(1, 16):
        values = teacher_loss_weights(count)
        assert sum(values, Fraction()) == Fraction(4, 5)
        assert all(left > right for left, right in zip(values[1:], values[2:]))
    with pytest.raises(ValueError):
        teacher_loss_weights(0)


def test_three_spines_all_prior_teacher_registries_and_task_census(monkeypatch):
    monkeypatch.setattr(
        "hlt_classification.jetclass2_delphes.salience_mt20_campaign.validate_foundation_spec",
        lambda value: value["content_hash"],
    )
    plan = build_campaign_plan(_foundation())
    assert plan["branch_order"] == list(BRANCH_ORDER)
    assert plan["ultradense_present"] is False
    assert set(plan["branches"]) == {"DIRECT", "COARSE", "DENSE"}
    assert plan["fresh_fit_count"] == 16
    assert plan["probability_publication_count"] == 12
    assert plan["role_counts"] == {
        "train": 500_000, "validation": 1_000_000, "final_test": 1_000_000,
    }
    by_id = {row["node_id"]: row for row in plan["nodes"]}
    for branch, node_ids in plan["branches"].items():
        ancestry = ["U000"]
        for node_id in node_ids:
            node = by_id[node_id]
            assert [row["node_id"] for row in node["teachers"]] == list(reversed(ancestry))
            assert node["teacher"] == ancestry[-1]
            assert sum(
                Fraction(row["loss_weight"]["numerator"], row["loss_weight"]["denominator"])
                for row in node["teachers"]
            ) == Fraction(4, 5)
            ancestry.append(node_id)
    tasks = task_graph(plan)
    assert len(tasks) == 33
    assert tuple(row["task_id"] for row in tasks[:3]) == GATE_TASKS
    assert sum(row["kind"] == "train" for row in tasks) == 16
    assert sum(row["kind"] == "reduce" for row in tasks) == 12
    last = plan["branches"]["DENSE"][-1]
    fit = next(row for row in tasks if row["task_id"] == "train_" + last)
    assert len(fit["dependencies"]) == 8
    assert all(value.startswith("reduce_") for value in fit["dependencies"])


def test_ram_mixture_matches_exact_weighted_sum_and_rejects_bad_inputs():
    rng = np.random.default_rng(17)
    arrays = []
    for _ in range(3):
        raw = rng.random((9, 11), dtype=np.float32)
        arrays.append(np.ascontiguousarray(raw / raw.sum(axis=1, keepdims=True)))
    weights = teacher_loss_weights(3)
    registry = [{"node_id": str(index), "loss_weight": {
        "numerator": weight.numerator, "denominator": weight.denominator,
    }} for index, weight in enumerate(weights)]
    actual = mix_probabilities(arrays, registry)
    expected = sum(
        values.astype(np.float64) * float(weight / Fraction(4, 5))
        for values, weight in zip(arrays, weights, strict=True)
    )
    expected /= expected.sum(axis=1, keepdims=True)
    assert actual.dtype == np.float32 and actual.flags.c_contiguous
    assert np.allclose(actual, expected.astype(np.float32), atol=1e-7, rtol=0)
    broken = arrays[0].copy(); broken[0, 0] = np.nan
    with pytest.raises(ValueError):
        mix_probabilities([broken, *arrays[1:]], registry)


def test_configurable_loss_preserves_old_default_and_exact_c20p80():
    torch = pytest.importorskip("torch")
    from hlt_classification.jetclass2_delphes.model import distillation_loss
    logits = torch.tensor([[.1] * 11, [.2, .1, .0, -.1, .3, .4, .5, .6, .7, .8, .9]])
    labels = torch.tensor([0, 10])
    q = torch.softmax(torch.flip(logits, dims=(1,)) / 2., dim=-1)
    old = distillation_loss(logits, labels, teacher_probabilities=q)
    explicit = distillation_loss(
        logits, labels, teacher_probabilities=q,
        ce_weight=.25, kd_weight=.75, temperature=2.,
    )
    assert torch.equal(old, explicit)
    ce = torch.nn.functional.cross_entropy(logits.float(), labels)
    kd = torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(logits.float() / 2., dim=-1), q,
        reduction="batchmean",
    ) * 4.
    mt20 = distillation_loss(
        logits, labels, teacher_probabilities=q,
        ce_weight=.20, kd_weight=.80, temperature=2.,
    )
    assert torch.allclose(mt20, .20 * ce + .80 * kd)


def test_cli_help_imports():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "scripts/jetclass2_delphes_salience_mt20.py"), "--help"],
        check=True, capture_output=True, text=True,
    )
    assert "salience MT20" in result.stdout


def test_source_lock_reuses_only_selected_linear_foundation(tmp_path, monkeypatch):
    from hlt_classification.jetclass2_delphes import salience_mt20_production as production

    foundation_root = (tmp_path / "linear" / "foundation").resolve()
    screen_root = (tmp_path / "screen").resolve()
    screen = artifact(
        "SALIENCE_SCREEN_SPEC", screen_root=str(screen_root),
        data_root=str((tmp_path / "data").resolve()),
        candidate_registry=["SALIENCE_PT_LINEAR", "LOSING_CANDIDATE"],
        final_test_accessed=False,
    )
    foundation = _foundation()
    selection = artifact(
        "SALIENCE_SELECTION_LOCK", screen_sha256=screen["content_hash"],
        winner="SALIENCE_PT_LINEAR", winner_foundation_root=str(foundation_root),
        winner_foundation_sha256=foundation["content_hash"],
        candidates=["SALIENCE_PT_LINEAR", "LOSING_CANDIDATE"],
        final_test_accessed=False,
    )
    complete = artifact(
        "SALIENCE_SCREEN_COMPLETE", screen_sha256=screen["content_hash"],
        selection_lock_sha256=selection["content_hash"], scientific_fit_count=4,
        final_test_accessed=False,
    )
    profile = artifact(
        "SALIENCE_RUNTIME_PROFILE", screen_sha256=screen["content_hash"],
        passed=True, final_test_accessed=False,
    )
    lock = artifact(
        "SALIENCE_FOUNDATION_LOCK", foundation_sha256=foundation["content_hash"],
        matcher_sha256=foundation["matcher"]["content_hash"],
    )
    values = {
        screen_root / "screen_spec.json": screen,
        screen_root / "selection_lock.json": selection,
        screen_root / "screen_complete.json": complete,
        screen_root / "runtime_profile.json": profile,
        foundation_root / "foundation_spec.json": foundation,
    }
    monkeypatch.setattr(production, "load_json", lambda path: values[Path(path).resolve()])
    monkeypatch.setattr(
        production, "validate_foundation_spec", lambda value: value["content_hash"],
    )
    monkeypatch.setattr(
        production, "authenticate_preparation", lambda value, root: lock,
    )
    producer = {"content_hash": "8" * 64}
    monkeypatch.setattr(production, "assignment_source", lambda: producer)

    source, actual_foundation, actual_profile = production.source_lock(
        foundation_root, screen_root,
    )
    assert actual_foundation == foundation and actual_profile == profile
    assert source["selected_candidate"] == "SALIENCE_PT_LINEAR"
    assert source["foundation_root"] == str(foundation_root)
    assert source["selection_lock_sha256"] == selection["content_hash"]
    assert source["assignment_producer_sha256"] == producer["content_hash"]
    assert source["matching_recomputed"] is False
    assert source["matcher_selection_repeated"] is False


def test_gate_command_plan_is_exact_sporc_and_science_is_separately_guarded(monkeypatch):
    # Use the real first three task declarations without constructing a full
    # source artifact; campaign validation itself is covered by production
    # integration tests after the selected salience screen exists.
    tasks = [
        {"task_id": "authenticate", "kind": "authenticate", "dependencies": [],
         "resource": "metadata"},
        {"task_id": "audit_sources_and_storage", "kind": "storage",
         "dependencies": ["authenticate"], "resource": "metadata"},
        {"task_id": "preflight", "kind": "preflight",
         "dependencies": ["audit_sources_and_storage"], "resource": "preflight"},
    ]
    spec = {
        "content_hash": "1" * 64, "project_dir": "/project",
        "campaign_root": "/campaign", "tasks": tasks,
        "runtime_profile": {"execution_site": execution_site("sporc_a100")},
    }
    monkeypatch.setattr(
        "hlt_classification.jetclass2_delphes.salience_mt20_production.validate_campaign",
        lambda value: value["content_hash"],
    )
    monkeypatch.setattr(
        "hlt_classification.jetclass2_delphes.salience_mt20_production.completed_task",
        lambda spec, task: None,
    )
    result = command_plan(spec, stage="gate")
    assert [row["task_id"] for row in result["commands"]] == list(GATE_TASKS)
    preflight = result["commands"][-1]["command"]
    assert "--partition=debug" in preflight
    assert "--qos=qos_tier3" in preflight
    assert "--account=reu-aisocial" in preflight
    assert "--gres=gpu:a100:1" in preflight
    assert "--cpus-per-task=8" in preflight
    assert "--mem=73728M" in preflight
    assert "--time=360" in preflight
    assert preflight[-1] == "sporc_a100_debug"
    routed = command_plan(spec, stage="gate", tier3_tasks=["preflight"])
    routed_preflight = routed["commands"][-1]
    assert routed_preflight["execution_site"] == "sporc_a100"
    assert "--partition=tier3" in routed_preflight["command"]
    assert routed_preflight["command"][-1] == "sporc_a100"
    assert routed["tier3_tasks"] == ["preflight"]
    with pytest.raises(ValueError, match="tier3 task overrides"):
        command_plan(spec, stage="gate", tier3_tasks=["aggregate"])
    with pytest.raises(PermissionError):
        command_plan(spec, stage="science")


@pytest.mark.parametrize("site_name", ["sporc_a100_debug", "sporc_a100"])
def test_execution_gate_accepts_only_registered_debug_tier3_transfer(
    site_name, monkeypatch,
):
    from hlt_classification.jetclass2_delphes import salience_mt20_production as production

    profile = {
        "execution_site": execution_site("sporc_a100"),
        "cpus": 8, "memory_mb": 73728,
        "gpu": {"name": "A100"},
        "installed_environment": {"python": "pinned"},
    }
    monkeypatch.setenv("JC2_SITE", site_name)
    monkeypatch.setattr(production, "allocation", lambda site: ("123", 8, 73728))
    monkeypatch.setattr(production, "gpu_identity", lambda: {"name": "A100"})
    monkeypatch.setattr(
        production, "installed_environment", lambda: {"python": "pinned"},
    )
    production._execution_gate({"runtime_profile": profile}, "cuda")
    monkeypatch.setenv("JC2_SITE", "tigris_gh200")
    with pytest.raises(PermissionError, match="registered submission site"):
        production._execution_gate({"runtime_profile": profile}, "cuda")


def test_campaign_creation_is_fresh_500k_and_round_trip_validates(tmp_path, monkeypatch):
    from hlt_classification.jetclass2_delphes import salience_mt20_campaign as graph
    from hlt_classification.jetclass2_delphes import salience_mt20_production as production

    foundation = _foundation()
    foundation["inventory"] = {"content_hash": "f" * 64}
    source = artifact(
        "SALIENCE_MT20_SOURCE_LOCK", foundation_root=str(tmp_path / "foundation"),
        split_profile="TRAIN_500K", role_counts=foundation["splits"]["role_counts"],
        final_test_accessed=False,
    )
    profile = artifact("TEST_MT20_PROFILE", workers=8)
    model = artifact("TEST_MT20_MODEL")
    data_root = tmp_path / "data"; data_root.mkdir()
    screen_root = tmp_path / "screen"; screen_root.mkdir()
    write_immutable_json(
        screen_root / "screen_spec.json", {"data_root": str(data_root.resolve())},
    )
    monkeypatch.setattr(graph, "validate_foundation_spec", lambda value: value["content_hash"])
    monkeypatch.setattr(production, "_source", lambda project, commit: None)
    monkeypatch.setattr(production, "verify_snapshot", lambda root, inventory: None)
    monkeypatch.setattr(
        production, "source_lock",
        lambda foundation_root, screen_root: (source, foundation, profile),
    )
    monkeypatch.setattr(production, "model_contract", lambda: model)
    root = tmp_path / "campaign"
    spec = production.create_campaign(
        foundation_root=tmp_path / "foundation", screen_root=screen_root,
        data_root=data_root,
        campaign_root=root, project=tmp_path, source_commit="1" * 40,
    )
    assert spec["fresh_fit_count"] == 16
    assert spec["reducer_count"] == 12
    assert spec["science_task_count"] == 30
    assert spec["task_count"] == 33
    assert spec["internal_ablations"] == ["DIRECT", "COARSE", "DENSE"]
    assert spec["teacher_policy_ablation_included"] is False
    assert spec["submission_routing"]["default_site"] == "sporc_a100_debug"
    assert spec["submission_routing"]["alternate_site"] == "sporc_a100"
    assert spec["scientific_plan"]["split_profile"] == "TRAIN_500K"
    assert spec["ram_only_weighted_teacher_mixtures"] is True
    assert spec["durable_single_model_probability_banks_only"] is True
    assert spec["projected_bank_payload_bytes"] < 1024**3
    assert spec["projected_durable_bytes_upper_bound"] == 4 * 1024**3
    assert spec["rolling_resume"] is False
    assert production.validate_campaign(spec) == spec["content_hash"]
