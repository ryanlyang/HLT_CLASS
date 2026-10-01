from fractions import Fraction
import hashlib
import json

import numpy as np
import pytest

from hlt_classification.cms2jc2_response.bridge import Particles
from hlt_classification.cms_proxy_ladder.campaign import (
    BRANCHES, build_scientific_plan, coordinate, task_graph,
)
from hlt_classification.cms_proxy_ladder.contracts import artifact
from hlt_classification.cms_proxy_ladder.gate import debug_gate_tasks, gate_tasks
from hlt_classification.cms_proxy_ladder.inputs import build_inputs, input_contract
from hlt_classification.cms_proxy_ladder.release import COUNTS, release_request
from hlt_classification.cms_proxy_ladder.views import build_view, view_contract
from hlt_classification.jetclass2_delphes.execution import execution_site


def particles(p4, category, charge, *, tracking=None, valid=None, prefix="p"):
    p4 = np.asarray(p4, np.float64)
    count = len(p4)
    return Particles(
        p4,
        np.asarray(charge, np.int8),
        np.asarray(category, np.int8),
        np.zeros((count, 4), np.float64) if tracking is None else np.asarray(tracking, np.float64),
        np.zeros((count, 4), bool) if valid is None else np.asarray(valid, bool),
        tuple(f"{prefix}:{index}" for index in range(count)),
    )


def endpoints():
    proxy = particles(
        [[10, 0, 1, 10.1], [0, 8, -1, 8.1], [3, 4, 0, 5.1]],
        [0, 2, 5], [1, 0, 0], prefix="h",
    )
    offline = particles(
        [[11, 0, 1, 11.1], [0, 9, -1, 9.1], [4, 3, 0, 5.1], [1, 1, 0, 1.5]],
        [0, 2, 1, 4], [1, 0, 0, -1], prefix="o",
    )
    return proxy, offline


def test_registered_graph_is_exact_three_spine_17_fit_12_reducer():
    foundation = artifact("FOUNDATION_TEST", marker="synthetic")
    plan = build_scientific_plan(foundation)
    assert BRANCHES == {
        "DIRECT": ("D000",),
        "COARSE": ("U050", "U100", "D066", "D033", "D000"),
        "DENSE": ("U033", "U066", "U100", "D080", "D060", "D040", "D020", "D000"),
    }
    assert plan["fresh_fit_count"] == 17
    assert plan["probability_publication_count"] == 12
    assert plan["controls"] == ["M0HLT", "OFFLINE", "U000"]
    assert len(task_graph(plan)) == 31
    assert coordinate("U033") == (Fraction(1, 3), Fraction(0))
    assert coordinate("D066") == (Fraction(1), Fraction(1, 3))
    with pytest.raises(ValueError):
        coordinate("D0500")


def test_persistent_proxy_endpoints_and_tail_cardinality():
    proxy, offline = endpoints()
    identity = "12" * 32
    mapping = np.asarray([0, 1, 2], np.int32)
    u000 = build_view(
        identity=identity, proxy=proxy, offline=offline,
        coordinate="U000", mapping=mapping,
    )
    u100 = build_view(
        identity=identity, proxy=proxy, offline=offline,
        coordinate="U100", mapping=mapping,
    )
    d050 = build_view(
        identity=identity, proxy=proxy, offline=offline,
        coordinate="D050", mapping=mapping,
    )
    d066 = build_view(
        identity=identity, proxy=proxy, offline=offline,
        coordinate="D066", mapping=mapping,
    )
    d000 = build_view(
        identity=identity, proxy=proxy, offline=None,
        coordinate="D000", mapping=None,
    )
    pure = build_view(
        identity=identity, proxy=proxy, offline=offline,
        coordinate="OFFLINE", mapping=mapping,
    )
    assert len(u000) == 4
    assert len(u100) == len(proxy) == 3
    np.testing.assert_allclose(u100.p4, offline.p4[:3])
    np.testing.assert_allclose(d050.p4, .5 * (proxy.p4 + offline.p4[:3]))
    np.testing.assert_allclose(d066.p4, proxy.p4 / 3 + 2 * offline.p4[:3] / 3)
    np.testing.assert_array_equal(d000.p4, proxy.p4)
    np.testing.assert_array_equal(pure.p4, offline.p4)


def test_unmatched_proxy_is_retained_at_u000_and_u100():
    proxy, offline = endpoints()
    offline = offline.take([0, 1])
    mapping = np.asarray([0, 1, -1], np.int32)
    for coordinate_name in ("U000", "U100"):
        view = build_view(
            identity="34" * 32, proxy=proxy, offline=offline,
            coordinate=coordinate_name, mapping=mapping,
        )
        assert len(view) == len(proxy)
        np.testing.assert_array_equal(view.p4[2], proxy.p4[2])


def test_validity_aware_adapter_has_unknown_zero_one_hot_and_no_truncation():
    value = particles(
        [[5, 0, 0, 5.1], [0, 4, 0, 4.1]], [5, 0], [0, 1],
        tracking=[[0, 0, 0, 0], [.2, -.3, .05, .08]],
        valid=[[False] * 4, [True] * 4],
    )
    result = build_inputs(value, capacity=16)
    assert result.features.shape == (2, 17)
    assert result.vectors.shape == (2, 4)
    np.testing.assert_array_equal(result.features[0, 6:11], np.zeros(5))
    assert result.features[0, 11:15].tolist() == [0., 0., 0., 0.]
    oversized = particles(
        [[5 + index, 0, 0, 5.1 + index] for index in range(17)],
        [1] * 17, [0] * 17,
    )
    with pytest.raises(ValueError, match="truncation forbidden"):
        build_inputs(oversized, capacity=16)


def test_release_request_is_ordinary_role_only_and_gate_is_staged(tmp_path):
    request = release_request(study_root=tmp_path / "proxy", offline_root=tmp_path / "offline")
    assert request["counts"] == COUNTS == {"train": 200_000, "validation": 100_000}
    assert request["allowed_roles"] == ["train", "validation"]
    assert request["labels_read"] is False
    tasks = gate_tasks()
    assert [row["task_id"] for row in tasks] == [
        "authenticate_release", "build_foundation", "preflight",
    ]
    assert tasks[-1]["kind"] == "gpu"
    assert tasks[-1]["dependencies"] == ["build_foundation"]


def test_contracts_freeze_matcher_and_input_semantics():
    assert view_contract()["matcher"]["candidate"] == "SALIENCE_PT_LINEAR"
    inputs = input_contract(capacity=512)
    assert inputs["truncation"] == "forbidden_fail_before_training"
    assert inputs["invalid_tracking"].startswith("zero_without")


def test_gate_command_plan_is_tigris_exact_dependency_dag(tmp_path):
    from hlt_classification.cms_proxy_ladder.submission import gate_plan

    source = artifact("SOURCE", commit="a" * 40, files={})
    request = release_request(study_root=tmp_path / "proxy", offline_root=tmp_path / "offline")
    gate = artifact(
        "GATE_SPEC", parents={"source": source["content_hash"], "request": request["content_hash"]},
        source=source, request=request, gate_root=str((tmp_path / "gate").resolve()),
        project_dir=str(tmp_path.resolve()), source_commit="a" * 40,
        capacity=512, execution_site=execution_site("tigris_gh200"),
        tasks=gate_tasks(), workers=16, full_views_persisted=False,
        admission="release_then_new_assignments_then_real_tigris_full_population_preflight",
    )
    plan = gate_plan(gate)
    commands = {row["task_id"]: row for row in plan["commands"]}
    assert list(commands) == ["authenticate_release", "build_foundation", "preflight"]
    assert "--gres=gpu:gh200:1" not in commands["authenticate_release"]["command"]
    assert "--gres=gpu:gh200:1" in commands["preflight"]["command"]
    assert any(
        "${JOB_build_foundation}" in value
        for value in commands["preflight"]["command"]
    )
    assert all("--partition=tigris" in row["command"] for row in commands.values())


def test_sporc_debug_gate_is_bounded_and_transfers_science_to_tier3(tmp_path, monkeypatch):
    from hlt_classification.cms_proxy_ladder import submission

    tasks = debug_gate_tasks()
    assert [row["task_id"] for row in tasks] == [
        "authenticate_release", "build_foundation", "preflight",
    ]
    assert all(row["minutes"] <= 480 for row in tasks)
    assert max(row["cpus"] for row in tasks) == 36
    spec = artifact(
        "GATE_SPEC", version=2, gate_root=str(tmp_path / "gate"),
        project_dir=str(tmp_path), measurement_site=execution_site("sporc_a100_debug"),
        execution_site=execution_site("sporc_a100"), tasks=tasks,
    )
    monkeypatch.setattr(submission, "validate_gate", lambda value: value["content_hash"])
    plan = submission.gate_plan(spec)
    commands = {row["task_id"]: row for row in plan["commands"]}
    assert all("--partition=debug" in row["command"] for row in commands.values())
    assert all("--qos=qos_tier3" in row["command"] for row in commands.values())
    assert "--gres=gpu:a100:1" in commands["preflight"]["command"]
    assert "--gres=gpu:a100:1" not in commands["build_foundation"]["command"]
    assert "JC2_SITE=sporc_a100_debug" in commands["preflight"]["command"][-1]


def test_science_command_plan_has_17_fits_12_reducers_and_cpu_tail(tmp_path, monkeypatch):
    from hlt_classification.cms_proxy_ladder import submission

    foundation = artifact("FOUNDATION_TEST", marker="synthetic")
    plan = build_scientific_plan(foundation)
    spec = artifact(
        "CAMPAIGN_SPEC_TEST", campaign_root=str(tmp_path / "campaign"),
        project_dir=str(tmp_path), scientific_plan=plan, tasks=task_graph(plan),
        runtime_profile={
            "cpus": 32, "memory_mb": 384_000,
            "train_minutes": 600, "reduce_minutes": 120,
            "execution_site": execution_site("tigris_gh200"),
        },
    )
    monkeypatch.setattr(submission, "validate_campaign", lambda value: value["content_hash"])
    command_plan = submission.science_plan(spec)
    assert len(command_plan["commands"]) == 31
    rows = {row["task_id"]: row for row in command_plan["commands"]}
    assert sum(name.startswith("train_") for name in rows) == 17
    assert sum(name.startswith("reduce_") for name in rows) == 12
    assert "--gres=gpu:gh200:1" in rows["train_U000"]["command"]
    assert "--gres=gpu:gh200:1" not in rows["aggregate"]["command"]
    assert rows["campaign_complete"]["dependencies"] == ["aggregate"]


def test_small_committed_release_is_label_blind_and_identity_exact(tmp_path, monkeypatch):
    from hlt_classification.cms_proxy_ladder import release as module
    from hlt_classification.data.cache_contracts import sha256_file

    study_root = tmp_path / "study"
    offline_root = tmp_path / "offline"
    (study_root / "blocks").mkdir(parents=True)
    (study_root / "shards").mkdir()
    offline_root.mkdir()
    sources, blocks = {}, {}
    for role, offset in (("train", 0), ("validation", 100)):
        source_path = offline_root / f"{role}.root"
        source_path.write_bytes(f"authenticated-{role}-offline-placeholder".encode())
        identities = np.asarray([
            np.frombuffer(hashlib.sha256(f"row:{offset + index}".encode()).digest(), np.uint8)
            for index in range(7)
        ], np.uint8)
        counts = np.asarray([2] * len(identities), np.int64)
        offsets = np.concatenate(([0], np.cumsum(counts))).astype("<i8")
        particles_count = int(offsets[-1])
        block_path = study_root / "blocks" / f"{role}.npz"
        np.savez(
            block_path,
            offsets=offsets, jet_identity=identities,
            p4=np.tile(np.asarray([[1., 0., 0., 1.]], dtype="<f8"), (particles_count, 1)),
            charge=np.zeros(particles_count, dtype="i1"),
            category=np.ones(particles_count, dtype="i1"),
            tracking=np.zeros((particles_count, 4), dtype="<f8"),
            valid=np.zeros((particles_count, 4), dtype="?"),
        )
        sources[role] = {
            "path": f"{role}.root", "sha256": sha256_file(source_path),
            "source": "train_qcd", "tree_key": "tree;1", "raw_entries": 7,
            "role": role,
        }
        blocks[role] = {
            "relative": f"blocks/{role}.npz", "sha256": sha256_file(block_path),
            "bytes": block_path.stat().st_size, "jets": 7,
        }
    study = {
        "contract": "CMS2JC2_PROXY_STUDY/v1",
        "content_hash": "ab" * 32, "data_root": str(offline_root),
        "population": {"files": list(sources.values())},
        "shards": [
            {"shard_id": "train", "role": "train"},
            {"shard_id": "validation", "role": "validation"},
            {"shard_id": "sealed", "role": "final_test"},
        ],
    }
    (study_root / "study_spec.json").write_text(json.dumps(study))
    receipts = {
        "train": {
            "shard_id": "train", "role": "train", "jets": 7,
            "content_hash": "cd" * 32, "ordered_identities": "1" * 64,
            "blocks": [blocks["train"]],
        },
        "validation": {
            "shard_id": "validation", "role": "validation", "jets": 7,
            "content_hash": "ef" * 32, "ordered_identities": "2" * 64,
            "blocks": [blocks["validation"]],
        },
    }
    for name, receipt in receipts.items():
        (study_root / "shards" / f"{name}.json").write_text(json.dumps(receipt))
    (study_root / "shards" / "sealed.json").write_text("forbidden final-test sentinel")
    entries = np.arange(7, dtype=np.int64)
    monkeypatch.setattr(module, "COUNTS", {"train": 3, "validation": 2})
    monkeypatch.setattr(module, "validate_study", lambda value: value)
    monkeypatch.setattr(
        module.output, "verify_shard", lambda study, receipt, physical=False: receipt,
    )
    monkeypatch.setattr(
        module.population, "entries_for", lambda population, shard: (sources[shard["role"]], entries),
    )
    request = module.release_request(study_root=study_root, offline_root=offline_root)
    manifest = module.build_release(request, output_root=tmp_path / "release")
    bank = module.load_bank(manifest, root=tmp_path / "release")
    assert manifest["counts"] == {"train": 3, "validation": 2}
    assert manifest["labels_read"] is False
    assert len(bank["identity"]) == 5
    assert len({bytes(row) for row in bank["identity"]}) == 5
