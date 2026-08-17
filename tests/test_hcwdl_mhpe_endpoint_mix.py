from __future__ import annotations

import hashlib
import numpy as np
import pytest

from hlt_classification.data.cache_contracts import canonical_sha256, load_json
from hlt_classification.scouting.hcwdl_mhpe_endpoint_mix import (
    CREATION_PHRASE, GRAPH_SHA256, NODES, SOURCE_ENDPOINT, campaign_tasks,
    _validate_m0paired_lineage, command_plan, graph_payload, recipe_payload,
    validate_recipe,
    validate_source_semantics,
)
from hlt_classification.scouting.hcwdl_mhpe_graph import (
    PROFILE_C25P75_300K60, PROFILE_DENSE_C25P75_300K60,
)
from hlt_classification.scouting.hcwdl_mhpe_endpoint_mix_targets import (
    ephemeral_from_manifest, fp32_softmax, load_target, mix_probabilities,
    publish_lock, publish_manifest, publish_target, validate_bundle,
)
from hlt_classification.scouting.hcwdl_mhpe_endpoint_refinement import (
    BLENDS as REFINEMENT_BLENDS, blend_probabilities,
    validate_exact_hlt_training_report,
)
from hlt_classification.scouting.hcwdl_mhpe_endpoint_mix_runner import (
    endpoint_mix_loss,
)
from hlt_classification.scouting.hcwdl_mhpe_endpoint_mix_recovery import failed_downstream_closure


def test_endpoint_mix_graph_is_exactly_four_paired_m1_fits():
    assert list(NODES) == ["M1_D0only", "M1_mix90", "M1_mix75", "M1_mix50"]
    assert SOURCE_ENDPOINT == "D000E"
    assert graph_payload()["source_endpoint"] == SOURCE_ENDPOINT
    assert [NODES[name].payload()["endpoint_weight"] for name in NODES] == [
        [1, 1], [9, 10], [3, 4], [1, 2],
    ]
    assert len({NODES[name].payload()["seed_alias"] for name in NODES}) == 1
    assert all(NODES[name].payload()["ce_weight"] == .10 for name in NODES)
    assert all(NODES[name].payload()["kd_weight"] == .90 for name in NODES)
    assert graph_payload()["schema_version"] == 2
    assert graph_payload()["content_hash"] == GRAPH_SHA256


def test_endpoint_mix_task_graph_is_seven_jobs_and_four_parallel_children():
    tasks = {row["task_id"]: row for row in campaign_tasks()}
    assert len(tasks) == 7
    assert tasks["build_targets"]["dependencies"] == []
    for node_id in NODES:
        assert tasks[f"train_{node_id}"]["dependencies"] == ["build_targets"]
    assert tasks["aggregate"]["dependencies"] == [f"train_{name}" for name in NODES]
    assert tasks["campaign_complete"]["dependencies"] == ["aggregate"]
    spec = {
        "content_hash": "a" * 64, "project_dir": "/project",
        "spec_path": "/campaign/campaign_spec.json", "tasks": campaign_tasks(),
        "resources": {
            "gpu": {"cpus": 8, "memory": "96G", "walltime": "06:00:00", "gpu": "gpu:gh200:1"},
            "cpu": {"cpus": 4, "memory": "32G", "walltime": "01:00:00", "gpu": None},
        },
    }
    plan = command_plan(spec)
    assert plan["schema_version"] == 2
    assert len(plan["commands"]) == 7
    assert all(any(item.startswith("--job-name=hcwmix_") for item in row["command"]) for row in plan["commands"])


def test_endpoint_mix_recipe_is_exactly_60_pass_unweighted_c10p90_t1():
    recipe = recipe_payload(source_recipe_sha256="a" * 64)
    assert validate_recipe(recipe) == recipe["content_hash"]
    assert recipe["training_passes"] == 60
    assert recipe["loss"] == {"ce": .10, "kd": .90, "temperature": 1.0}
    assert recipe["class_weighting"] == "unweighted_per_jet_population_mean_v1"


def test_endpoint_mix_all_nodes_construct_registered_ub_losses():
    for node_id in NODES:
        loss = endpoint_mix_loss(node_id)
        assert loss.arm == f"HCWDL_UB_MHPE_ENDPOINT_MIX_{node_id}"
        assert (loss.ce, loss.parent_kd, loss.grandparent_kd) == (.10, .90, 0)
        assert (loss.parent_temperature, loss.grandparent_temperature) == (1.0, 1.0)


def test_endpoint_mix_v2_accepts_only_the_original_c25p75_d000e_source():
    spec = {
        "role_counts": {"train": 300_000, "validation": 100_000, "final_test": 100_000},
        "final_test_accessed": False,
    }
    validate_source_semantics(spec, profile=PROFILE_C25P75_300K60)
    with pytest.raises(ValueError, match="C25P75 300k/60-pass"):
        validate_source_semantics(spec, profile=PROFILE_DENSE_C25P75_300K60)
    with pytest.raises(ValueError, match="C25P75 300k/60-pass"):
        validate_source_semantics({**spec, "final_test_accessed": True}, profile=PROFILE_C25P75_300K60)


def test_endpoint_mix_m0_lineage_uses_report_bound_checkpoint(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"authenticated checkpoint")
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    report_hash = "a" * 64
    report = {
        "selected_checkpoint": checkpoint.name,
        "selected_checkpoint_sha256": checkpoint_hash,
    }
    # The immutable 300k reuse-lock schema binds the report, not a redundant
    # top-level checkpoint field.  The report and file close that lineage.
    reuse = {"m0paired_report_sha256": report_hash}
    _validate_m0paired_lineage(
        reuse=reuse, report=report, report_hash=report_hash,
        checkpoint=checkpoint,
    )
    with pytest.raises(ValueError, match="M0paired lineage"):
        _validate_m0paired_lineage(
            reuse={"m0paired_report_sha256": "b" * 64}, report=report,
            report_hash=report_hash, checkpoint=checkpoint,
        )
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="M0paired lineage"):
        _validate_m0paired_lineage(
            reuse=reuse, report=report, report_hash=report_hash,
            checkpoint=checkpoint,
        )


def test_endpoint_mix_numerics_are_fp32_softmax_exact_rational_fp64():
    logits = np.asarray([[8.0, 0.0] + [-1.0] * 13], np.float32)
    m0 = fp32_softmax(logits)
    d0 = np.full((1, 15), 1 / 15, np.float32)
    actual = mix_probabilities(d0, m0, numerator=9, denominator=10)
    expected = np.asarray((d0.astype(np.float64) * 9 + m0.astype(np.float64)) / 10, dtype="<f4")
    assert actual.dtype.str == "<f4"
    assert np.array_equal(actual, expected)
    assert np.array_equal(mix_probabilities(d0, m0, numerator=1, denominator=1), d0)
    assert np.allclose(actual.sum(1), 1, rtol=0, atol=2e-6)


def test_endpoint_refinement_blends_are_fixed_and_exact_rational():
    assert REFINEMENT_BLENDS == (
        ("D000E", 1, 1),
        ("D000E75_M1_25", 3, 4),
        ("D000E50_M1_50", 1, 2),
        ("D000E25_M1_75", 1, 4),
        ("M1", 0, 1),
    )
    endpoint = np.zeros((2, 15), np.float32); endpoint[:, 0] = 1
    refinement = np.zeros((2, 15), np.float32); refinement[:, 1] = 1
    actual = blend_probabilities(
        endpoint, refinement, endpoint_numerator=3, denominator=4,
    )
    assert actual.dtype.str == "<f4"
    assert np.array_equal(actual[:, :2], np.asarray([[.75, .25]] * 2, np.float32))
    with pytest.raises(ValueError, match="weight"):
        blend_probabilities(
            endpoint, refinement, endpoint_numerator=5, denominator=4,
        )


def test_endpoint_refinement_authenticates_registered_and_executed_hlt_input():
    report = {
        "scientific_config": {
            "node": {"student_domain": "hlt"},
            "input_key": "hlt",
        },
        "config": {"model_input": "hlt"},
    }
    validate_exact_hlt_training_report(report)
    for path, value in (
        (("scientific_config", "node", "student_domain"), "privileged"),
        (("scientific_config", "input_key"), "privileged"),
        (("config", "model_input"), "privileged"),
    ):
        changed = {
            "scientific_config": {
                "node": dict(report["scientific_config"]["node"]),
                "input_key": report["scientific_config"]["input_key"],
            },
            "config": dict(report["config"]),
        }
        target = changed
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        with pytest.raises(PermissionError, match="not exact-HLT"):
            validate_exact_hlt_training_report(changed)


def test_endpoint_mix_target_bundle_roundtrip_and_tamper(tmp_path):
    parents = {"campaign": "a" * 64}; lineage = {SOURCE_ENDPOINT: "b" * 64, "M0paired": "c" * 64}
    identities = ["j0", "j1"]; d0 = np.full((2, 15), 1 / 15, np.float32)
    m0 = fp32_softmax(np.arange(30, dtype=np.float32).reshape(2, 15) / 10)
    manifest_hashes = {}
    for node_id, node in NODES.items():
        manifest_hashes[node_id] = {}
        for role in ("train", "validation"):
            directory = tmp_path / node_id
            metadata = publish_target(
                directory / f"{role}_all", node_id=node_id, role=role,
                identities=identities,
                probabilities=mix_probabilities(
                    d0, m0, numerator=node.endpoint_weight_numerator,
                    denominator=node.endpoint_weight_denominator,
                ), component_lineage=lineage, parents=parents,
                producer_commit="d" * 40,
            )
            manifest = publish_manifest(
                directory / f"{role}_manifest.json", node_id=node_id, role=role,
                target_metadata=directory / f"{role}_all.json",
                expected_rows=2, parents=parents,
            )
            assert metadata["node_id"] == node_id
            manifest_hashes[node_id][role] = manifest["content_hash"]
    lock = publish_lock(tmp_path / "lock.json", manifests=manifest_hashes, parents=parents)
    assert validate_bundle(tmp_path)[0] == lock["content_hash"]
    ephemeral = ephemeral_from_manifest(
        tmp_path / "M1_mix90/train_manifest.json",
        split_manifest_sha256=canonical_sha256("split"),
    )
    assert np.array_equal(ephemeral.join(["j1", "j0"]), ephemeral.probabilities[[1, 0]])
    npz = tmp_path / "M1_mix90/train_all.npz"
    npz.write_bytes(npz.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="bytes"):
        load_target(tmp_path / "M1_mix90/train_all.json")


def test_endpoint_mix_rejects_bad_components_and_weights():
    good = np.full((2, 15), 1 / 15, np.float32)
    with pytest.raises(ValueError, match="weight"):
        mix_probabilities(good, good, numerator=11, denominator=10)
    bad = good.copy(); bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="component"):
        mix_probabilities(good, bad, numerator=9, denominator=10)


def test_endpoint_mix_recovery_is_exact_failed_downstream_closure():
    closure = failed_downstream_closure(["train_M1_mix90"])
    assert closure == ("train_M1_mix90", "aggregate", "campaign_complete")
    closure = failed_downstream_closure(["build_targets"])
    assert closure == tuple(row["task_id"] for row in campaign_tasks())
