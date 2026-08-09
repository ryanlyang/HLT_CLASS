from __future__ import annotations

import pytest

from hlt_classification.data.cache_contracts import canonical_sha256, with_content_hash
from hlt_classification.scouting.hcwdl_ladder import GRAPH_SHA256
from hlt_classification.scouting.highcov_resources import resource_validation_report
from hlt_classification.scouting.hcwdl_representation_final_policy import (
    build_final_assignment_spec, build_pretraining_finalist_policy_commitment,
    project_parent_finalists, validate_final_assignment_spec,
    validate_pretraining_finalist_policy_commitment,
)
from hlt_classification.scouting.training import derive_seed


def _report(node_id: str, seed: int, marker: str):
    return with_content_hash({
        "contract": "TEST_HCWDL_PARENT_REPORT/v1", "schema_version": 1,
        "experiment_id": marker,
        "config": {"master_seed": derive_seed(seed, f"hcwdl/{node_id}")},
        "selected_checkpoint_sha256": canonical_sha256({"checkpoint": marker}),
    })


def _parent_authority():
    raw_rows = []
    locked_members = {}
    repeated = (
        "M0", "M6c", "M6w", "NULL_M1_SELF_KD",
        "NULL_M6_PREDECESSOR_ONLY", "M3c", "M3w",
    )
    for node_id in repeated:
        for seed in (11, 22):
            marker = f"{node_id}-{seed}"
            report = _report(node_id, seed, marker)
            raw_rows.append({
                "node_id": node_id, "seed": seed,
                "checkpoint_sha256": report["selected_checkpoint_sha256"],
                "report_sha256": report["content_hash"],
                "report_path": f"/parent/{marker}/training_report.json",
            })
            locked_members[f"{node_id}:{seed}"] = {
                "relative": f"{marker}/training_report.json", "report": report,
            }
    for node_id in ("D100", "TOFF"):
        report = _report(node_id, 1337, node_id)
        raw_rows.append({
            "node_id": node_id, "seed": 1337,
            "checkpoint_sha256": report["selected_checkpoint_sha256"],
            "report_sha256": report["content_hash"],
            "report_path": f"/parent/{node_id}/training_report.json",
        })
        locked_members[f"{node_id}:1337"] = {
            "relative": f"{node_id}/training_report.json", "report": report,
        }
    lock = with_content_hash({
        "contract": "HCWDL_LOCK/v1", "schema_version": 1,
        "campaign": "HCWDL", "level": "finalist",
        "campaign_spec_sha256": "a" * 64,
        "parent_lock_sha256": "b" * 64, "graph_sha256": GRAPH_SHA256,
        "payload": {
            "confirmation_report_sha256": "c" * 64,
            "finalists": raw_rows, "selection_used_validation_only": True,
        },
    })
    paired = {}
    for node_id in ("M0", "M6c", "M6w"):
        report = _report(node_id, 1337, f"{node_id}-screen")
        paired[node_id] = {
            "relative": f"screen/{node_id}/training_report.json", "report": report,
        }
    return lock, locked_members, paired


def test_final_assignment_spec_is_derived_from_exact_matcher_and_shards():
    resources = resource_validation_report()
    value = build_final_assignment_spec(
        matcher_resources=resources, source_partitions=("a.root", "b.root"),
    )
    assert validate_final_assignment_spec(
        value, matcher_resources=resources,
        source_partitions=("a.root", "b.root"),
    ) == value["content_hash"]
    changed = dict(resources)
    changed["donor_commit"] = "0" * 40
    changed = with_content_hash({
        key: raw for key, raw in changed.items() if key != "content_hash"
    })
    with pytest.raises(PermissionError, match="not canonical"):
        build_final_assignment_spec(
            matcher_resources=changed, source_partitions=("a.root", "b.root"),
        )


def test_parent_lock_projection_and_policy_reject_omission_substitution_and_tamper():
    lock, locked, paired = _parent_authority()
    parents = project_parent_finalists(
        parent_finalist_lock=lock, locked_model_members=locked,
        paired_screen_model_members=paired,
    )
    ids = {row["finalist_id"] for row in parents}
    assert {"M0", "M6c", "M6w", "D100", "TOFF"} <= ids
    assert {"M0__seed11", "M0__seed22"} <= ids
    value = build_pretraining_finalist_policy_commitment(
        parent_finalists=parents, parent_finalist_lock=lock,
    )
    assert validate_pretraining_finalist_policy_commitment(
        value, parent_finalists=parents, parent_finalist_lock=lock,
    ) == value["content_hash"]
    for changed in (
        parents[:-1],
        [*parents, {**parents[0], "finalist_id": "EXTRA"}],
        [{**parents[0], "checkpoint_sha256": "f" * 64}, *parents[1:]],
    ):
        with pytest.raises(ValueError):
            validate_pretraining_finalist_policy_commitment(
                value, parent_finalists=changed, parent_finalist_lock=lock,
            )
    tampered = dict(value)
    tampered["representation_endpoints"] = list(tampered["representation_endpoints"])
    tampered["representation_endpoints"][0] = {
        **tampered["representation_endpoints"][0], "screening_seed": 11,
    }
    tampered = with_content_hash({
        key: raw for key, raw in tampered.items() if key != "content_hash"
    })
    with pytest.raises(ValueError, match="not canonical"):
        validate_pretraining_finalist_policy_commitment(
            tampered, parent_finalists=parents, parent_finalist_lock=lock,
        )


def test_parent_projection_rejects_missing_lock_member_and_non1337_screen():
    lock, locked, paired = _parent_authority()
    locked.pop(next(iter(locked)))
    with pytest.raises(ValueError, match="omit or add"):
        project_parent_finalists(
            parent_finalist_lock=lock, locked_model_members=locked,
            paired_screen_model_members=paired,
        )
    lock, locked, paired = _parent_authority()
    paired["M0"] = {
        "relative": "screen/M0/training_report.json",
        "report": _report("M0", 22, "wrong-screen"),
    }
    with pytest.raises(ValueError, match="screening-seed"):
        project_parent_finalists(
            parent_finalist_lock=lock, locked_model_members=locked,
            paired_screen_model_members=paired,
        )
