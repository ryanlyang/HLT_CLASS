from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

import hlt_classification.scouting.hcwdl_representation_locks as locks
from hlt_classification.data.cache_contracts import with_content_hash


H = "a" * 64


def _row(node_id: str, *, teacher: bool) -> dict[str, object]:
    if node_id == "TOFF":
        domain, track = "native_offline", "shared"
    elif teacher:
        domain = (
            "hlt" if node_id.startswith("D0")
            else "d" + node_id[1:].rstrip("cw").lower()
        )
        track = "cold" if node_id.endswith("c") else (
            "warm" if node_id.endswith("w") else "shared"
        )
    else:
        domain = "hlt"
        track = "cold" if node_id.endswith("c") else (
            "warm" if node_id.endswith("w") else "shared"
        )
    ordinal = sum(node_id.encode("ascii"))
    return {
        "node_id": node_id, "domain": domain, "track": track,
        "report_path": f"/authenticated/{node_id}/training_report.json",
        "report_sha256": f"{ordinal % 16:x}" * 64,
        "checkpoint_path": f"/authenticated/{node_id}/selected.pt",
        "checkpoint_sha256": f"{(ordinal + 1) % 16:x}" * 64,
        "checkpoint_byte_sha256": f"{(ordinal + 1) % 16:x}" * 64,
    }


def _imports():
    teachers = {node: _row(node, teacher=True) for node in locks.IMPORTED_TEACHERS}
    controls = {
        node: _row(node, teacher=False) for node in locks.IMPORTED_LOGIT_CONTROLS
    }
    imported = {**teachers, **controls}
    architecture = {
        "content_hash": "1" * 64,
        "checkpoint_audits": [
            {
                "node_id": node,
                "checkpoint_sha256": row["checkpoint_byte_sha256"],
                "report_path": Path(str(row["report_path"])).resolve().as_posix(),
                "report_sha256": row["report_sha256"],
                "checkpoint_path": Path(str(row["checkpoint_path"])).resolve().as_posix(),
                "actual_file_evidence": True,
            }
            for node, row in sorted(imported.items())
        ],
    }
    loss = {
        "content_hash": "2" * 64,
        "parent_artifacts": [
            {
                "node_id": node,
                "training_report_sha256": row["report_sha256"],
                "checkpoint_sha256": row["checkpoint_sha256"],
            }
            for node, row in sorted(imported.items())
        ],
    }
    return teachers, controls, architecture, loss


def _digest(label: str) -> str:
    digit = f"{sum(label.encode()) % 15 + 1:x}"
    return digit * 64


def _authority_projection():
    teachers, controls, architecture, loss = _imports()
    parents = {name: _digest(name) for name in locks.PARENT_AUTHORITY_PARENT_KEYS}
    parents["architecture_attestation"] = architecture["content_hash"]
    parents["parent_loss_attestation"] = loss["content_hash"]
    campaign = {
        "contract": "HCWDL_CAMPAIGN_SPEC/v7", "source_commit": "4" * 40,
        "endpoint_continuation": "preauthorized_automatic",
    }
    recipe = {"contract": "HCWDL_RECIPE/v4"}
    return {
        "campaign": campaign, "recipe": recipe,
        "architecture_attestation": architecture,
        "parent_loss_attestation": loss, "parents": parents,
    }, teachers, controls


def _dummy_paths():
    return {
        name: Path(f"C:/authority/{name}.json")
        for name in locks.PARENT_AUTHORITY_FILE_KEYS
    }


def _dummy_qualifiers():
    return {
        name: Path(f"C:/authority/{name}.json")
        for name in ("T0", "TFS", "THC", "TSOFT", "TSHELL", "TOFF")
    }


def test_file_backed_parent_import_binds_v7_v4_and_every_authority_parent(
    monkeypatch,
) -> None:
    authority, teachers, controls = _authority_projection()
    calls = []
    monkeypatch.setattr(
        locks, "_validated_parent_authority",
        lambda **kwargs: calls.append(kwargs) or authority,
    )
    artifact = locks.build_parent_import_from_files(
        authority_files=_dummy_paths(), qualifier_report_paths=_dummy_qualifiers(),
        teachers=teachers, logit_controls=controls,
    )
    assert calls
    assert artifact["contract"] == "HCWDL_REPRESENTATION_PARENT_IMPORT/v2"
    assert artifact["parents"] == authority["parents"]
    assert artifact["payload"]["parent_campaign_contract"] == "HCWDL_CAMPAIGN_SPEC/v7"
    assert artifact["payload"]["parent_recipe_contract"] == "HCWDL_RECIPE/v4"
    assert artifact["payload"]["authority_derived_from_registered_files"] is True
    assert locks.validate_parent_import(artifact) == artifact["content_hash"]


def test_file_backed_parent_import_derives_exact_rows_from_architecture(
    monkeypatch,
) -> None:
    authority, teachers, controls = _authority_projection()
    monkeypatch.setattr(locks, "_validated_parent_authority", lambda **_: authority)
    artifact = locks.build_parent_import_from_files(
        authority_files=_dummy_paths(), qualifier_report_paths=_dummy_qualifiers(),
    )
    expected_teachers = deepcopy(teachers)
    expected_controls = deepcopy(controls)
    for row in (*expected_teachers.values(), *expected_controls.values()):
        row["report_path"] = Path(str(row["report_path"])).resolve().as_posix()
        row["checkpoint_path"] = Path(
            str(row["checkpoint_path"])
        ).resolve().as_posix()
    assert {
        row["node_id"]: row for row in artifact["payload"]["teachers"]
    } == expected_teachers
    assert {
        row["node_id"]: row for row in artifact["payload"]["logit_controls"]
    } == expected_controls


def test_parent_import_reopens_authority_and_rejects_cross_campaign_splice(
    monkeypatch,
) -> None:
    authority, teachers, controls = _authority_projection()
    monkeypatch.setattr(locks, "_validated_parent_authority", lambda **_: authority)
    artifact = locks.build_parent_import_from_files(
        authority_files=_dummy_paths(), qualifier_report_paths=_dummy_qualifiers(),
        teachers=teachers, logit_controls=controls,
    )
    alternate = deepcopy(authority)
    alternate["parents"] = dict(alternate["parents"])
    alternate["parents"]["parent_campaign_spec"] = "0" * 64
    monkeypatch.setattr(locks, "_validated_parent_authority", lambda **_: alternate)
    with pytest.raises(ValueError, match="authority files differ"):
        locks.validate_parent_import_against_authority_files(
            artifact, authority_files=_dummy_paths(),
            qualifier_report_paths=_dummy_qualifiers(),
        )


def test_parent_loss_source_authority_is_cross_bound_to_reopened_v7_campaign() -> None:
    campaign_sha256 = "1" * 64
    source_commit = "2" * 40
    campaign = {"source_commit": source_commit}
    attestation = {
        "parent_campaign_spec_sha256": campaign_sha256,
        "parent_source_commit": source_commit,
        "parent_source_snapshot": {"git_commit": source_commit},
    }
    locks._validate_parent_loss_campaign_source(
        attestation, campaign=campaign, campaign_sha256=campaign_sha256,
    )

    for field, value in (
        ("parent_campaign_spec_sha256", "3" * 64),
        ("parent_source_commit", "4" * 40),
        ("parent_source_snapshot", {"git_commit": "5" * 40}),
    ):
        forged = deepcopy(attestation)
        forged[field] = value
        with pytest.raises(PermissionError, match="parent campaign authority"):
            locks._validate_parent_loss_campaign_source(
                forged, campaign=campaign, campaign_sha256=campaign_sha256,
            )


def test_attestation_only_parent_import_boundary_is_nonauthorizing() -> None:
    with pytest.raises(PermissionError, match="registered parent authority files"):
        locks.validate_parent_import_against_evidence({}, architecture_attestation={})


def test_legacy_hash_and_boolean_fixture_requires_explicit_flag_and_is_not_v2() -> None:
    teachers, controls, architecture, loss = _imports()
    kwargs = {
        "parent_campaign_spec_sha256": "3" * 64,
        "parent_source_commit": "4" * 40,
        "lineage_hashes": {name: H for name in locks.PARENT_IMPORT_LINEAGE_KEYS},
        "architecture_attestation": architecture,
        "parent_loss_attestation": loss,
        "teachers": teachers, "logit_controls": controls,
        "original_contract_validation": {
            name: True for name in locks.PARENT_IMPORT_VALIDATION_KEYS
        },
    }
    with pytest.raises(PermissionError, match="nonauthorizing flag"):
        locks.build_parent_import_fixture(**kwargs, nonauthorizing_fixture=False)
    fixture = locks.build_parent_import_fixture(
        **kwargs, nonauthorizing_fixture=True,
    )
    assert fixture["contract"] == "HCWDL_REPRESENTATION_PARENT_IMPORT/v1"
    with pytest.raises(ValueError, match="contract"):
        locks.validate_parent_import(fixture)


def test_parent_import_rejects_non_hlt_logit_control_and_byte_hash_mismatch() -> None:
    _, controls, _, _ = _imports()
    bad_domain = deepcopy(controls["M0"])
    bad_domain["domain"] = "native_offline"
    with pytest.raises(ValueError, match="logit-control.*domain"):
        locks._import_row(bad_domain, node_id="M0", teacher=False)
    bad_bytes = deepcopy(controls["M0"])
    bad_bytes["checkpoint_byte_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="selected-checkpoint byte proof"):
        locks._import_row(bad_bytes, node_id="M0", teacher=False)


def test_complete_lock_chain_rejects_forged_leaf_predecessor() -> None:
    from hlt_classification.scouting import hcwdl_locks as parent_locks

    campaign = "1" * 64
    rows = []
    parent = None
    for level in (
        "assignment", "recipe", "shell_endpoint_qualification",
        "confirmation_registry", "finalist",
    ):
        row = parent_locks.create_lock(
            level, campaign_spec_sha256=campaign, parent_lock=parent,
            payload={"level": level},
        )
        rows.append(row)
        parent = row
    locks._validate_lock_chain(
        campaign_sha256=campaign, assignment_lock=rows[0], recipe_lock=rows[1],
        endpoint_lock=rows[2], confirmation_lock=rows[3], finalist_lock=rows[4],
    )
    forged = deepcopy(rows[4])
    forged.pop("content_hash")
    forged["parent_lock_sha256"] = "f" * 64
    forged = with_content_hash(forged)
    with pytest.raises(ValueError, match="predecessor differs"):
        locks._validate_lock_chain(
            campaign_sha256=campaign, assignment_lock=rows[0], recipe_lock=rows[1],
            endpoint_lock=rows[2], confirmation_lock=rows[3], finalist_lock=forged,
        )


def test_confirmation_registry_rejects_screen_selection_splice() -> None:
    from hlt_classification.scouting.hcwdl_reporting import (
        build_confirmation_registry,
    )

    screen = with_content_hash({
        "contract": "HCWDL_SCREEN_AGGREGATE/v1", "schema_version": 1,
        "selected_intermediate_cold": {"selected_node_id": "M2c"},
        "selected_intermediate_warm": {"selected_node_id": "M2w"},
    })
    registry = build_confirmation_registry(
        screen, seeds=(11, 22, 33, 44, 55),
        include_label_only_warm_continuation=False,
    )
    spliced = deepcopy(registry)
    spliced[7]["node_id"] = "M5c"
    with pytest.raises(ValueError, match="differs from screen selection"):
        locks._require_exact_confirmation_registry(
            spliced, screen=screen,
            include_label_only_warm_continuation=False,
        )


def test_finalist_registry_rejects_bogus_extra_row() -> None:
    expected = [{
        "node_id": "M0", "seed": 11, "checkpoint_sha256": "1" * 64,
        "report_sha256": "2" * 64, "report_path": "/parent/M0.json",
    }]
    actual = deepcopy(expected)
    actual.append({
        "node_id": "BOGUS", "seed": 11, "checkpoint_sha256": "3" * 64,
        "report_sha256": "4" * 64, "report_path": "/parent/bogus.json",
    })
    with pytest.raises(ValueError, match="canonical reports"):
        locks._require_exact_finalists(actual, expected)


def test_primary_report_lineage_rejects_screen_teacher_splice() -> None:
    from hlt_classification.scouting.hcwdl_ladder import (
        GRAPH_SHA256, NODE_REGISTRY,
    )
    from hlt_classification.scouting.training import derive_seed

    split = "1" * 64
    source = "2" * 64
    assignment = "3" * 64
    qualification = "4" * 64
    recipe = "5" * 64
    screening = {
        "M1c": {"content_hash": "6" * 64},
        "D25c": {"content_hash": "7" * 64},
    }
    report = {
        "experiment_id": "M2c", "complete": True,
        "selected_checkpoint_sha256": "8" * 64,
        "config": {
            "master_seed": derive_seed(11, "hcwdl/M2c"),
            "selection_policy": "hcwdl_macro_auc",
        },
        "scientific_config": {
            "campaign": "HCWDL", "graph_sha256": GRAPH_SHA256,
            "recipe_sha256": recipe, "node": NODE_REGISTRY["M2c"].payload(),
        },
        "parents": {
            "split_manifest_sha256": split,
            "source_snapshot_sha256": source,
            "assignment_lock_sha256": assignment,
            "qualification_lock_sha256": qualification,
            "recipe": recipe,
            "teacher_predecessor_report_sha256": "6" * 64,
            "teacher_privileged_report_sha256": "7" * 64,
        },
    }
    arguments = {
        "node_id": "M2c", "replicate_seed": 11,
        "split_sha256": split, "source_sha256": source,
        "assignment_lock_sha256": assignment,
        "qualification_lock_sha256": qualification,
        "recipe_sha256": recipe, "screening_reports": screening,
    }
    locks._validate_primary_engine_lineage(report, **arguments)
    spliced = deepcopy(report)
    spliced["parents"] = dict(spliced["parents"])
    spliced["parents"]["teacher_privileged_report_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="primary report lineage"):
        locks._validate_primary_engine_lineage(spliced, **arguments)


def test_control_report_lineage_rejects_unrelated_screen_teacher() -> None:
    from hlt_classification.scouting.training import derive_seed

    split = "1" * 64
    source = "2" * 64
    assignment = "3" * 64
    qualification = "4" * 64
    recipe = "5" * 64
    screening = {"M0": {"content_hash": "6" * 64}}
    report = {
        "experiment_id": "NULL_M1_SELF_KD", "complete": True,
        "selected_checkpoint_sha256": "8" * 64,
        "config": {
            "master_seed": derive_seed(11, "hcwdl/control/NULL_M1_SELF_KD"),
            "selection_policy": "hcwdl_macro_auc",
        },
        "scientific_config": {
            "campaign": "HCWDL", "control_id": "NULL_M1_SELF_KD",
            "initialization": "fresh",
        },
        "parents": {
            "split_manifest_sha256": split,
            "source_snapshot_sha256": source,
            "assignment_lock_sha256": assignment,
            "qualification_lock_sha256": qualification,
            "recipe_sha256": recipe,
            "teacher_report_sha256": "6" * 64,
        },
    }
    arguments = {
        "control_id": "NULL_M1_SELF_KD", "replicate_seed": 11,
        "split_sha256": split, "source_sha256": source,
        "assignment_lock_sha256": assignment,
        "qualification_lock_sha256": qualification,
        "recipe_sha256": recipe, "screening_reports": screening,
    }
    locks._validate_control_engine_lineage(report, **arguments)
    unrelated = deepcopy(screening)
    unrelated["M0"] = {"content_hash": "9" * 64}
    with pytest.raises(ValueError, match="control report lineage"):
        locks._validate_control_engine_lineage(
            report, **{**arguments, "screening_reports": unrelated},
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("extra_parent", "parent lineage registry"),
        ("bad_commit", "source commit"),
        ("legacy_campaign", "current contract"),
        ("legacy_recipe", "current contract"),
        ("manual_continuation", "current contract"),
        ("false_file_authority", "file-authority completion proof"),
    ),
)
def test_v2_validator_recomputes_closed_identity(monkeypatch, mutation, message) -> None:
    authority, teachers, controls = _authority_projection()
    monkeypatch.setattr(locks, "_validated_parent_authority", lambda **_: authority)
    artifact = locks.build_parent_import_from_files(
        authority_files=_dummy_paths(), qualifier_report_paths=_dummy_qualifiers(),
        teachers=teachers, logit_controls=controls,
    )
    forged = deepcopy(artifact)
    forged.pop("content_hash")
    if mutation == "extra_parent":
        forged["parents"]["unused"] = "f" * 64
    elif mutation == "bad_commit":
        forged["payload"]["parent_source_commit"] = "not-a-commit"
    elif mutation == "legacy_campaign":
        forged["payload"]["parent_campaign_contract"] = "HCWDL_CAMPAIGN_SPEC/v6"
    elif mutation == "legacy_recipe":
        forged["payload"]["parent_recipe_contract"] = "HCWDL_RECIPE/v3"
    elif mutation == "manual_continuation":
        forged["payload"]["endpoint_continuation"] = "manual_posthoc"
    else:
        forged["payload"]["authority_derived_from_registered_files"] = False
    forged = with_content_hash(forged)
    with pytest.raises(ValueError, match=message):
        locks.validate_parent_import(forged)


def test_strict_authority_rejects_v6_before_any_generic_validator(monkeypatch) -> None:
    artifacts = {
        name: {"contract": "TEST/v1", "schema_version": 1, "content_hash": H}
        for name in locks.PARENT_AUTHORITY_FILE_KEYS
    }
    artifacts["campaign_spec"] = {
        "contract": "HCWDL_CAMPAIGN_SPEC/v6", "schema_version": 6,
        "content_hash": H,
    }
    monkeypatch.setattr(
        locks, "_registered_json_file",
        lambda path, *, name: (Path(path), artifacts[name]),
    )
    with pytest.raises(ValueError, match="HCWDL_CAMPAIGN_SPEC/v7"):
        locks._validated_parent_authority(
            authority_files=_dummy_paths(), qualifier_report_paths=_dummy_qualifiers(),
        )
