from __future__ import annotations

import hashlib

import numpy as np
import pytest

from hlt_classification.data.cache_contracts import canonical_sha256, with_content_hash
from hlt_classification.scouting.hcwdl_final_stream import (
    ASSIGNMENT_FINAL_BRANCHES,
    HLT_FINAL_BRANCHES,
    NATIVE_OFFLINE_FINAL_BRANCHES,
    SELECTION_BRANCHES,
    SHELL_EXACT_FINAL_BRANCHES,
    build_branch_access_record,
    class_stratified_selection,
    iterate_final_hlt_inputs,
    iterate_final_native_offline_inputs,
    iterate_final_shell_exact_inputs,
    label_escrow_sidecar,
    load_label_escrow,
    publish_label_escrow,
    validate_projected_branches,
)
from hlt_classification.scouting.hcwdl_shared_final import (
    FINAL_EXECUTION_CLAIM_CONTRACT,
    build_final_task_registry,
    derive_claim_owner_id,
    issue_role_capability,
)
from hlt_classification.scouting.schema import LABEL_BRANCHES, TREE_NAME
from hlt_classification.scouting.streaming import iterate_projected_chunks


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _claim_registry(*, task_id: str, kind: str, branch_family: str, execution: str | None):
    def row(identity, row_kind, branch, output, dependencies, **extra):
        return {
            "task_id": identity, "kind": row_kind, "purpose": "test final boundary",
            "branch_family": branch, "source_partition": None,
            "finalist_id": None, "checkpoint_sha256": None,
            "resource_signature": {"class": "cpu"},
            "registered_outputs": [output], "dependencies": dependencies,
            **extra,
        }

    selection_id = task_id if kind == "row_selection" else "selection"
    prediction_id = task_id if kind == "prediction_shard" else "predict:M0"
    prediction_branch = branch_family if kind == "prediction_shard" else "hlt"
    tasks = (
        row(selection_id, "row_selection", "selection", "final/selection", []),
        row("assignment", "assignment_shard", "assignment", "final/assignment/shard", [selection_id],
            source_partition="source.root"),
        row("assignment-final", "assignment_finalize", None, "final/assignment/manifest", ["assignment"]),
        row("data", "data_attestation", None, "final/data", ["assignment-final"]),
        row("execute", "execution_lock", None, "final/execution", ["data"]),
        row(prediction_id, "prediction_shard", prediction_branch, "final/prediction/shard", ["execute"],
            source_partition="source.root", finalist_id="M0", checkpoint_sha256="9" * 64),
        row("prediction-final", "prediction_finalize", None, "final/prediction/manifest", [prediction_id],
            finalist_id="M0"),
        row("metric", "metric_join", "label_escrow", "final/metric", ["prediction-final"]),
        row("aggregate", "final_aggregate", None, "final/aggregate", ["metric"]),
    )
    registry = build_final_task_registry(
        population_sha256="a" * 64, campaign_spec_sha256="b" * 64,
        finalist_lock_sha256="c" * 64, submission_ledger_sha256="d" * 64,
        tasks=tasks,
    )
    claim = with_content_hash({
        "contract": FINAL_EXECUTION_CLAIM_CONTRACT, "schema_version": 1,
        "population_sha256": "a" * 64, "reservation_sha256": "e" * 64,
        "campaign_spec_sha256": "b" * 64, "finalist_lock_sha256": "c" * 64,
        "task_registry_sha256": registry["content_hash"],
        "submission_ledger_sha256": "d" * 64,
        "legacy_cancellation_sha256": None,
        "source_commit": "f" * 40,
        "claim_owner_id": derive_claim_owner_id(
            campaign_spec_sha256="b" * 64,
            task_registry_sha256=registry["content_hash"],
            finalist_lock_sha256="c" * 64,
        ),
        "same_owner_recovery_allowed": True,
    })
    capability = issue_role_capability(
        claim=claim, task_registry=registry, task_id=task_id,
        execution_lock_sha256=execution,
    )
    return registry, claim, capability


def _selection_inputs():
    identities = [canonical_sha256({
        "source_file_sha256": "8" * 64, "source_entry": index,
    }) for index in range(30)]
    labels = np.repeat(np.arange(15, dtype=np.uint8), 2)
    records = [
        {
            "identity_digest": digest, "source_path": "source.root",
            "source_file_sha256": "8" * 64, "source_entry": index,
        }
        for index, digest in enumerate(identities)
    ]
    return identities, labels, records


def test_frozen_branch_allowlists_keep_every_prediction_path_label_free() -> None:
    assert set(SELECTION_BRANCHES) & set(LABEL_BRANCHES)
    for path, branches in (
        ("hlt", HLT_FINAL_BRANCHES),
        ("shell_exact", SHELL_EXACT_FINAL_BRANCHES),
        ("native_offline", NATIVE_OFFLINE_FINAL_BRANCHES),
        ("assignment", ASSIGNMENT_FINAL_BRANCHES),
    ):
        assert not set(branches) & set(LABEL_BRANCHES)
        assert validate_projected_branches(path=path, branches=branches) == tuple(sorted(branches))
        with pytest.raises(PermissionError, match="allow-list"):
            validate_projected_branches(path=path, branches=tuple(branches) + ("scoutfj_label",))
        with pytest.raises(PermissionError, match="allow-list"):
            validate_projected_branches(path=path, branches=tuple(branches)[1:])


def test_selection_manifest_and_sealed_uint8_escrow_are_separate(tmp_path) -> None:
    registry, claim, capability = _claim_registry(
        task_id="selection", kind="row_selection", branch_family="selection", execution=None,
    )
    identities, labels, records = _selection_inputs()
    manifest, arrays = class_stratified_selection(
        identities=identities, labels=labels, rows_per_class=[1] * 15,
        population_sha256="a" * 64, selection_rule_sha256="2" * 64,
        capability=capability, execution_claim=claim, task_registry=registry,
        task_id="selection", identity_records=records,
        selection_ranks=list(reversed(range(30))),
        expected_population_identity_digests=identities,
    )
    assert manifest["row_count"] == 15
    assert manifest["labels_sealed_separately"] is True
    assert "labels" not in manifest
    assert arrays["identity_digests"].dtype == np.uint8
    assert arrays["identity_digests"].shape == (15, 32)
    assert arrays["labels"].dtype == np.uint8
    with pytest.raises(ValueError, match="exact registered final population"):
        class_stratified_selection(
            identities=identities, labels=labels, rows_per_class=[1] * 15,
            population_sha256="a" * 64, selection_rule_sha256="2" * 64,
            capability=capability, execution_claim=claim, task_registry=registry,
            task_id="selection", identity_records=records,
            selection_ranks=list(reversed(range(30))),
            expected_population_identity_digests=identities[:-1],
        )
    with pytest.raises(ValueError, match="selection quotas differ"):
        class_stratified_selection(
            identities=identities, labels=labels,
            rows_per_class=[0, *([1] * 14)],
            population_sha256="a" * 64, selection_rule_sha256="2" * 64,
            capability=capability, execution_claim=claim, task_registry=registry,
            task_id="selection", identity_records=records,
            selection_ranks=list(reversed(range(30))),
            expected_population_identity_digests=identities,
        )
    payload = label_escrow_sidecar(
        arrays=arrays, selection_sha256=manifest["content_hash"],
        population_sha256="a" * 64, capability_sha256=capability["content_hash"],
    )
    assert payload["contains_model_inputs"] is False
    assert payload["contains_model_outputs"] is False
    envelope = publish_label_escrow(
        tmp_path, arrays=arrays, selection_sha256=manifest["content_hash"],
        population_sha256="a" * 64, capability_sha256=capability["content_hash"],
        producer_task_id="selection", registered_output_row={"path": "final/labels"},
        campaign_or_recovery_owner={"campaign": "b" * 64},
    )
    sidecar, loaded = load_label_escrow(
        tmp_path, envelope.envelope_id, selection_sha256=manifest["content_hash"],
        population_sha256="a" * 64, capability_sha256=capability["content_hash"],
    )
    assert sidecar["content_hash"] == envelope.sidecar["content_hash"]
    assert np.array_equal(loaded["labels"], arrays["labels"])


@pytest.mark.parametrize(
    ("path", "iterator"),
    (
        ("hlt", iterate_final_hlt_inputs),
        ("shell_exact", iterate_final_shell_exact_inputs),
        ("native_offline", iterate_final_native_offline_inputs),
    ),
)
def test_instrumented_prediction_reader_proves_exact_allowlist_before_io(path, iterator) -> None:
    lock = "6" * 64
    registry, claim, capability = _claim_registry(
        task_id=f"predict:{path}", kind="prediction_shard",
        branch_family=path, execution=lock,
    )
    identities = (_digest("a"), _digest("b")); calls = []

    def reader(*, branches):
        calls.append(tuple(branches))
        assert set(branches).isdisjoint(LABEL_BRANCHES)
        for identity in identities:
            yield {"identity_digest": identity, "model_inputs": object()}

    rows = list(iterator(
        reader=reader, population_sha256="a" * 64,
        task_id=f"predict:{path}", capability=capability,
        execution_claim=claim, task_registry=registry,
        execution_lock_sha256=lock, selected_identities=identities, reader_kwargs={},
    ))
    assert [row.identity_digest for row in rows] == list(identities)
    assert calls == [tuple(sorted({
        "hlt": HLT_FINAL_BRANCHES,
        "shell_exact": SHELL_EXACT_FINAL_BRANCHES,
        "native_offline": NATIVE_OFFLINE_FINAL_BRANCHES,
    }[path]))]


def test_label_output_and_wrong_execution_capability_fail_closed() -> None:
    lock = "6" * 64
    registry, claim, capability = _claim_registry(
        task_id="predict:hlt", kind="prediction_shard", branch_family="hlt", execution=lock,
    )

    def reader(*, branches):
        assert "scoutfj_label" not in branches
        yield {"identity_digest": _digest("a"), "model_inputs": object(), "label": 1}

    with pytest.raises(PermissionError, match="emitted label"):
        list(iterate_final_hlt_inputs(
            reader=reader, population_sha256="a" * 64, task_id="predict:hlt",
            capability=capability, execution_claim=claim, task_registry=registry,
            execution_lock_sha256=lock,
            selected_identities=(_digest("a"),), reader_kwargs={},
        ))
    with pytest.raises(PermissionError, match="execution lock"):
        list(iterate_final_hlt_inputs(
            reader=reader, population_sha256="a" * 64, task_id="predict:hlt",
            capability=capability, execution_claim=claim, task_registry=registry,
            execution_lock_sha256="7" * 64,
            selected_identities=(_digest("a"),), reader_kwargs={},
        ))


def test_low_level_reader_rejects_prediction_capability_for_labels_before_io(tmp_path) -> None:
    lock = "6" * 64
    registry, claim, capability = _claim_registry(
        task_id="predict:hlt", kind="prediction_shard", branch_family="hlt", execution=lock,
    )
    with pytest.raises(PermissionError, match="allow-list"):
        list(iterate_projected_chunks(
            (tmp_path / "never-opened.root",), LABEL_BRANCHES,
            data_root=tmp_path, role="final_test",
            shared_final_capability=capability,
            shared_final_claim=claim,
            shared_final_task_registry=registry,
            final_population_sha256="a" * 64,
            final_task_id="predict:hlt",
            final_branch_family="hlt",
            final_execution_lock_sha256=lock,
        ))


def test_branch_access_record_binds_source_range_and_capability() -> None:
    record = build_branch_access_record(
        path="hlt", capability_sha256="1" * 64, branches=HLT_FINAL_BRANCHES,
        source_rows=({
            "source_path": "source.root", "source_file_sha256": "2" * 64,
            "tree": TREE_NAME, "entry_start": 10, "entry_stop": 20,
        },),
        population_sha256="3" * 64, task_id="predict:hlt",
        execution_lock_sha256="4" * 64,
    )
    assert record["label_free"] is True
    assert record["projected_branches"] == sorted(HLT_FINAL_BRANCHES)
    with pytest.raises(ValueError, match="range"):
        build_branch_access_record(
            path="hlt", capability_sha256="1" * 64, branches=HLT_FINAL_BRANCHES,
            source_rows=({
                "source_path": "source.root", "source_file_sha256": "2" * 64,
                "tree": TREE_NAME, "entry_start": 20, "entry_stop": 20,
            },), population_sha256="3" * 64, task_id="predict:hlt",
            execution_lock_sha256="4" * 64,
        )
