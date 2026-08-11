"""Sole final label reader and concrete label-free model-input streams."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Final

import numpy as np

from hlt_classification.data.cache_contracts import (
    canonical_sha256, deterministic_npz_bytes,
    load_npz_arrays, require_sha256, sha256_file, validate_content_hash,
    with_content_hash,
)
from .hcwdl_representation_contracts import SHARED_FINAL_BRANCH_ACCESS_CONTRACT
from .hcwdl_representation_artifacts import (
    CommittedBinaryEnvelope, publish_binary_envelope, validate_binary_envelope,
)
from .hcwdl_representation_contracts import logical_array_sha256
from .hcwdl_shared_final import validate_role_capability
from .highcov_cache import DenseAssignmentStore
from .identity import normalize_source_path
from .inputs import NativeOfflineInputs, ParticleInputs, build_hlt_inputs, build_native_offline_inputs
from .repair import build_alpha_repaired_inputs, combined_offline_p4, full_endpoint_required_branches
from .schema import (
    BASELINE_BRANCHES, LABEL_BRANCHES, TREE_NAME, hlt_required_branches,
    matching_required_branches, native_offline_required_branches,
)
from .splits import role_records
from .streaming import iterate_projected_chunks


FINAL_ROW_SELECTION_CONTRACT: Final = "HCWDL_SHARED_FINAL_ROW_SELECTION/v1"
FINAL_LABEL_ESCROW_CONTRACT: Final = "HCWDL_SHARED_FINAL_LABEL_ESCROW/v1"
BRANCH_ACCESS_CONTRACT: Final = SHARED_FINAL_BRANCH_ACCESS_CONTRACT
SELECTION_BRANCHES: Final = frozenset(BASELINE_BRANCHES) | frozenset(LABEL_BRANCHES)
HLT_FINAL_BRANCHES: Final = hlt_required_branches()
SHELL_EXACT_FINAL_BRANCHES: Final = hlt_required_branches() | full_endpoint_required_branches()
NATIVE_OFFLINE_FINAL_BRANCHES: Final = native_offline_required_branches()
ASSIGNMENT_FINAL_BRANCHES: Final = matching_required_branches()
LABEL_SET: Final = frozenset(LABEL_BRANCHES)
_ALLOWED = {
    "selection": SELECTION_BRANCHES, "assignment": ASSIGNMENT_FINAL_BRANCHES,
    "hlt": HLT_FINAL_BRANCHES, "shell_exact": SHELL_EXACT_FINAL_BRANCHES,
    "native_offline": NATIVE_OFFLINE_FINAL_BRANCHES,
}
if any(branches & LABEL_SET for name, branches in _ALLOWED.items() if name != "selection"):
    raise RuntimeError("label-free final allow-list contains a label branch")


def feature_identity_streamer_sha256() -> str:
    """Hash the exact source surfaces that build final model inputs/identities."""

    root = Path(__file__).resolve().parent
    names = (
        "hcwdl_final_stream.py", "dataset.py", "pmard_stream.py",
        "inputs.py", "identity.py", "repair.py", "streaming.py",
    )
    return canonical_sha256({
        "contract": "HCWDL_SHARED_FINAL_FEATURE_IDENTITY_STREAMER/v1",
        "source_files": {name: sha256_file(root / name) for name in names},
        "branch_allowlists": {
            name: sorted(branches) for name, branches in sorted(_ALLOWED.items())
        },
    })


@dataclass(frozen=True)
class FinalInputRow:
    identity_digest: str
    model_inputs: ParticleInputs | NativeOfflineInputs


def validate_projected_branches(*, path: str, branches: Iterable[str]) -> tuple[str, ...]:
    requested = tuple(sorted(set(str(value) for value in branches)))
    allowed = _ALLOWED.get(path)
    if allowed is None:
        raise ValueError("unknown final stream path")
    if set(requested) != set(allowed):
        raise PermissionError("final projected branches differ from frozen allow-list")
    if path != "selection" and set(requested) & LABEL_SET:
        raise PermissionError("label branch requested before final I/O")
    return requested


def build_branch_access_record(
    *, path: str, capability_sha256: str, branches: Iterable[str],
    source_rows: Sequence[Mapping[str, Any]], population_sha256: str,
    task_id: str, execution_lock_sha256: str | None,
) -> dict[str, Any]:
    requested = validate_projected_branches(path=path, branches=branches)
    rows = []
    for value in source_rows:
        start, stop = int(value.get("entry_start", -1)), int(value.get("entry_stop", -1))
        if value.get("tree") != TREE_NAME or start < 0 or stop <= start:
            raise ValueError("branch access range differs")
        rows.append({
            "source_path": normalize_source_path(str(value["source_path"])),
            "source_file_sha256": require_sha256(value["source_file_sha256"], name="source"),
            "tree": TREE_NAME, "entry_start": start, "entry_stop": stop,
        })
    return with_content_hash({
        "contract": BRANCH_ACCESS_CONTRACT, "schema_version": 1,
        "path": path, "population_sha256": require_sha256(population_sha256, name="population"),
        "task_id": str(task_id),
        "capability_sha256": require_sha256(capability_sha256, name="capability"),
        "execution_lock_sha256": None if execution_lock_sha256 is None else require_sha256(execution_lock_sha256, name="execution lock"),
        "projected_branches": list(requested), "sources": rows,
        "label_free": path != "selection",
    })


def _selection_record(value: Mapping[str, Any], digest: str) -> dict[str, Any]:
    if set(value) != {"identity_digest", "source_path", "source_file_sha256", "source_entry"}:
        raise ValueError("selection identity record differs")
    entry = value["source_entry"]
    if isinstance(entry, bool) or not isinstance(entry, int) or entry < 0:
        raise ValueError("selection source entry differs")
    if value["identity_digest"] != digest:
        raise ValueError("selection identity digest differs")
    result = {
        "identity_digest": require_sha256(digest, name="identity"),
        "source_path": normalize_source_path(str(value["source_path"])),
        "source_file_sha256": require_sha256(value["source_file_sha256"], name="source"),
        "source_entry": int(entry),
    }
    if canonical_sha256({
        "source_file_sha256": result["source_file_sha256"],
        "source_entry": result["source_entry"],
    }) != result["identity_digest"]:
        raise ValueError("selection identity is not derived from its canonical source identity")
    return result


def class_stratified_selection(
    *, identities: Sequence[str], labels: np.ndarray, rows_per_class: Sequence[int],
    population_sha256: str, selection_rule_sha256: str,
    capability: Mapping[str, Any], execution_claim: Mapping[str, Any],
    task_registry: Mapping[str, Any], task_id: str,
    identity_records: Sequence[Mapping[str, Any]], selection_ranks: Sequence[int],
    expected_population_identity_digests: Sequence[str],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Select by frozen ranks; publish labels only via the separate escrow."""
    validate_role_capability(
        capability, execution_claim=execution_claim,
        task_registry=task_registry,
        expected_population_sha256=population_sha256,
        expected_task_id=task_id, allowed_kinds=("row_selection",),
        expected_execution_lock_sha256=None, expected_branch_family="selection",
    )
    keys = np.asarray([str(value) for value in identities]); target = np.asarray(labels)
    if (
        len(rows_per_class) != 15
        or any(isinstance(value, bool) or not isinstance(value, Integral) for value in rows_per_class)
    ):
        raise ValueError("selection quotas differ")
    quotas = tuple(int(value) for value in rows_per_class)
    ranks = tuple(int(value) for value in selection_ranks)
    if keys.ndim != 1 or target.shape != keys.shape or target.dtype.kind not in "iu":
        raise ValueError("selection identities/labels differ")
    if len(identity_records) != len(keys) or len(ranks) != len(keys):
        raise ValueError("selection metadata differs")
    if len(set(keys.tolist())) != len(keys) or any(rank < 0 for rank in ranks):
        raise ValueError("selection identity/rank differs")
    population_keys = tuple(
        require_sha256(value, name="population identity")
        for value in expected_population_identity_digests
    )
    if (
        len(population_keys) != len(set(population_keys))
        or len(keys) != len(population_keys)
        or set(keys.tolist()) != set(population_keys)
    ):
        raise ValueError("selection scan does not cover the exact registered final population")
    if any(value <= 0 for value in quotas):
        raise ValueError("selection quotas differ")
    target = target.astype(np.int64, copy=False)
    if np.any((target < 0) | (target >= 15)):
        raise ValueError("selection label lies outside 0..14")
    records = [_selection_record(row, keys[i]) for i, row in enumerate(identity_records)]
    selected = []
    for class_index, quota in enumerate(quotas):
        candidates = np.flatnonzero(target == class_index).tolist()
        candidates.sort(key=lambda index: (ranks[index], keys[index]))
        if len(candidates) < quota:
            raise ValueError("selection class lacks quota")
        selected.extend(candidates[:quota])
    selected.sort(key=lambda index: (records[index]["source_path"], records[index]["source_entry"]))
    width = max(64, max((len(value) for value in keys.tolist()), default=64))
    selected_keys = keys[selected].astype(f"<U{width}")
    try:
        selected_key_bytes = np.asarray(
            [np.frombuffer(bytes.fromhex(value), dtype=np.uint8) for value in selected_keys],
            dtype=np.uint8,
        )
    except ValueError as error:
        raise ValueError("selected identity is not a canonical SHA-256 digest") from error
    if selected_key_bytes.shape != (len(selected_keys), 32):
        raise ValueError("selected identity digest width differs")
    selected_labels = target[selected].astype(np.uint8)
    manifest = with_content_hash({
        "contract": FINAL_ROW_SELECTION_CONTRACT, "schema_version": 1,
        "population_sha256": require_sha256(population_sha256, name="population"),
        "selection_rule_sha256": require_sha256(selection_rule_sha256, name="selection rule"),
        "capability_sha256": capability["content_hash"], "row_count": len(selected),
        "class_counts": np.bincount(selected_labels, minlength=15).tolist(),
        "identity_order_sha256": canonical_sha256(selected_keys.tolist()),
        "identity_digests": selected_keys.tolist(),
        "selected_rows": [records[index] for index in selected],
        "selection_rank_sha256": canonical_sha256([f"{ranks[index]:032x}" for index in selected]),
        "labels_sealed_separately": True, "particle_branches_read": False,
    })
    return manifest, {"identity_digests": selected_key_bytes, "labels": selected_labels}


def label_escrow_sidecar(
    *, arrays: Mapping[str, np.ndarray], selection_sha256: str,
    population_sha256: str, capability_sha256: str,
) -> dict[str, Any]:
    identities = np.asarray(arrays.get("identity_digests")); labels = np.asarray(arrays.get("labels"))
    if identities.dtype != np.uint8 or identities.ndim != 2 or identities.shape[1:] != (32,):
        raise ValueError("escrow identities must be uint8 [rows,32]")
    if labels.shape != (len(identities),) or labels.dtype != np.uint8 or np.any(labels >= 15):
        raise ValueError("escrow labels must be uint8 0..14")
    identity_hex = [bytes(row).hex() for row in identities]
    if len(set(identity_hex)) != len(identities):
        raise ValueError("escrow repeats identity")
    return {
        "selection_sha256": require_sha256(selection_sha256, name="selection"),
        "population_sha256": require_sha256(population_sha256, name="population"),
        "capability_sha256": require_sha256(capability_sha256, name="capability"),
        "rows": len(labels), "identity_order_sha256": canonical_sha256(identity_hex),
        "array_sha256": {
            name: logical_array_sha256(name, value) for name, value in sorted(arrays.items())
        },
        "contains_model_inputs": False, "contains_model_outputs": False,
        "label_dtype": "uint8", "class_count": 15,
    }


def publish_label_escrow(
    root: str | Path, *, arrays: Mapping[str, np.ndarray], selection_sha256: str,
    population_sha256: str, capability_sha256: str, producer_task_id: str,
    registered_output_row: Mapping[str, Any], campaign_or_recovery_owner: Mapping[str, Any],
    failure_hook: Callable[[str], None] | None = None,
) -> CommittedBinaryEnvelope:
    values = {name: np.asarray(value) for name, value in arrays.items()}
    payload = label_escrow_sidecar(
        arrays=values, selection_sha256=selection_sha256,
        population_sha256=population_sha256, capability_sha256=capability_sha256,
    )
    return publish_binary_envelope(
        root, artifact_contract=FINAL_LABEL_ESCROW_CONTRACT,
        producer_task_id=producer_task_id,
        schema={"arrays": ["identity_digests", "labels"], "identity_dtype": "uint8", "identity_width": 32, "label_dtype": "uint8"},
        immutable_parent_hashes={"selection": selection_sha256, "population": population_sha256, "capability": capability_sha256},
        registered_output_row=registered_output_row,
        campaign_or_recovery_owner=campaign_or_recovery_owner,
        payloads={"labels.npz": deterministic_npz_bytes(values)},
        member_metadata={"labels.npz": {
            "logical_sha256": canonical_sha256(payload["array_sha256"]),
            "dtype": "npz", "shape": [len(values["labels"])],
        }}, sidecar_payload=payload, failure_hook=failure_hook,
    )


def load_label_escrow(
    root: str | Path, envelope_id: str, *, selection_sha256: str,
    population_sha256: str, capability_sha256: str,
) -> tuple[Mapping[str, Any], dict[str, np.ndarray]]:
    envelope = validate_binary_envelope(
        root, envelope_id, expected_contract=FINAL_LABEL_ESCROW_CONTRACT,
        expected_parents={"selection": selection_sha256, "population": population_sha256, "capability": capability_sha256},
    )
    arrays = load_npz_arrays(envelope.directory / "labels.npz")
    expected = label_escrow_sidecar(
        arrays=arrays, selection_sha256=selection_sha256,
        population_sha256=population_sha256, capability_sha256=capability_sha256,
    )
    if any(envelope.sidecar["payload"].get(key) != value for key, value in expected.items()):
        raise ValueError("escrow arrays differ from sidecar")
    return envelope.sidecar, arrays


def _slice_arrays(arrays: Mapping[str, object], indexes: np.ndarray) -> dict[str, object]:
    return {name: value[indexes] for name, value in arrays.items()}


def _slice_particle(value: ParticleInputs, index: int) -> ParticleInputs:
    return ParticleInputs(value.features[index:index+1], value.vectors[index:index+1], value.mask[index:index+1], value.raw_lengths[index:index+1])


def _slice_native(value: NativeOfflineInputs, index: int) -> NativeOfflineInputs:
    return NativeOfflineInputs(_slice_particle(value.charged, index), _slice_particle(value.neutral, index))


def _iterate_concrete(
    *, path: str, split_manifest: Mapping[str, Any], data_root: str | Path,
    population_sha256: str, task_id: str, capability: Mapping[str, Any],
    execution_claim: Mapping[str, Any], task_registry: Mapping[str, Any],
    execution_lock_sha256: str, selection: Mapping[str, Any], source_partition: str,
    assignment_store: DenseAssignmentStore | None = None, step_size: int = 4096,
    branch_access_collector: list[dict[str, Any]] | None = None,
) -> Iterator[FinalInputRow]:
    validate_role_capability(
        capability, execution_claim=execution_claim,
        task_registry=task_registry,
        expected_population_sha256=population_sha256,
        expected_task_id=task_id, allowed_kinds=("prediction_shard",),
        expected_execution_lock_sha256=execution_lock_sha256,
        expected_branch_family=path,
    )
    if capability["task"].get("source_partition") != source_partition:
        raise PermissionError("prediction capability source differs")
    selection_hash = validate_content_hash(
        selection, expected_contract=FINAL_ROW_SELECTION_CONTRACT,
        expected_schema_version=1,
    )
    if selection.get("population_sha256") != population_sha256:
        raise ValueError("prediction selection population differs")
    source_partition = normalize_source_path(source_partition)
    records = {record.path: record for record in role_records(split_manifest, "final_test")}
    if source_partition not in records:
        raise ValueError("prediction source is outside final split")
    source_record = records[source_partition]
    resource_signature = capability["task"].get("resource_signature")
    if not isinstance(resource_signature, Mapping) or (
        resource_signature.get("row_selection_sha256") != selection_hash
        or resource_signature.get("source_file_sha256") != source_record.sha256
    ):
        raise PermissionError("prediction task data lineage differs")
    rows = [row for row in selection.get("selected_rows", ()) if row["source_path"] == source_partition]
    rows.sort(key=lambda row: int(row["source_entry"]))
    if not rows or any(row["source_file_sha256"] != source_record.sha256 for row in rows):
        raise ValueError("selected prediction source/hash differs")
    wanted = {int(row["source_entry"]): row for row in rows}
    if len(wanted) != len(rows):
        raise ValueError("selected source entries repeat")
    branches = validate_projected_branches(path=path, branches=_ALLOWED[path])
    emitted, accesses = [], []
    for chunk in iterate_projected_chunks(
        (Path(data_root) / source_partition,), branches, data_root=data_root,
        role="final_test", shared_final_capability=capability,
        shared_final_claim=execution_claim,
        shared_final_task_registry=task_registry,
        final_population_sha256=population_sha256, final_task_id=task_id,
        final_branch_family=path,
        final_execution_lock_sha256=execution_lock_sha256,
        shared_reservation_active=True, step_size=step_size,
    ):
        absolute = np.asarray(sorted(entry for entry in wanted if chunk.entry_start <= entry < chunk.entry_stop), np.int64)
        if not len(absolute):
            continue
        indexes = absolute - chunk.entry_start; arrays = _slice_arrays(chunk.arrays, indexes)
        if any(name in arrays for name in LABEL_BRANCHES):
            raise PermissionError("label-free reader received labels")
        if path == "hlt":
            view: ParticleInputs | NativeOfflineInputs = build_hlt_inputs(arrays)
        elif path == "native_offline":
            view = build_native_offline_inputs(arrays)
        else:
            if assignment_store is None:
                raise ValueError("Shell-Exact prediction requires assignments")
            assignment, confidence = assignment_store.join(source_partition, absolute)
            offline_p4 = [combined_offline_p4(arrays, arrays, row) for row in range(len(absolute))]
            keys = [f"{source_partition}::{TREE_NAME}::{int(entry)}" for entry in absolute]
            view = build_alpha_repaired_inputs(
                arrays, offline_p4, assignment, alpha=1.0,
                repair_family="HIGHCOV_SHELL_EXACT/v1", confidence_weights=confidence,
                offline_arrays=arrays, identity_keys=keys, discrete_seed=1337,
            )
        accesses.append({"source_path": source_partition, "source_file_sha256": source_record.sha256, "tree": TREE_NAME, "entry_start": chunk.entry_start, "entry_stop": chunk.entry_stop})
        for local, entry in enumerate(absolute):
            digest = wanted[int(entry)]["identity_digest"]; emitted.append(digest)
            model_inputs = _slice_native(view, local) if isinstance(view, NativeOfflineInputs) else _slice_particle(view, local)
            yield FinalInputRow(digest, model_inputs)
    expected = [row["identity_digest"] for row in rows]
    if emitted != expected:
        raise ValueError("label-free reader coverage/order differs")
    if branch_access_collector is not None:
        branch_access_collector.append(build_branch_access_record(
            path=path, capability_sha256=capability["content_hash"], branches=branches,
            source_rows=accesses, population_sha256=population_sha256,
            task_id=task_id, execution_lock_sha256=execution_lock_sha256,
        ))


def _iterate_instrumented(
    *, path: str, reader: Callable[..., Iterator[Mapping[str, Any]]],
    population_sha256: str, task_id: str, capability: Mapping[str, Any],
    execution_claim: Mapping[str, Any], task_registry: Mapping[str, Any],
    execution_lock_sha256: str, selected_identities: Sequence[str],
    reader_kwargs: Mapping[str, Any],
) -> Iterator[FinalInputRow]:
    validate_role_capability(
        capability, execution_claim=execution_claim,
        task_registry=task_registry,
        expected_population_sha256=population_sha256,
        expected_task_id=task_id, allowed_kinds=("prediction_shard",),
        expected_execution_lock_sha256=execution_lock_sha256,
        expected_branch_family=path,
    )
    branches = validate_projected_branches(path=path, branches=_ALLOWED[path])
    expected = tuple(str(value) for value in selected_identities); emitted = []
    if len(expected) != len(set(expected)):
        raise ValueError("instrumented selection repeats identities")
    for row in reader(branches=branches, **dict(reader_kwargs)):
        if "label" in row or any(name in row for name in LABEL_BRANCHES):
            raise PermissionError("label-free reader emitted label")
        identity = str(row.get("identity_digest", ""))
        if identity not in set(expected):
            raise ValueError("reader emitted unexpected identity")
        emitted.append(identity); yield FinalInputRow(identity, row["model_inputs"])
    if tuple(emitted) != expected:
        raise ValueError("instrumented reader coverage/order differs")


def _dispatch(path: str, **kwargs: Any) -> Iterator[FinalInputRow]:
    reader = kwargs.pop("reader", None)
    return _iterate_instrumented(path=path, reader=reader, **kwargs) if reader is not None else _iterate_concrete(path=path, **kwargs)


def iterate_final_hlt_inputs(**kwargs: Any) -> Iterator[FinalInputRow]:
    return _dispatch("hlt", **kwargs)


def iterate_final_shell_exact_inputs(**kwargs: Any) -> Iterator[FinalInputRow]:
    return _dispatch("shell_exact", **kwargs)


def iterate_final_native_offline_inputs(**kwargs: Any) -> Iterator[FinalInputRow]:
    return _dispatch("native_offline", **kwargs)


__all__ = [
    "ASSIGNMENT_FINAL_BRANCHES", "BRANCH_ACCESS_CONTRACT", "FINAL_LABEL_ESCROW_CONTRACT",
    "FINAL_ROW_SELECTION_CONTRACT", "HLT_FINAL_BRANCHES", "NATIVE_OFFLINE_FINAL_BRANCHES",
    "SELECTION_BRANCHES", "SHELL_EXACT_FINAL_BRANCHES", "FinalInputRow",
    "build_branch_access_record", "class_stratified_selection", "iterate_final_hlt_inputs",
    "iterate_final_native_offline_inputs", "iterate_final_shell_exact_inputs",
    "label_escrow_sidecar", "load_label_escrow", "publish_label_escrow",
    "validate_projected_branches",
]
