"""Immutable residual-shell shards, manifests, audits, and lazy lookup."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping, Sequence
import hashlib

import numpy as np

from hlt_classification.data.cache_contracts import (
    array_sha256, atomic_publish_bytes, deterministic_npz_bytes, load_json,
    load_npz_arrays, require_sha256, sha256_file, validate_content_hash,
    with_content_hash, write_immutable_json,
)

from .hcwdl_homotopy_contracts import (
    BASE_MANIFEST_CONTRACT, BASE_SHARD_CONTRACT, COUPLING_AUDIT_CONTRACT,
    COUPLING_LOCK_CONTRACT, COUPLING_MANIFEST_CONTRACT,
    EDIT_INSERTION, EDIT_REMOVAL, EDIT_SUBSTITUTION, SWITCH_SIDECAR_CONTRACT,
)
from .hcwdl_upper_coupling import ResidualEdit


BASE_ARRAYS: Final = (
    "entries", "row_offsets", "edit_kind", "source_native_offline_index",
    "target_hlt_slot", "target_kind", "target_native_offline_index",
    "cost_q", "mass_q",
)


def _logical_hashes(arrays: Mapping[str, np.ndarray]) -> dict[str, str]:
    return {name: array_sha256(name, arrays[name]) for name in sorted(arrays)}


def _base_arrays(
    entries: Sequence[int], edit_rows: Sequence[Sequence[ResidualEdit]],
) -> dict[str, np.ndarray]:
    if len(entries) != len(edit_rows):
        raise ValueError("coupling base entries/edit rows differ")
    entry_array = np.asarray(entries, dtype="<i8")
    if len(entry_array) and (
        np.any(entry_array < 0) or np.any(entry_array[1:] <= entry_array[:-1])
    ):
        raise ValueError("coupling base entries must be strictly increasing")
    offsets = [0]
    flat: list[ResidualEdit] = []
    for row in edit_rows:
        ordered = sorted(row, key=lambda edit: edit.key)
        if list(row) != ordered or any(edit.mass_q <= 0 for edit in row):
            raise ValueError("coupling base edit order/mass differs")
        flat.extend(row); offsets.append(len(flat))
    arrays = {
        "entries": entry_array,
        "row_offsets": np.asarray(offsets, dtype="<u8"),
        "edit_kind": np.asarray([row.edit_kind for row in flat], dtype="u1"),
        "source_native_offline_index": np.asarray(
            [row.source_native_index for row in flat], dtype="<i4",
        ),
        "target_hlt_slot": np.asarray(
            [row.target_hlt_slot for row in flat], dtype="<u2",
        ),
        "target_kind": np.asarray([row.target_kind for row in flat], dtype="u1"),
        "target_native_offline_index": np.asarray(
            [row.target_native_index for row in flat], dtype="<i4",
        ),
        "cost_q": np.asarray([row.cost_q for row in flat], dtype="<u4"),
        "mass_q": np.asarray([row.mass_q for row in flat], dtype="<u4"),
    }
    validate_base_arrays(arrays)
    return arrays


def validate_base_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    if set(arrays) != set(BASE_ARRAYS):
        raise ValueError("coupling base array names differ")
    expected_dtype = {
        "entries": "<i8", "row_offsets": "<u8", "edit_kind": "|u1",
        "source_native_offline_index": "<i4", "target_hlt_slot": "<u2",
        "target_kind": "|u1", "target_native_offline_index": "<i4",
        "cost_q": "<u4", "mass_q": "<u4",
    }
    for name, dtype in expected_dtype.items():
        value = np.asarray(arrays[name])
        if value.ndim != 1 or value.dtype.str != dtype:
            raise ValueError(f"coupling base array {name} dtype/shape differs")
    entries = arrays["entries"]; offsets = arrays["row_offsets"]
    edits = len(arrays["edit_kind"])
    if len(offsets) != len(entries) + 1 or not len(offsets) or offsets[0] != 0 or offsets[-1] != edits:
        raise ValueError("coupling base offsets differ")
    if np.any(offsets[1:] < offsets[:-1]) or (len(entries) and np.any(entries[1:] <= entries[:-1])):
        raise ValueError("coupling base row order differs")
    if any(len(arrays[name]) != edits for name in BASE_ARRAYS[2:]):
        raise ValueError("coupling base edit array lengths differ")
    kinds = arrays["edit_kind"]
    if np.any(~np.isin(kinds, (EDIT_SUBSTITUTION, EDIT_REMOVAL, EDIT_INSERTION))):
        raise ValueError("coupling base edit kind differs")
    if np.any(arrays["cost_q"] > 1_000_000) or np.any(arrays["mass_q"] == 0):
        raise ValueError("coupling base cost/mass quantum differs")
    source = arrays["source_native_offline_index"]
    target = arrays["target_hlt_slot"]
    if np.any((kinds == EDIT_INSERTION) & (source != -1)):
        raise ValueError("coupling insertion has a real source")
    if np.any((kinds == EDIT_REMOVAL) & (target != 65535)):
        raise ValueError("coupling removal has a real target")


def publish_base_shard(
    base_path: str | Path, *, role: str, source_path: str,
    entries: Sequence[int], edit_rows: Sequence[Sequence[ResidualEdit]],
    parents: Mapping[str, str], producer_commit: str,
) -> tuple[Path, Path]:
    if role not in {"train", "validation"}:
        raise PermissionError("HCWDL-UJ coupling supports train/validation only")
    if any(len(str(value)) != 64 for value in parents.values()):
        raise ValueError("coupling base parent hash differs")
    arrays = _base_arrays(entries, edit_rows)
    base = Path(base_path)
    npz_path, json_path = base.with_suffix(".npz"), base.with_suffix(".json")
    data = deterministic_npz_bytes(arrays)
    atomic_publish_bytes(npz_path, data)
    metadata = with_content_hash({
        "contract": BASE_SHARD_CONTRACT,
        "schema_version": 1,
        "role": role,
        "source_path": source_path,
        "rows": len(entries),
        "edits": len(arrays["edit_kind"]),
        "npz_filename": npz_path.name,
        "npz_sha256": sha256_file(npz_path),
        "logical_array_sha256": _logical_hashes(arrays),
        "parents": dict(sorted(parents.items())),
        "producer_commit": producer_commit,
        "labels_read": False,
        "final_test_accessed": False,
    })
    write_immutable_json(json_path, metadata)
    return npz_path, json_path


def load_base_shard(metadata_path: str | Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    path = Path(metadata_path)
    metadata = load_json(path)
    validate_content_hash(metadata, expected_contract=BASE_SHARD_CONTRACT, expected_schema_version=1)
    if metadata.get("role") not in {"train", "validation"} or metadata.get("labels_read") is not False or metadata.get("final_test_accessed") is not False:
        raise PermissionError("coupling base role/access differs")
    npz = path.with_name(str(metadata["npz_filename"]))
    if sha256_file(npz) != require_sha256(metadata.get("npz_sha256"), name="coupling NPZ SHA-256"):
        raise ValueError("coupling base NPZ hash differs")
    arrays = load_npz_arrays(npz); validate_base_arrays(arrays)
    if _logical_hashes(arrays) != metadata.get("logical_array_sha256"):
        raise ValueError("coupling base logical arrays differ")
    if len(arrays["entries"]) != metadata.get("rows") or len(arrays["edit_kind"]) != metadata.get("edits"):
        raise ValueError("coupling base row/edit totals differ")
    return metadata, arrays


def publish_base_manifest(
    output: str | Path, *, role: str, shard_metadata_paths: Sequence[str | Path],
    expected_sources: Sequence[str], expected_rows: int,
    parents: Mapping[str, str],
) -> dict[str, Any]:
    if role not in {"train", "validation"} or expected_rows <= 0:
        raise ValueError("coupling base manifest role/rows differ")
    if len(shard_metadata_paths) != len(expected_sources):
        raise ValueError("coupling base manifest source count differs")
    records = []
    rows = edits = 0
    for expected_source, path in zip(expected_sources, shard_metadata_paths, strict=True):
        metadata, arrays = load_base_shard(path)
        if metadata["role"] != role or metadata["source_path"] != expected_source:
            raise ValueError("coupling base manifest shard order/source differs")
        rows += int(metadata["rows"]); edits += int(metadata["edits"])
        records.append({
            "source_path": expected_source,
            "metadata_path": str(Path(path).resolve()),
            "metadata_sha256": metadata["content_hash"],
            "npz_sha256": metadata["npz_sha256"],
            "rows": len(arrays["entries"]), "edits": len(arrays["edit_kind"]),
        })
    if rows != expected_rows:
        raise ValueError("coupling base manifest coverage differs")
    payload = with_content_hash({
        "contract": BASE_MANIFEST_CONTRACT, "schema_version": 1,
        "role": role, "expected_rows": expected_rows, "observed_rows": rows,
        "edits": edits, "shards": records,
        "ordered_base_shard_sha256": [row["metadata_sha256"] for row in records],
        "parents": dict(sorted(parents.items())),
        "complete_source_coverage": True, "labels_read": False,
        "final_test_accessed": False,
    })
    write_immutable_json(output, payload)
    return payload


def validate_base_manifest(value: Mapping[str, Any], *, role: str | None = None) -> str:
    digest = validate_content_hash(value, expected_contract=BASE_MANIFEST_CONTRACT, expected_schema_version=1)
    if value.get("role") not in {"train", "validation"} or (role is not None and value.get("role") != role):
        raise ValueError("coupling base manifest role differs")
    if value.get("complete_source_coverage") is not True or value.get("observed_rows") != value.get("expected_rows"):
        raise ValueError("coupling base manifest is incomplete")
    shards = value.get("shards")
    if not isinstance(shards, list) or value.get("ordered_base_shard_sha256") != [row.get("metadata_sha256") for row in shards]:
        raise ValueError("coupling base manifest shard vector differs")
    return digest


def publish_switch_sidecar(
    output: str | Path, *, base_metadata_path: str | Path,
    switch_u16: np.ndarray, switch_calibration_sha256: str,
) -> tuple[Path, Path]:
    base, arrays = load_base_shard(base_metadata_path)
    switches = np.asarray(switch_u16, dtype="<u2")
    if switches.ndim != 1 or len(switches) != len(arrays["edit_kind"]):
        raise ValueError("coupling switch sidecar shape differs")
    base_path = Path(output)
    npz_path, json_path = base_path.with_suffix(".npz"), base_path.with_suffix(".json")
    payload_arrays = {"switch_u16": switches}
    atomic_publish_bytes(npz_path, deterministic_npz_bytes(payload_arrays))
    metadata = with_content_hash({
        "contract": SWITCH_SIDECAR_CONTRACT, "schema_version": 1,
        "role": base["role"], "source_path": base["source_path"],
        "base_shard_sha256": base["content_hash"],
        "switch_calibration_sha256": require_sha256(
            switch_calibration_sha256, name="switch calibration SHA-256",
        ),
        "edits": len(switches), "npz_filename": npz_path.name,
        "npz_sha256": sha256_file(npz_path),
        "logical_array_sha256": _logical_hashes(payload_arrays),
        "final_test_accessed": False,
    })
    write_immutable_json(json_path, metadata)
    return npz_path, json_path


def load_switch_sidecar(path: str | Path) -> tuple[dict[str, Any], np.ndarray]:
    source = Path(path); metadata = load_json(source)
    validate_content_hash(metadata, expected_contract=SWITCH_SIDECAR_CONTRACT, expected_schema_version=1)
    npz = source.with_name(str(metadata["npz_filename"]))
    if sha256_file(npz) != metadata.get("npz_sha256"):
        raise ValueError("coupling switch NPZ hash differs")
    arrays = load_npz_arrays(npz)
    if set(arrays) != {"switch_u16"} or arrays["switch_u16"].dtype.str != "<u2" or arrays["switch_u16"].ndim != 1:
        raise ValueError("coupling switch array differs")
    if _logical_hashes(arrays) != metadata.get("logical_array_sha256") or len(arrays["switch_u16"]) != metadata.get("edits"):
        raise ValueError("coupling switch logical content differs")
    return metadata, arrays["switch_u16"]


def publish_coupling_manifest(
    output: str | Path, *, role: str, base_manifest_path: str | Path,
    switch_sidecar_paths: Sequence[str | Path], switch_calibration_sha256: str,
) -> dict[str, Any]:
    base = load_json(base_manifest_path); base_hash = validate_base_manifest(base, role=role)
    if len(switch_sidecar_paths) != len(base["shards"]):
        raise ValueError("coupling manifest sidecar count differs")
    pairs = []
    for base_row, sidecar_path in zip(base["shards"], switch_sidecar_paths, strict=True):
        sidecar, switches = load_switch_sidecar(sidecar_path)
        if sidecar.get("base_shard_sha256") != base_row["metadata_sha256"] or sidecar.get("source_path") != base_row["source_path"]:
            raise ValueError("coupling manifest base/sidecar pair differs")
        if sidecar.get("switch_calibration_sha256") != switch_calibration_sha256:
            raise ValueError("coupling manifest switch calibration differs")
        pairs.append({
            "source_path": base_row["source_path"],
            "base_metadata_path": base_row["metadata_path"],
            "base_shard_sha256": base_row["metadata_sha256"],
            "sidecar_path": str(Path(sidecar_path).resolve()),
            "sidecar_sha256": sidecar["content_hash"],
            "rows": base_row["rows"], "edits": len(switches),
        })
    payload = with_content_hash({
        "contract": COUPLING_MANIFEST_CONTRACT, "schema_version": 1,
        "role": role, "base_manifest_sha256": base_hash,
        "switch_calibration_sha256": require_sha256(
            switch_calibration_sha256, name="switch calibration SHA-256",
        ),
        "rows": base["observed_rows"], "edits": base["edits"],
        "pairs": pairs, "complete_source_coverage": True,
        "final_test_accessed": False,
    })
    write_immutable_json(output, payload)
    return payload


def validate_coupling_manifest(value: Mapping[str, Any], *, role: str | None = None) -> str:
    digest = validate_content_hash(value, expected_contract=COUPLING_MANIFEST_CONTRACT, expected_schema_version=1)
    if value.get("role") not in {"train", "validation"} or (role is not None and value.get("role") != role):
        raise ValueError("coupling manifest role differs")
    if value.get("complete_source_coverage") is not True or not isinstance(value.get("pairs"), list):
        raise ValueError("coupling manifest is incomplete")
    if sum(int(row["rows"]) for row in value["pairs"]) != value.get("rows") or sum(int(row["edits"]) for row in value["pairs"]) != value.get("edits"):
        raise ValueError("coupling manifest totals differ")
    return digest


@dataclass(frozen=True)
class CouplingRow:
    edits: tuple[ResidualEdit, ...]


class ResidualCouplingStore:
    """Validated lazy source/entry lookup over base+switch immutable pairs."""

    def __init__(self, manifest_path: str | Path) -> None:
        self.path = Path(manifest_path); self.manifest = load_json(self.path)
        validate_coupling_manifest(self.manifest)
        self._sources = {str(row["source_path"]): row for row in self.manifest["pairs"]}
        self._loaded: dict[str, tuple[dict[str, np.ndarray], np.ndarray, dict[int, int]]] = {}

    def _load(self, source_path: str):
        if source_path not in self._sources:
            raise KeyError(f"coupling source is absent: {source_path}")
        if source_path not in self._loaded:
            row = self._sources[source_path]
            base_meta, arrays = load_base_shard(row["base_metadata_path"])
            sidecar_meta, switches = load_switch_sidecar(row["sidecar_path"])
            if base_meta["content_hash"] != row["base_shard_sha256"] or sidecar_meta["content_hash"] != row["sidecar_sha256"]:
                raise ValueError("coupling store pair hash differs")
            lookup = {int(value): index for index, value in enumerate(arrays["entries"])}
            self._loaded[source_path] = arrays, switches, lookup
        return self._loaded[source_path]

    def get(self, source_path: str, entry: int) -> CouplingRow:
        arrays, switches, lookup = self._load(source_path)
        if int(entry) not in lookup:
            raise KeyError(f"coupling entry is absent: {source_path}::{entry}")
        row = lookup[int(entry)]; start = int(arrays["row_offsets"][row]); stop = int(arrays["row_offsets"][row + 1])
        edits = tuple(ResidualEdit(
            int(arrays["edit_kind"][i]), int(arrays["source_native_offline_index"][i]),
            int(arrays["target_hlt_slot"][i]), int(arrays["target_kind"][i]),
            int(arrays["target_native_offline_index"][i]), int(arrays["cost_q"][i]),
            int(arrays["mass_q"][i]), int(switches[i]),
        ) for i in range(start, stop))
        return CouplingRow(edits)


def build_coupling_audit(
    *, coupling_config_sha256: str, train_manifest_sha256: str,
    validation_manifest_sha256: str, expected_rows: Mapping[str, int],
    observed_rows: Mapping[str, int], counters: Mapping[str, int],
    endpoint_logical_sha256: Mapping[str, str], branch_allowlist_sha256: str,
    branch_access_trace_sha256: str, independent_sample_sha256: str,
) -> dict[str, Any]:
    if dict(expected_rows) != dict(observed_rows):
        raise ValueError("coupling full-role audit coverage differs")
    required_zero = (
        "partition_failures", "assignment_injectivity_failures",
        "endpoint_payload_mismatches", "duplicate_endpoint_failures",
        "cardinality_failures", "active_count_overflow", "truncation_events",
        "u000_mismatches", "u100_mismatches", "j100_mismatches",
        "nonfinite_active_values", "forbidden_branch_reads",
        "independent_sample_mismatches", "solver_optimum_failures",
    )
    if any(int(counters.get(name, -1)) != 0 for name in required_zero):
        raise ValueError("coupling full-role audit invariant failed")
    return with_content_hash({
        "contract": COUPLING_AUDIT_CONTRACT, "schema_version": 1,
        "coupling_config_sha256": require_sha256(coupling_config_sha256, name="coupling config"),
        "train_manifest_sha256": require_sha256(train_manifest_sha256, name="train coupling manifest"),
        "validation_manifest_sha256": require_sha256(validation_manifest_sha256, name="validation coupling manifest"),
        "expected_rows": dict(expected_rows), "observed_rows": dict(observed_rows),
        "counters": dict(counters),
        "endpoint_logical_sha256": {name: require_sha256(value, name=name) for name, value in endpoint_logical_sha256.items()},
        "branch_allowlist_sha256": require_sha256(branch_allowlist_sha256, name="branch allowlist"),
        "branch_access_trace_sha256": require_sha256(branch_access_trace_sha256, name="branch access trace"),
        "independent_sample_sha256": require_sha256(independent_sample_sha256, name="independent sample"),
        "complete_train_validation_coverage": True,
        "labels_read": False, "final_test_accessed": False,
    })


def validate_coupling_audit(value: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        value, expected_contract=COUPLING_AUDIT_CONTRACT,
        expected_schema_version=1,
    )
    if (
        value.get("expected_rows") != value.get("observed_rows")
        or value.get("complete_train_validation_coverage") is not True
        or value.get("labels_read") is not False
        or value.get("final_test_accessed") is not False
    ):
        raise ValueError("coupling audit coverage/access differs")
    if any(int(count) != 0 for count in value.get("counters", {}).values()):
        raise ValueError("coupling audit contains an invariant failure")
    for name in (
        "coupling_config_sha256", "train_manifest_sha256",
        "validation_manifest_sha256", "branch_allowlist_sha256",
        "branch_access_trace_sha256", "independent_sample_sha256",
        "switch_calibration_sha256",
    ):
        require_sha256(value.get(name), name=name)
    endpoints = value.get("endpoint_logical_sha256")
    if not isinstance(endpoints, Mapping) or set(endpoints) != {"p0", "d100", "hlt"}:
        raise ValueError("coupling audit endpoint hash registry differs")
    for name, endpoint_hash in endpoints.items():
        require_sha256(endpoint_hash, name=f"coupling audit endpoint {name}")
    solver = value.get("solver_audit")
    if (
        not isinstance(solver, Mapping)
        or solver.get("algorithm") != "scipy_hungarian_plus_canonical_edge_fixing_v1"
        or int(solver.get("rows", -1)) != sum(map(int, value["observed_rows"].values()))
        or int(solver.get("optimum_total_cost_q", -1)) < 0
    ):
        raise ValueError("coupling solver audit differs")
    require_sha256(solver.get("integer_matrix_sha256"), name="solver matrix")
    require_sha256(solver.get("selected_edge_tuple_sha256"), name="solver selection")
    return digest


def build_coupling_lock(
    *, campaign_spec_sha256: str, coupling_config_sha256: str,
    scale_calibration_sha256: str, switch_calibration_sha256: str,
    train_manifest_sha256: str, validation_manifest_sha256: str,
    audit_sha256: str,
) -> dict[str, Any]:
    hashes = {name: require_sha256(value, name=name) for name, value in locals().items()}
    return with_content_hash({
        "contract": COUPLING_LOCK_CONTRACT, "schema_version": 1,
        **hashes, "authorized": True, "complete_train_validation_coverage": True,
        "final_test_accessed": False,
    })


def validate_coupling_lock(
    value: Mapping[str, Any], *, campaign_spec_sha256: str | None = None,
    expected: Mapping[str, str] | None = None,
) -> str:
    digest = validate_content_hash(
        value, expected_contract=COUPLING_LOCK_CONTRACT, expected_schema_version=1,
    )
    for name in (
        "campaign_spec_sha256", "coupling_config_sha256",
        "scale_calibration_sha256", "switch_calibration_sha256",
        "train_manifest_sha256", "validation_manifest_sha256", "audit_sha256",
    ):
        require_sha256(value.get(name), name=name)
    if (
        value.get("authorized") is not True
        or value.get("complete_train_validation_coverage") is not True
        or value.get("final_test_accessed") is not False
    ):
        raise PermissionError("HCWDL-UJ coupling lock is incomplete")
    if (
        campaign_spec_sha256 is not None
        and value.get("campaign_spec_sha256")
        != require_sha256(campaign_spec_sha256, name="expected campaign specification")
    ):
        raise ValueError("HCWDL-UJ coupling lock campaign differs")
    if expected is not None:
        for name, expected_value in expected.items():
            if value.get(name) != require_sha256(
                expected_value, name=f"expected coupling-lock {name}",
            ):
                raise ValueError(f"HCWDL-UJ coupling lock {name} differs")
    return digest


__all__ = [
    "BASE_ARRAYS", "CouplingRow", "ResidualCouplingStore",
    "build_coupling_audit", "build_coupling_lock", "load_base_shard",
    "load_switch_sidecar", "publish_base_manifest", "publish_base_shard",
    "publish_coupling_manifest", "publish_switch_sidecar",
    "validate_base_arrays", "validate_base_manifest", "validate_coupling_audit",
    "validate_coupling_lock", "validate_coupling_manifest",
]
