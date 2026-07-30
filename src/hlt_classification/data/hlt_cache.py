"""Streaming construction and audit of deployable HLT-v3 cache shards."""

from __future__ import annotations

import json
from numbers import Integral
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .cache_contracts import (
    MANIFEST_FILENAME,
    array_sha256,
    build_cache_manifest,
    cache_spec_sha256,
    canonical_sha256,
    identity_key_array,
    load_completed_shard_record,
    load_json,
    publish_shard,
    require_sha256,
    source_files_sha256,
    validate_cache_manifest,
    validate_shard_record,
    write_immutable_json,
)
from .dataset import ShardedCacheDataset
from .hlt_v3 import (
    DEGRADATION_PROFILES,
    build_hlt_v3_view,
    degradation_profile,
    measurement_validity_states,
    validate_hlt_v3_profile_contract,
)
from .replicas import REALIZATION_POLICIES, validate_hlt_replica_manifest

ProgressCallback = Callable[[Mapping[str, Any]], None]
ShardCallback = Callable[[int, Mapping[str, Any]], None]


def _generator_source_sha256() -> str:
    source_root = Path(__file__).resolve().parent
    return source_files_sha256(
        {
            "cache_contracts.py": source_root / "cache_contracts.py",
            "dataset.py": source_root / "dataset.py",
            "hlt_cache.py": source_root / "hlt_cache.py",
            "hlt_v3.py": source_root / "hlt_v3.py",
            "replicas.py": source_root / "replicas.py",
        }
    )


def _emit(progress: ProgressCallback | None, **payload: Any) -> None:
    if progress is not None:
        progress(payload)


def _default_source_snapshot_sha256() -> str:
    module_root = Path(__file__).resolve().parent
    return source_files_sha256(
        {
            "cache_contracts.py": module_root / "cache_contracts.py",
            "dataset.py": module_root / "dataset.py",
            "hlt_cache.py": module_root / "hlt_cache.py",
            "hlt_v3.py": module_root / "hlt_v3.py",
            "replicas.py": module_root / "replicas.py",
        }
    )


def _build_report(
    *,
    complete: bool,
    root: Path,
    rows: int,
    new_shards: int,
    reused_shards: int,
    manifest: Mapping[str, Any] | None = None,
    next_shard_index: int | None = None,
) -> dict[str, Any]:
    return {
        "cache_kind": "hlt",
        "complete": complete,
        "output_dir": str(root),
        "rows": rows,
        "new_shards": new_shards,
        "reused_shards": reused_shards,
        "next_shard_index": next_shard_index,
        "manifest_sha256": None if manifest is None else manifest["content_hash"],
    }


def _source_range_digest(arrays: Mapping[str, np.ndarray]) -> str:
    return canonical_sha256(
        {
            name: array_sha256(name, arrays[name])
            for name in sorted(arrays)
        }
    )


def _summarize_diagnostics(
    diagnostics: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Retain aggregate performance diagnostics, never construction mappings."""

    summary: dict[str, Any] = {
        "event_count": len(diagnostics),
        "n_offline": 0,
        "n_output": 0,
        "mechanism_counts": {},
        "probability_sums": {},
        "type_input_counts": {},
        "type_output_counts": {},
        "measurement_states": {},
    }
    for diagnostic in diagnostics:
        summary["n_offline"] += int(diagnostic.get("n_offline", 0))
        summary["n_output"] += int(diagnostic.get("n_output", 0))
        for group in (
            "mechanism_counts",
            "probability_sums",
            "type_input_counts",
            "type_output_counts",
            "measurement_states",
        ):
            values = diagnostic.get(group, {})
            if not isinstance(values, Mapping):
                raise ValueError(f"HLT diagnostic group {group!r} is malformed")
            target = summary[group]
            for key, value in values.items():
                target[str(key)] = target.get(str(key), 0) + value
    return summary


def _combine_summaries(
    summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "event_count": 0,
        "n_offline": 0,
        "n_output": 0,
        "mechanism_counts": {},
        "probability_sums": {},
        "type_input_counts": {},
        "type_output_counts": {},
        "measurement_states": {},
    }
    for summary in summaries:
        for scalar in ("event_count", "n_offline", "n_output"):
            result[scalar] += int(summary.get(scalar, 0))
        for group in (
            "mechanism_counts",
            "probability_sums",
            "type_input_counts",
            "type_output_counts",
            "measurement_states",
        ):
            for key, value in dict(summary.get(group, {})).items():
                result[group][str(key)] = result[group].get(str(key), 0) + value
    return result


def _validate_hlt_arrays(
    arrays: Mapping[str, np.ndarray],
    source: Mapping[str, np.ndarray],
) -> None:
    if not np.array_equal(arrays["labels"], source["labels"]):
        raise ValueError("HLT cache labels differ from the offline parent")
    if not np.array_equal(arrays["identity_keys"], source["identity_keys"]):
        raise ValueError("HLT cache identity order differs from the offline parent")
    expected_states = measurement_validity_states(
        np.asarray(arrays["tokens"]), np.asarray(arrays["mask"])
    )
    if not np.array_equal(arrays["measurement_states"], expected_states):
        raise ValueError("HLT cache validity states differ from its degraded tokens")


def _expected_source_parents(
    *,
    offline: ShardedCacheDataset,
    offline_manifest_sha256: str,
    source: Mapping[str, np.ndarray],
    start: int,
    stop: int,
) -> dict[str, Any]:
    return {
        "offline_cache_manifest_sha256": offline_manifest_sha256,
        "source_array_content_sha256": _source_range_digest(source),
        "source_shards": [
            {
                "shard_index": int(parent["shard_index"]),
                "content_hash": parent["content_hash"],
                "shard_file_sha256": parent["shard_file_sha256"],
            }
            for parent in offline.shard_records_for_range(start, stop)
        ],
    }


def build_hlt_cache(
    offline_cache_dir: str | Path,
    *,
    output_dir: str | Path,
    profile_contract: Mapping[str, Any],
    replica_manifest: Mapping[str, Any],
    logical_role: str,
    replica_id: int,
    realization_policy: str = "R_MULTI",
    degradation_profile_id: str = "D_NOMINAL",
    profile_id: str | None = None,
    source_snapshot_sha256: str | None = None,
    shard_size: int = 4096,
    processing_batch_size: int = 256,
    max_new_shards: int | None = None,
    progress: ProgressCallback | None = None,
    on_shard_complete: ShardCallback | None = None,
) -> dict[str, Any]:
    """Build or exactly resume a deployable HLT cache from an offline parent."""

    if shard_size <= 0 or processing_batch_size <= 0:
        raise ValueError("shard and processing batch sizes must be positive")
    if max_new_shards is not None and max_new_shards < 0:
        raise ValueError("max_new_shards must be nonnegative")
    if profile_id is not None:
        if (
            degradation_profile_id != "D_NOMINAL"
            and degradation_profile_id != profile_id
        ):
            raise ValueError("profile_id and degradation_profile_id differ")
        degradation_profile_id = profile_id
    if realization_policy not in REALIZATION_POLICIES:
        raise ValueError(f"unknown HLT realization policy {realization_policy!r}")
    if degradation_profile_id not in DEGRADATION_PROFILES:
        raise ValueError(
            f"unknown degradation profile {degradation_profile_id!r}"
        )
    degradation_profile(degradation_profile_id)
    if (
        isinstance(replica_id, bool)
        or not isinstance(replica_id, Integral)
        or int(replica_id) not in range(4)
    ):
        raise ValueError("replica_id must be an integer in [0,3]")
    replica_id = int(replica_id)
    source_hash = require_sha256(
        (
            _default_source_snapshot_sha256()
            if source_snapshot_sha256 is None
            else source_snapshot_sha256
        ),
        name="source_snapshot_sha256",
    )
    profile_hash = validate_hlt_v3_profile_contract(profile_contract)
    replica_hash = validate_hlt_replica_manifest(replica_manifest)
    if profile_contract["hlt_replica_manifest_sha256"] != replica_hash:
        raise ValueError("HLT profile and replica-manifest lineage differ")

    offline = ShardedCacheDataset(
        offline_cache_dir,
        expected_cache_kind="offline",
        validate_shards=True,
    )
    offline_manifest = offline.manifest
    if offline.logical_role != logical_role:
        raise ValueError("offline cache role differs from requested HLT role")
    offline_hash = require_sha256(
        offline_manifest.get("content_hash"),
        name="offline_cache_manifest_sha256",
    )
    if (
        offline_manifest["lineage"].get("split_manifest_sha256")
        != replica_manifest["split_manifest_sha256"]
    ):
        raise ValueError("offline cache and HLT replica split lineage differ")
    if (
        offline_manifest["lineage"].get("raw_input_schema_sha256")
        != profile_contract["raw_input_schema_sha256"]
    ):
        raise ValueError("offline cache and HLT profile schema lineage differ")

    generator = {
        "qualified_name": "hlt_classification.data.hlt_cache.build_hlt_cache",
        "source_sha256": _generator_source_sha256(),
    }
    build_intent = {
        "cache_kind": "hlt",
        "logical_role": logical_role,
        "replica_id": replica_id,
        "realization_policy": realization_policy,
        "degradation_profile_id": degradation_profile_id,
        "shard_size": shard_size,
        "total_rows": len(offline),
        "identity_order_sha256": offline_manifest["identity_order_sha256"],
        "source_snapshot_sha256": source_hash,
        "offline_cache_manifest_sha256": offline_hash,
        "split_manifest_sha256": replica_manifest["split_manifest_sha256"],
        "raw_input_schema_sha256": profile_contract["raw_input_schema_sha256"],
        "hlt_profile_contract_sha256": profile_hash,
        "hlt_replica_manifest_sha256": replica_hash,
        "generator": generator,
    }
    specification_hash = cache_spec_sha256(build_intent)
    lineage = {
        "cache_spec_sha256": specification_hash,
        "degradation_profile_id": degradation_profile_id,
        "generator": generator,
        "hlt_profile_contract_sha256": profile_hash,
        "hlt_replica_manifest_sha256": replica_hash,
        "offline_cache_manifest_sha256": offline_hash,
        "raw_input_schema_sha256": profile_contract["raw_input_schema_sha256"],
        "realization_policy": realization_policy,
        "replica_id": replica_id,
        "source_snapshot_sha256": source_hash,
        "split_manifest_sha256": replica_manifest["split_manifest_sha256"],
    }
    root = Path(output_dir)
    manifest_path = root / MANIFEST_FILENAME
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        validate_cache_manifest(
            manifest,
            cache_root=root,
            expected_cache_kind="hlt",
            expected_role=logical_role,
            expected_lineage=lineage,
        )
        for record in manifest["shards"]:
            start, stop = int(record["row_start"]), int(record["row_stop"])
            source = offline.read_range(start, stop)
            identity_keys = [
                str(value) for value in source["identity_keys"].tolist()
            ]
            arrays = validate_shard_record(
                record,
                cache_root=root,
                expected_cache_kind="hlt",
                expected_role=logical_role,
                expected_lineage=lineage,
                expected_identity_keys=identity_keys,
            )
            expected_parents = _expected_source_parents(
                offline=offline,
                offline_manifest_sha256=offline_hash,
                source=source,
                start=start,
                stop=stop,
            )
            if record.get("parents") != expected_parents:
                raise ValueError("completed HLT shard parent lineage differs")
            _validate_hlt_arrays(arrays, source)
        _emit(progress, event="cache_reused", path=str(root), rows=len(offline))
        return _build_report(
            complete=True,
            root=root,
            rows=len(offline),
            new_shards=0,
            reused_shards=int(manifest["shard_count"]),
            manifest=manifest,
        )

    records: list[dict[str, Any]] = []
    summaries: list[Mapping[str, Any]] = []
    new_shards = 0
    reused_shards = 0
    total_shards = (len(offline) + shard_size - 1) // shard_size
    for shard_index, start in enumerate(range(0, len(offline), shard_size)):
        stop = min(start + shard_size, len(offline))
        source = offline.read_range(start, stop)
        identity_keys = [
            str(value) for value in source["identity_keys"].tolist()
        ]
        expected_parents = _expected_source_parents(
            offline=offline,
            offline_manifest_sha256=offline_hash,
            source=source,
            start=start,
            stop=stop,
        )
        existing = load_completed_shard_record(root, shard_index)
        status = "reused"
        if existing is not None:
            if (
                int(existing["row_start"]) != start
                or int(existing["row_stop"]) != stop
            ):
                raise ValueError("completed HLT shard range differs")
            arrays = validate_shard_record(
                existing,
                cache_root=root,
                expected_cache_kind="hlt",
                expected_role=logical_role,
                expected_lineage=lineage,
                expected_identity_keys=identity_keys,
            )
            if existing.get("parents") != expected_parents:
                raise ValueError("completed HLT shard parent lineage differs")
            _validate_hlt_arrays(arrays, source)
            record = existing
            summary = dict(record.get("diagnostics", {}))
            reused_shards += 1
        else:
            if max_new_shards is not None and new_shards >= max_new_shards:
                _emit(
                    progress,
                    event="cache_interrupted",
                    completed_shards=len(records),
                    rows_complete=start,
                )
                return _build_report(
                    complete=False,
                    root=root,
                    rows=len(offline),
                    new_shards=new_shards,
                    reused_shards=reused_shards,
                    next_shard_index=shard_index,
                )
            output_tokens: list[np.ndarray] = []
            output_masks: list[np.ndarray] = []
            output_states: list[np.ndarray] = []
            event_diagnostics: list[Mapping[str, Any]] = []
            for local_start in range(0, stop - start, processing_batch_size):
                local_stop = min(
                    local_start + processing_batch_size,
                    stop - start,
                )
                tokens, mask, states, batch_diagnostics = build_hlt_v3_view(
                    source["tokens"][local_start:local_stop],
                    source["mask"][local_start:local_stop],
                    canonical_identities=identity_keys[local_start:local_stop],
                    logical_role=logical_role,
                    replica_id=replica_id,
                    realization_policy=realization_policy,
                    profile_id=degradation_profile_id,
                )
                output_tokens.append(tokens)
                output_masks.append(mask)
                output_states.append(states)
                event_diagnostics.extend(batch_diagnostics)
            arrays = {
                "tokens": np.ascontiguousarray(
                    np.concatenate(output_tokens, axis=0), dtype=np.float32
                ),
                "mask": np.ascontiguousarray(
                    np.concatenate(output_masks, axis=0), dtype=np.bool_
                ),
                "labels": np.ascontiguousarray(source["labels"], dtype=np.int64),
                "identity_keys": identity_key_array(identity_keys),
                "measurement_states": np.ascontiguousarray(
                    np.concatenate(output_states, axis=0), dtype=np.int8
                ),
            }
            _validate_hlt_arrays(arrays, source)
            summary = _summarize_diagnostics(event_diagnostics)
            record, status = publish_shard(
                cache_root=root,
                cache_kind="hlt",
                logical_role=logical_role,
                shard_index=shard_index,
                row_start=start,
                row_stop=stop,
                arrays=arrays,
                lineage=lineage,
                parents=expected_parents,
                diagnostics=summary,
            )
            new_shards += 1
        records.append(record)
        summaries.append(summary)
        _emit(
            progress,
            event="shard_complete",
            shard_index=shard_index,
            shard_count=total_shards,
            row_start=start,
            row_stop=stop,
            status=status,
        )
        if on_shard_complete is not None:
            on_shard_complete(shard_index, record)

    manifest = build_cache_manifest(
        cache_kind="hlt",
        logical_role=logical_role,
        shard_size=shard_size,
        total_rows=len(offline),
        identity_keys=None,
        identity_order_digest=offline_manifest["identity_order_sha256"],
        lineage=lineage,
        shard_records=records,
        aggregate_diagnostics=_combine_summaries(summaries),
    )
    write_immutable_json(manifest_path, manifest)
    validate_cache_manifest(
        manifest,
        cache_root=root,
        expected_cache_kind="hlt",
        expected_role=logical_role,
        expected_lineage=lineage,
    )
    _emit(progress, event="cache_complete", path=str(root), rows=len(offline))
    return _build_report(
        complete=True,
        root=root,
        rows=len(offline),
        new_shards=new_shards,
        reused_shards=reused_shards,
        manifest=manifest,
    )


def audit_hlt_cache(
    cache_dir: str | Path,
    *,
    expected_profile_contract_sha256: str | None = None,
    expected_replica_manifest_sha256: str | None = None,
    expected_degradation_profile_id: str | None = None,
) -> dict[str, Any]:
    """Perform full shard authentication and optional exact lineage checks."""

    dataset = ShardedCacheDataset(
        cache_dir,
        expected_cache_kind="hlt",
        validate_shards=True,
    )
    lineage = dataset.manifest["lineage"]
    for _, arrays in dataset.iter_shards():
        expected_states = measurement_validity_states(
            np.asarray(arrays["tokens"]),
            np.asarray(arrays["mask"]),
        )
        if not np.array_equal(arrays["measurement_states"], expected_states):
            raise ValueError(
                "HLT cache validity states differ from degraded tokens"
            )
    serialized = json.dumps(dataset.manifest, sort_keys=True, allow_nan=False)
    forbidden = (
        "canonical_output_indices",
        "construction_indices",
        "output_indices",
    )
    leaked = [name for name in forbidden if name in serialized]
    if leaked:
        raise ValueError(f"deployable HLT cache leaks construction indices: {leaked}")
    expected = {
        "hlt_profile_contract_sha256": expected_profile_contract_sha256,
        "hlt_replica_manifest_sha256": expected_replica_manifest_sha256,
        "degradation_profile_id": expected_degradation_profile_id,
    }
    for key, value in expected.items():
        if value is None:
            continue
        if key.endswith("_sha256"):
            require_sha256(value, name=key)
        if lineage.get(key) != value:
            raise ValueError(f"HLT cache {key} differs from expectation")
    return {
        "ok": True,
        "cache_manifest_sha256": dataset.manifest["content_hash"],
        "logical_role": dataset.logical_role,
        "rows": len(dataset),
        "shards": int(dataset.manifest["shard_count"]),
        "lineage": dict(lineage),
        "aggregate_diagnostics": dict(
            dataset.manifest.get("aggregate_diagnostics", {})
        ),
    }


__all__ = ["audit_hlt_cache", "build_hlt_cache"]
