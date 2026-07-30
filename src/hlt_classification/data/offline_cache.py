"""Restartable, immutable, bounded-memory offline JetClass cache builder."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from .cache_contracts import (
    build_cache_manifest,
    cache_spec_sha256,
    canonical_sha256,
    identity_key_array,
    identity_order_sha256,
    load_completed_shard_record,
    load_json,
    publish_shard,
    require_sha256,
    source_files_sha256,
    validate_cache_arrays,
    validate_cache_manifest,
    validate_shard_record,
    write_immutable_json,
)
from .root_reader import JetView, load_offline_view
from .schema import SCHEMA_CONTRACT, SCHEMA_VERSION, schema_payload
from .splits import SplitManifest, audit_split_manifest

OFFLINE_CACHE_GENERATOR_CONTRACT = "hlt_classification_offline_cache_generator_v1"


def _validate_offline_arrays(
    arrays: dict[str, np.ndarray],
    *,
    expected_keys: list[str] | tuple[str, ...] | None = None,
    expected_labels: np.ndarray | None = None,
) -> None:
    validate_cache_arrays(
        arrays,
        cache_kind="offline",
        expected_identity_keys=expected_keys,
    )
    if expected_labels is not None and not np.array_equal(
        arrays["labels"], expected_labels
    ):
        raise ValueError("offline cache labels differ from the split manifest")


def offline_cache_lineage(
    manifest: SplitManifest,
    *,
    source_snapshot_sha256: str | None = None,
) -> dict[str, object]:
    manifest.verify_hash()
    source_root = Path(__file__).resolve().parent
    generator_source_sha256 = source_files_sha256(
        {
            "cache_contracts.py": source_root / "cache_contracts.py",
            "offline_cache.py": source_root / "offline_cache.py",
            "root_reader.py": source_root / "root_reader.py",
        }
    )
    return {
        "generator_contract": OFFLINE_CACHE_GENERATOR_CONTRACT,
        "generator_source_sha256": generator_source_sha256,
        "source_snapshot_sha256": require_sha256(
            (
                generator_source_sha256
                if source_snapshot_sha256 is None
                else source_snapshot_sha256
            ),
            name="source_snapshot_sha256",
        ),
        "split_manifest_sha256": manifest.content_hash,
        "raw_schema_contract": SCHEMA_CONTRACT,
        "raw_schema_version": SCHEMA_VERSION,
        "raw_input_schema_sha256": canonical_sha256(schema_payload()),
        "tree_name": manifest.tree_name,
        "max_constituents": manifest.max_constituents,
    }


def build_offline_cache(
    manifest: SplitManifest,
    *,
    logical_role: str,
    output_dir: str | Path,
    data_root: str | Path | None = None,
    shard_size: int = 4096,
    read_chunk_size: int = 4096,
    source_snapshot_sha256: str | None = None,
    max_new_shards: int | None = None,
    view_loader: Callable[..., JetView] = load_offline_view,
    progress: Callable[[str], None] | None = print,
) -> dict[str, object]:
    """Build or exactly resume one role cache.

    ``max_new_shards`` is a test/operations interruption hook. A partial run
    publishes only complete authenticated shards and intentionally withholds
    the final manifest.
    """

    if shard_size <= 0 or read_chunk_size <= 0:
        raise ValueError("shard and read chunk sizes must be positive")
    if max_new_shards is not None and max_new_shards < 0:
        raise ValueError("max_new_shards must be non-negative")
    audit = audit_split_manifest(manifest)
    if not audit["ok"]:
        raise ValueError(f"split manifest failed audit: {audit}")
    if logical_role not in manifest.splits:
        raise ValueError(f"unknown split role {logical_role!r}")

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    lineage = offline_cache_lineage(
        manifest,
        source_snapshot_sha256=source_snapshot_sha256,
    )
    identities = manifest.splits[logical_role]
    identity_keys = [identity.key for identity in identities]
    lineage["cache_spec_sha256"] = cache_spec_sha256(
        {
            "cache_kind": "offline",
            "logical_role": logical_role,
            "shard_size": shard_size,
            "total_rows": len(identity_keys),
            "identity_order_sha256": identity_order_sha256(identity_keys),
            "lineage": lineage,
        }
    )
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        payload = load_json(manifest_path)
        digest = validate_cache_manifest(
            payload,
            cache_root=root,
            expected_cache_kind="offline",
            expected_role=logical_role,
            expected_lineage=lineage,
            expected_identity_keys=identity_keys,
            validate_shards=True,
        )
        for record in payload["shards"]:
            start, stop = int(record["row_start"]), int(record["row_stop"])
            arrays = validate_shard_record(
                record,
                cache_root=root,
                expected_cache_kind="offline",
                expected_role=logical_role,
                expected_lineage=lineage,
                expected_identity_keys=identity_keys[start:stop],
            )
            _validate_offline_arrays(
                arrays,
                expected_keys=identity_keys[start:stop],
                expected_labels=np.asarray(
                    [identity.label for identity in identities[start:stop]],
                    dtype=np.int64,
                ),
            )
        return {
            "complete": True,
            "new_shards": 0,
            "reused_shards": int(payload["shard_count"]),
            "manifest_sha256": digest,
        }

    shard_records: list[dict[str, object]] = []
    new_shards = 0
    reused_shards = 0
    for shard_index, start in enumerate(range(0, len(identities), shard_size)):
        stop = min(start + shard_size, len(identities))
        expected_keys = identity_keys[start:stop]
        record = load_completed_shard_record(root, shard_index)
        if record is not None:
            if int(record.get("row_start", -1)) != start or int(
                record.get("row_stop", -1)
            ) != stop:
                raise ValueError("reusable offline shard range differs")
            reused_arrays = validate_shard_record(
                record,
                cache_root=root,
                expected_cache_kind="offline",
                expected_role=logical_role,
                expected_lineage=lineage,
                expected_identity_keys=expected_keys,
            )
            _validate_offline_arrays(
                reused_arrays,
                expected_keys=expected_keys,
                expected_labels=np.asarray(
                    [identity.label for identity in identities[start:stop]],
                    dtype=np.int64,
                ),
            )
            shard_records.append(record)
            reused_shards += 1
            continue
        if max_new_shards is not None and new_shards >= max_new_shards:
            return {
                "complete": False,
                "new_shards": new_shards,
                "reused_shards": reused_shards,
                "next_shard_index": shard_index,
            }
        view = view_loader(
            identities[start:stop],
            data_root=manifest.data_root if data_root is None else data_root,
            tree_name=manifest.tree_name,
            max_constituents=manifest.max_constituents,
            read_chunk_size=read_chunk_size,
            verify_label_branches=True,
        )
        actual_keys = [identity.key for identity in view.identities]
        if actual_keys != expected_keys:
            raise RuntimeError("ROOT cache reader returned identities out of order")
        expected_labels = np.asarray(
            [identity.label for identity in identities[start:stop]], dtype=np.int64
        )
        if not np.array_equal(view.labels, expected_labels):
            raise RuntimeError("ROOT cache reader labels differ from split identities")
        arrays = {
            "tokens": np.asarray(view.tokens, dtype=np.float32),
            "mask": np.asarray(view.mask, dtype=np.bool_),
            "labels": np.asarray(view.labels, dtype=np.int64),
            "identity_keys": identity_key_array(expected_keys),
        }
        _validate_offline_arrays(
            arrays,
            expected_keys=expected_keys,
            expected_labels=expected_labels,
        )
        record, _ = publish_shard(
            cache_root=root,
            cache_kind="offline",
            logical_role=logical_role,
            shard_index=shard_index,
            row_start=start,
            row_stop=stop,
            arrays=arrays,
            lineage=lineage,
            diagnostics={
                "source_files_read": int(view.stats.files_read),
                "source_chunks_read": int(view.stats.chunks_read),
            },
        )
        shard_records.append(record)
        new_shards += 1
        if progress is not None:
            progress(
                f"offline cache {logical_role}: shard {shard_index + 1}/"
                f"{(len(identities) + shard_size - 1) // shard_size}"
            )

    payload = build_cache_manifest(
        cache_kind="offline",
        logical_role=logical_role,
        shard_size=shard_size,
        total_rows=len(identities),
        identity_keys=identity_keys,
        lineage=lineage,
        shard_records=shard_records,
    )
    write_immutable_json(manifest_path, payload)
    digest = validate_cache_manifest(
        payload,
        cache_root=root,
        expected_cache_kind="offline",
        expected_role=logical_role,
        expected_lineage=lineage,
        expected_identity_keys=identity_keys,
        validate_shards=True,
    )
    return {
        "complete": True,
        "new_shards": new_shards,
        "reused_shards": reused_shards,
        "manifest_sha256": digest,
    }


def validate_offline_cache(
    cache_dir: str | Path,
    *,
    expected_role: str | None = None,
    expected_lineage: dict[str, object] | None = None,
) -> dict[str, object]:
    """Validate a complete offline cache without loading it all into RAM."""

    root = Path(cache_dir)
    manifest = load_json(root / "manifest.json")
    digest = validate_cache_manifest(
        manifest,
        cache_root=root,
        expected_cache_kind="offline",
        expected_role=expected_role,
        expected_lineage=expected_lineage,
        validate_shards=True,
    )
    for record in manifest["shards"]:
        arrays = validate_shard_record(
            record,
            cache_root=root,
            expected_cache_kind="offline",
            expected_role=str(manifest["logical_role"]),
            expected_lineage=dict(manifest["lineage"]),
        )
        _validate_offline_arrays(arrays)
    return {
        "ok": True,
        "manifest_sha256": digest,
        "rows": int(manifest["total_rows"]),
        "shards": int(manifest["shard_count"]),
        "logical_role": str(manifest["logical_role"]),
        "lineage": dict(manifest["lineage"]),
    }


__all__ = [
    "OFFLINE_CACHE_GENERATOR_CONTRACT",
    "build_offline_cache",
    "offline_cache_lineage",
    "validate_offline_cache",
]
