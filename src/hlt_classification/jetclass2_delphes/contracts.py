"""Versioned local data contracts; scientific identities exclude machine paths."""
from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Mapping

from hlt_classification.data.cache_contracts import (
    canonical_sha256, validate_content_hash, with_content_hash,
)

PREFIX = "JETCLASS2_DELPHES_"


def artifact(kind: str, *, version: int = 1, **fields) -> dict:
    return with_content_hash(dict(
        contract=f"{PREFIX}{kind}/v{version}", schema_version=version, **fields,
    ))


def validate(value: Mapping, kind: str, *, version: int = 1) -> str:
    return validate_content_hash(
        value, expected_contract=f"{PREFIX}{kind}/v{version}", expected_schema_version=version,
    )


def relative_file(root: Path, name: str) -> Path:
    """Resolve manifest paths without admitting escapes or platform ambiguity."""
    rel = PurePosixPath(name)
    if (not name or "\\" in name or ":" in name or rel.is_absolute()
            or ".." in rel.parts or rel.as_posix() != name):
        raise ValueError(f"Unsafe dataset-relative path: {name!r}")
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Dataset path escapes root: {name}")
    return path


def assumptions() -> dict:
    return artifact(
        "ASSUMPTIONS",
        status="user_authorized_provisional_not_producer_confirmed",
        raw_label_mapping="standard_upstream_zero_based_labels_unchanged",
        zero_error_policy="unavailable_uncertainty_preserve_finite_displacement",
        negative_error_policy="reject",
        generation_independence="whole_file_groups_cross_file_independence_unverified",
        producer_revision=None, producer_cards=None,
        public_release_permission="not_established_by_this_artifact",
    )


def row_identity(inventory_hash: str, relative_path: str, tree_key: str, entry: int) -> str:
    return canonical_sha256({
        "domain": "JETCLASS2_DELPHES_ROW/v1", "inventory": inventory_hash,
        "file": relative_path, "tree": tree_key, "entry": int(entry),
    })
