"""Version-aware assignment-store dispatch without changing old artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hlt_classification.data.cache_contracts import load_json

from .hcwdl_fullcard_bottleneck_cache import (
    FullCardinalityAssignmentStore,
    load_assignment_shard as load_fullcard_assignment_shard,
)
from .hcwdl_fullcard_bottleneck_contracts import (
    ASSIGNMENT_MANIFEST_CONTRACT as FULLCARD_MANIFEST_CONTRACT,
    ASSIGNMENT_SHARD_CONTRACT as FULLCARD_SHARD_CONTRACT,
)
from .highcov_cache import (
    MANIFEST_CONTRACT as HIGHCOV_MANIFEST_CONTRACT,
    SHARD_CONTRACT as HIGHCOV_SHARD_CONTRACT,
    DenseAssignmentStore,
    load_assignment_shard as load_highcov_assignment_shard,
)


def open_assignment_store(path: str | Path) -> Any:
    contract = load_json(path).get("contract")
    if contract == HIGHCOV_MANIFEST_CONTRACT:
        return DenseAssignmentStore(path)
    if contract == FULLCARD_MANIFEST_CONTRACT:
        return FullCardinalityAssignmentStore(path)
    raise ValueError(f"unsupported assignment manifest contract: {contract!r}")


def load_assignment_shard(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = load_json(path).get("contract")
    if contract == HIGHCOV_SHARD_CONTRACT:
        return load_highcov_assignment_shard(path)
    if contract == FULLCARD_SHARD_CONTRACT:
        return load_fullcard_assignment_shard(path)
    raise ValueError(f"unsupported assignment shard contract: {contract!r}")


__all__ = ["load_assignment_shard", "open_assignment_store"]
