"""Label-blind nested populations derived from an authenticated foundation."""
from __future__ import annotations

from pathlib import Path
import hashlib
import heapq

import numpy as np

from .contracts import artifact, validate
from .data import load_assignments, validate_foundation

DIRECT_COARSE_COUNTS = {"train": 100_000, "validation": 50_000}
DIRECT_COARSE_SELECTION_DOMAIN = (
    "JETCLASS2_CMS_PROXY_LADDER_100K_50K_FROM_200K_100K/v1"
)


def _identity_digest(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for row in values:
        digest.update(bytes(row))
    return digest.hexdigest()


def _rank(foundation_sha256: str, role: str, identity: bytes) -> tuple[bytes, bytes]:
    payload = b"\0".join((
        DIRECT_COARSE_SELECTION_DOMAIN.encode(),
        foundation_sha256.encode(),
        role.encode(),
        identity,
    ))
    return hashlib.sha256(payload).digest(), identity


def _selected_ordinals(
    foundation: dict, *, foundation_root: Path, role: str, count: int,
) -> np.ndarray:
    identities = load_assignments(
        foundation, root=Path(foundation_root), role=role,
    )[0]
    if type(count) is not int or not 0 < count <= len(identities):
        raise ValueError("Invalid proxy-ladder nested-population count")
    chosen = heapq.nsmallest(
        count,
        range(len(identities)),
        key=lambda index: _rank(
            foundation["content_hash"], role, bytes(identities[index]),
        ),
    )
    return np.asarray(sorted(chosen), np.int64)


def create_direct_coarse_population(
    foundation: dict, *, foundation_root: Path,
) -> dict:
    """Select an exact nested 100k/50k population without reading labels."""
    root = Path(foundation_root).resolve(strict=True)
    validate_foundation(foundation, root=root)
    digests = {}
    for role, count in DIRECT_COARSE_COUNTS.items():
        identities = load_assignments(foundation, root=root, role=role)[0]
        selected = _selected_ordinals(
            foundation, foundation_root=root, role=role, count=count,
        )
        digests[role] = _identity_digest(identities[selected])
    value = artifact(
        "POPULATION_SELECTION",
        parents={"foundation": foundation["content_hash"]},
        selection_domain=DIRECT_COARSE_SELECTION_DOMAIN,
        source_counts=dict(foundation["role_counts"]),
        counts=dict(DIRECT_COARSE_COUNTS),
        identity_sha256=digests,
        total_rows=sum(DIRECT_COARSE_COUNTS.values()),
        method="smallest_sha256_rank_then_preserve_foundation_role_order",
        labels_read=False,
        selection_depends_on_labels=False,
        final_test_in_selection=False,
    )
    validate_direct_coarse_population(value, foundation=foundation, foundation_root=root)
    return value


def validate_direct_coarse_population(
    value: dict, *, foundation: dict, foundation_root: Path,
) -> str:
    root = Path(foundation_root).resolve(strict=True)
    validate_foundation(foundation, root=root)
    digest = validate(
        value, "POPULATION_SELECTION",
        parents={"foundation": foundation["content_hash"]},
    )
    if (
        value["selection_domain"] != DIRECT_COARSE_SELECTION_DOMAIN
        or value["source_counts"] != foundation["role_counts"]
        or value["counts"] != DIRECT_COARSE_COUNTS
        or value["total_rows"] != sum(DIRECT_COARSE_COUNTS.values())
        or value["method"]
        != "smallest_sha256_rank_then_preserve_foundation_role_order"
        or value["labels_read"] is not False
        or value["selection_depends_on_labels"] is not False
        or value["final_test_in_selection"] is not False
    ):
        raise ValueError("Proxy-ladder nested-population semantics differ")
    for role, count in DIRECT_COARSE_COUNTS.items():
        identities = load_assignments(foundation, root=root, role=role)[0]
        selected = _selected_ordinals(
            foundation, foundation_root=root, role=role, count=count,
        )
        if _identity_digest(identities[selected]) != value["identity_sha256"][role]:
            raise ValueError("Proxy-ladder nested-population identities differ")
    return digest


def selection_mask(
    value: dict, *, foundation: dict, foundation_root: Path, role: str,
) -> np.ndarray:
    validate_direct_coarse_population(
        value, foundation=foundation, foundation_root=foundation_root,
    )
    if role not in DIRECT_COARSE_COUNTS:
        raise PermissionError("Proxy-ladder nested population has no final-test role")
    mask = np.zeros(foundation["role_counts"][role], dtype=np.bool_)
    mask[_selected_ordinals(
        foundation, foundation_root=foundation_root, role=role,
        count=value["counts"][role],
    )] = True
    return mask


__all__ = [
    "DIRECT_COARSE_COUNTS", "DIRECT_COARSE_SELECTION_DOMAIN",
    "create_direct_coarse_population", "selection_mask",
    "validate_direct_coarse_population",
]
