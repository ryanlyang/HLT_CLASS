"""Closed registry of graph identities accepted by representation runtimes."""

from __future__ import annotations

from typing import Final

from hlt_classification.data.cache_contracts import require_sha256

from .hcwdl_direct_offline_kd_graph import GRAPH_SHA256 as DIRECT_GRAPH_SHA256
from .hcwdl_homotopy_representation_graph import (
    GRAPH_SHA256 as HOMOTOPY_RKD_GRAPH_SHA256,
)
from .hcwdl_representation_graph import ASCENT_GRAPH_SHA256


REGISTERED_GRAPH_SHA256S: Final = frozenset({
    ASCENT_GRAPH_SHA256,
    HOMOTOPY_RKD_GRAPH_SHA256,
    DIRECT_GRAPH_SHA256,
})


def registered_graph_sha256(execution_id: str) -> str:
    """Resolve the one graph contract authorized for an execution."""

    if execution_id.startswith(("F_RSET_", "F_RREL_")):
        return HOMOTOPY_RKD_GRAPH_SHA256
    if execution_id in {"HLT_RSET", "HLT_RREL"}:
        return DIRECT_GRAPH_SHA256
    return ASCENT_GRAPH_SHA256


def validate_registered_graph_sha256(value: str) -> str:
    """Require one exact graph identity from the closed runtime registry."""

    digest = require_sha256(value, name="registered representation graph SHA-256")
    if digest not in REGISTERED_GRAPH_SHA256S:
        raise ValueError("representation graph is not registered")
    return digest


__all__ = [
    "DIRECT_GRAPH_SHA256",
    "HOMOTOPY_RKD_GRAPH_SHA256",
    "REGISTERED_GRAPH_SHA256S",
    "registered_graph_sha256",
    "validate_registered_graph_sha256",
]
