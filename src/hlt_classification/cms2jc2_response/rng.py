"""Versioned counter-based common random numbers; no global RNG state."""
from __future__ import annotations

import hashlib
import struct
import numpy as np
from scipy.special import ndtri

from .contracts import SEED

DOMAIN = "CMS2JC2_RESPONSE_RNG/v1"
COMPONENTS = frozenset({"topology", "survival", "identity", "kinematics", "tracking",
                        "shared_jet", "additional_count", "additional_object"})


def _digest(*fields: str) -> bytes:
    h = hashlib.sha256()
    for field in fields:
        encoded = field.encode("utf-8")
        h.update(struct.pack(">Q", len(encoded)))
        h.update(encoded)
    return h.digest()


def uniform(jet: str, replica: int, component: str, object_key: str, counter: int = 0) -> float:
    if not jet or not isinstance(jet, str) or not isinstance(object_key, str):
        raise ValueError("Canonical jet and object keys are required")
    if component not in COMPONENTS or any(type(x) is not int or x < 0 for x in (replica, counter)):
        raise ValueError("Invalid random component/replica/counter")
    bits = int.from_bytes(_digest(DOMAIN, str(SEED), jet, str(replica), component,
                                  object_key, str(counter))[:8], "big")
    return ((bits >> 12) + .5) / (2 ** 52)


def normals(jet: str, replica: int, component: str, object_key: str, count: int) -> np.ndarray:
    if type(count) is not int or count < 0:
        raise ValueError("Invalid normal-vector dimension")
    return ndtri([uniform(jet, replica, component, object_key, i) for i in range(count)])


def group_key(ids) -> str:
    keys = sorted(str(x) for x in ids)
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("Group keys must be nonempty and distinct")
    return _digest("CMS2JC2_GROUP/v1", *keys).hex()


def categorical(probabilities, jet: str, replica: int, component: str, object_key: str) -> int:
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 1 or len(p) == 0 or not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError("Invalid categorical probabilities")
    if not np.isclose(p.sum(), 1., rtol=0, atol=1e-10):
        raise ValueError("Categorical probabilities must sum to one")
    return min(int(np.searchsorted(np.cumsum(p), uniform(jet, replica, component, object_key),
                                    side="right")), len(p) - 1)

